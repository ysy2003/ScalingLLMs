"""
Comprehensive evaluation script for Design2Code models.
Evaluates Qwen3-VL-2B and VLM_WebSight_finetuned on the full Design2Code dataset.
Uses all applicable metrics from the metrics package.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from PIL import Image
import pandas as pd
import asyncio
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from metrics import (
    compute_efficiency_local,
    semantic_html_usage,
    accessibility_score,
    EfficiencyScores
)


# ============================================================================
# Model Loading
# ============================================================================

def load_qwen_model(device):
    """Load Qwen3-VL-2B-Instruct model and processor."""
    print("Loading Qwen3-VL-2B-Instruct...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.float16,
        cache_dir="/root/"
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        cache_dir="/root/"
    )
    model.to(device)
    model.eval()
    return model, processor


def load_websight_model(device):
    """Load VLM_WebSight_finetuned model and processor."""
    print("Loading VLM_WebSight_finetuned...")
    local_dir = "../models--HuggingFaceM4--VLM_WebSight_finetuned/snapshots/a5c2b06bfee0bd713cf2a6b3e4d46f94dd8fe839/"

    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, processor


# ============================================================================
# Generation Functions
# ============================================================================

def generate_qwen(image, model, processor, prompt, device):
    """Generate HTML/CSS using Qwen3-VL-2B-Instruct."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True
    )

    latency = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated()
    num_tokens = output_ids.shape[1] - inputs['input_ids'].shape[1]

    generated_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return generated_text, latency, peak_vram, num_tokens


def convert_to_rgb(image):
    """Convert image to RGB."""
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")


def generate_websight(image, model, processor, device):
    """Generate HTML/CSS using VLM_WebSight_finetuned."""
    image = convert_to_rgb(image)

    x = to_numpy_array(image)
    x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
    x = processor.image_processor.rescale(x, scale=1 / 255)
    x = processor.image_processor.normalize(
        x,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    pixel_values = torch.tensor(x).unsqueeze(0).to(device)

    image_seq_len = model.config.perceiver_config.resampler_n_latents
    BOS = processor.tokenizer.bos_token
    bad_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"],
        add_special_tokens=False
    ).input_ids

    inputs = processor.tokenizer(
        f"{BOS}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
        return_tensors="pt",
        add_special_tokens=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = pixel_values

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    ids = model.generate(
        **inputs,
        bad_words_ids=bad_ids,
        max_length=2048,
        use_cache=False
    )

    latency = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated()
    num_tokens = ids.shape[1]

    generated_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    return generated_text, latency, peak_vram, num_tokens


# ============================================================================
# Metric Computation - Using metrics package
# ============================================================================

class SimpleNode:
    """Simple DOM node for metric computation."""
    def __init__(self, tag_name, attrs, children):
        self.tag = tag_name
        self.attrs = attrs
        self.children = children


def soup_to_node(element):
    """Convert BeautifulSoup element to SimpleNode."""
    if isinstance(element, str):
        return None
    tag_name = element.name if hasattr(element, 'name') else ''
    attrs = dict(element.attrs) if hasattr(element, 'attrs') else {}
    children = [soup_to_node(child) for child in element.children if hasattr(child, 'name')]
    children = [c for c in children if c is not None]
    return SimpleNode(tag_name, attrs, children)


def compute_structural_metrics(html_string):
    """Compute structural alignment metrics using metrics package."""
    soup = BeautifulSoup(html_string, 'html.parser')
    dom = soup_to_node(soup)

    semantic_ratio = semantic_html_usage(dom) if dom else 0.0
    acc_score = accessibility_score(dom) if dom else 0.0

    return {
        'semantic_html_ratio': semantic_ratio,
        'accessibility_score': acc_score
    }


async def check_html_correctness(html_path):
    """Check HTML file for render success and errors using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        errors = []
        render_success = False

        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        page.on('pageerror', lambda err: errors.append(err.message))

        file_url = f'file://{Path(html_path).resolve()}'
        await page.goto(file_url, wait_until='load')

        body_content = await page.evaluate("() => document.body.innerHTML")
        if body_content and len(body_content.strip()) > 0:
            render_success = True

        await context.close()
        await browser.close()

        return {
            'render_success': render_success,
            'error_count': len(errors)
        }


def save_html_file(html_string, output_path):
    """Save generated HTML to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_string, encoding='utf-8')


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def evaluate_model(model, processor, dataset, model_name, generate_fn, device, prompt, output_dir):
    """Evaluate a single model on the dataset using all applicable metrics."""
    results = []

    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")

    for idx, sample in enumerate(dataset):
        print(f"\nProcessing sample {idx + 1}/{len(dataset)}...")

        image = sample['image'].convert('RGB')

        # Generate HTML/CSS
        if model_name == "Qwen3-VL-2B":
            generated_html, latency, peak_vram, num_tokens = generate_fn(
                image, model, processor, prompt, device
            )
        else:
            generated_html, latency, peak_vram, num_tokens = generate_fn(
                image, model, processor, device
            )

        # Metric 1: Efficiency (from metrics package)
        efficiency = compute_efficiency_local(
            generated_tokens=num_tokens,
            wall_time_sec=latency,
            peak_vram_bytes=peak_vram,
            latency_sec=latency
        )

        # Metric 2: Structural Alignment (from metrics package)
        structural = compute_structural_metrics(generated_html)

        # Save generated HTML
        html_path = Path(output_dir) / model_name / f"sample_{idx}.html"
        save_html_file(generated_html, html_path)

        # Metric 3: Correctness (from metrics/correctness.py approach)
        correctness = asyncio.run(check_html_correctness(html_path))

        # Store results
        results.append({
            'sample_id': idx,
            'model': model_name,
            'latency_seconds': efficiency.latency_seconds,
            'tokens_per_second': efficiency.tokens_per_second,
            'peak_vram_gb': efficiency.peak_vram_gb,
            'generated_tokens': num_tokens,
            'semantic_html_ratio': structural['semantic_html_ratio'],
            'accessibility_score': structural['accessibility_score'],
            'render_success': correctness['render_success'],
            'error_count': correctness['error_count'],
            'html_path': str(html_path)
        })

        print(f"  Latency: {latency:.2f}s | Tokens/s: {efficiency.tokens_per_second:.2f} | "
              f"VRAM: {efficiency.peak_vram_gb:.2f}GB | Render: {correctness['render_success']}")

    return results


def main():
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prompt = "Strictly follow the design in the image. Generate semantic HTML and responsive CSS. Output only the code."
    output_dir = Path("evaluation_results")

    print("\nLoading SALT-NLP/Design2Code-hf dataset...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
    print(f"Dataset loaded: {len(dataset)} samples")

    # Load models
    qwen_model, qwen_processor = load_qwen_model(device)
    websight_model, websight_processor = load_websight_model(device)

    # Evaluate Qwen3-VL
    qwen_results = evaluate_model(
        model=qwen_model,
        processor=qwen_processor,
        dataset=dataset,
        model_name="Qwen3-VL-2B",
        generate_fn=generate_qwen,
        device=device,
        prompt=prompt,
        output_dir=output_dir
    )

    del qwen_model
    torch.cuda.empty_cache()

    # Evaluate WebSight
    websight_results = evaluate_model(
        model=websight_model,
        processor=websight_processor,
        dataset=dataset,
        model_name="VLM_WebSight",
        generate_fn=generate_websight,
        device=device,
        prompt=prompt,
        output_dir=output_dir
    )

    # Combine and save results
    all_results = qwen_results + websight_results
    df = pd.DataFrame(all_results)

    output_file = output_dir / "evaluation_results.xlsx"
    df.to_excel(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        print(f"\n{model_name}:")
        print(f"  Average Latency: {model_df['latency_seconds'].mean():.2f}s")
        print(f"  Average Tokens/s: {model_df['tokens_per_second'].mean():.2f}")
        print(f"  Average VRAM: {model_df['peak_vram_gb'].mean():.2f}GB")
        print(f"  Average Semantic HTML Ratio: {model_df['semantic_html_ratio'].mean():.3f}")
        print(f"  Average Accessibility Score: {model_df['accessibility_score'].mean():.3f}")
        render_rate = (model_df['render_success'].sum() / len(model_df)) * 100
        print(f"  Render Success Rate: {render_rate:.2f}%")
        print(f"  Average Error Count: {model_df['error_count'].mean():.2f}")


if __name__ == "__main__":
    main()
