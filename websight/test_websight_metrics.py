"""
Comprehensive WebSight test with all 4 metrics
Stores all outputs in sample_results/
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from datasets import load_dataset
import pandas as pd
import time
import os
import json
import asyncio
from pathlib import Path
from tqdm import tqdm

# Import metrics
import sys
sys.path.append('metrics')
from efficiency import compute_efficiency_local
from structural_alignment import compute_structural_alignment_scores
from bs4 import BeautifulSoup
import re

def sanitize_for_excel(text):
    """Remove illegal characters that Excel/openpyxl can't handle."""
    if not isinstance(text, str):
        return text
    # Remove control characters except tab, newline, carriage return
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

# Configuration
TEST_SAMPLES = None  # None = all samples (484)
OUTPUT_DIR = "results_WebSight"
PREDICTIONS_DIR = f"{OUTPUT_DIR}/predictions"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# Create directories
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Predictions: {PREDICTIONS_DIR}")
print(f"Results: {RESULTS_DIR}")

# Load model
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

LOCAL_MODEL_DIR = "/root/models--HuggingFaceM4--VLM_WebSight_finetuned/snapshots/a5c2b06bfee0bd713cf2a6b3e4d46f94dd8fe839/"
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✓ Model loaded on {device}")

# Load prompt
with open("prompt.txt", 'r', encoding='utf-8') as f:
    DEFAULT_PROMPT = f.read().strip()
print(f"✓ Loaded prompt from prompt.txt")

# Load dataset
if TEST_SAMPLES is None:
    print(f"\n✓ Loading full dataset...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
else:
    print(f"\n✓ Loading {TEST_SAMPLES} test samples...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split=f"train[:{TEST_SAMPLES}]")

# DOM wrapper for structural alignment
class DOMNode:
    """Wrapper around BeautifulSoup to match structural_alignment interface"""
    def __init__(self, soup_element):
        self.element = soup_element
        self.tag = getattr(soup_element, 'name', "") or ""
        self.attrs = dict(soup_element.attrs) if hasattr(soup_element, 'attrs') else {}
        self.children = [DOMNode(child) for child in soup_element.children
                        if hasattr(child, 'name') and child.name] if hasattr(soup_element, 'children') else []

def parse_html_to_dom(html_string):
    """Parse HTML string to DOM tree"""
    try:
        soup = BeautifulSoup(html_string, 'lxml')
        body = soup.find('body')
        return DOMNode(body) if body else DOMNode(soup)
    except Exception as e:
        print(f"  Warning: HTML parsing failed: {e}")
        return None

def convert_to_rgb(image):
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")

def custom_transform(image):
    image = convert_to_rgb(image)
    image = to_numpy_array(image)
    image = resize(image, (960, 960), resample=PILImageResampling.BILINEAR)
    image = processor.image_processor.rescale(image, scale=1 / 255)
    image = processor.image_processor.normalize(
        image,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    image = to_channel_dimension_format(image, ChannelDimension.FIRST)
    return torch.tensor(image)

def generate_code(image, prompt=None):
    """Generate HTML/CSS code from image with full metrics"""
    if prompt is None:
        prompt = DEFAULT_PROMPT

    image_seq_len = model.config.perceiver_config.resampler_n_latents
    BOS = processor.tokenizer.bos_token
    bad_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"],
        add_special_tokens=False
    ).input_ids

    pixel_values = custom_transform(image).unsqueeze(0).to(device)
    inputs = processor.tokenizer(
        f"{BOS}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
        return_tensors="pt",
        add_special_tokens=False
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = pixel_values

    input_tokens = inputs['input_ids'].shape[1]

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            bad_words_ids=bad_ids,
            max_length=9000,
            use_cache=False
        )
    wall_time = time.time() - start_time

    # Extract metrics
    output_tokens = output_ids.shape[1] - input_tokens
    total_tokens = output_ids.shape[1]
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return {
        "text": generated_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "peak_vram_bytes": peak_vram,
        "wall_time_sec": wall_time
    }

# Run inference
print("\n" + "="*60)
print("RUNNING INFERENCE")
print("="*60)
results = []

for idx, sample in enumerate(tqdm(dataset, desc="Generating")):
    image = sample["image"]
    ground_truth = sample["text"]

    # Generate
    output = generate_code(image)

    # Save HTML file
    html_file = f"{idx}.html"
    with open(os.path.join(PREDICTIONS_DIR, html_file), "w", encoding="utf-8") as f:
        f.write(output["text"])

    # Compute efficiency metrics
    efficiency = compute_efficiency_local(
        generated_tokens=output["output_tokens"],
        wall_time_sec=output["wall_time_sec"],
        peak_vram_bytes=output["peak_vram_bytes"],
        latency_sec=output["wall_time_sec"]
    )

    # Compute structural alignment metrics
    ref_dom = parse_html_to_dom(ground_truth)
    pred_dom = parse_html_to_dom(output["text"])

    if ref_dom and pred_dom:
        struct_scores = compute_structural_alignment_scores(ref_dom, pred_dom)
        tree_sim = struct_scores.tree_edit_similarity
        semantic_ratio = struct_scores.semantic_html_ratio
        accessibility = struct_scores.accessibility_score
    else:
        tree_sim = semantic_ratio = accessibility = 0.0

    # Gemini-compatible format
    result = {
        "number": idx,
        "response_text": output["text"],
        "ground_truth": ground_truth,
        "prompt_token_count": output["input_tokens"],
        "candidates_token_count": output["output_tokens"],
        "total_token_count": output["total_tokens"],
        "latency": output["wall_time_sec"],
        # WebSight-specific: efficiency metrics
        "tokens_per_second": efficiency.tokens_per_second,
        "peak_vram_gb": efficiency.peak_vram_gb,
        "peak_vram_bytes": output["peak_vram_bytes"],
        # Structural alignment metrics
        "tree_edit_similarity": tree_sim,
        "semantic_html_ratio": semantic_ratio,
        "accessibility_score": accessibility,
        # Reference
        "html_file": html_file
    }
    results.append(result)

    print(f"\nSample {idx}: {output['output_tokens']} tokens in {output['wall_time_sec']:.2f}s "
          f"({efficiency.tokens_per_second:.1f} tok/s, {efficiency.peak_vram_gb:.2f}GB VRAM)")

    # Save individual checkpoint for each sample
    sample_checkpoint = f"{RESULTS_DIR}/{idx}.json"
    with open(sample_checkpoint, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Also save milestone checkpoints every 100 samples (all results so far)
    if (idx + 1) % 100 == 0:
        milestone_file = f"{RESULTS_DIR}/checkpoint_{idx+1}.json"
        with open(milestone_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n Milestone checkpoint saved: {milestone_file}")

# Save detailed results (JSON for full data, sanitized Excel for viewing)
with open(f"{RESULTS_DIR}/detailed_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n Saved detailed results to {RESULTS_DIR}/detailed_results.json")

# Also save sanitized Excel version
df = pd.DataFrame(results)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(sanitize_for_excel)
df.to_excel(f"{RESULTS_DIR}/detailed_results.xlsx", index=False)
print(f" Saved Excel version to {RESULTS_DIR}/detailed_results.xlsx")

# Run correctness metrics
print("\n" + "="*60)
print("RUNNING CORRECTNESS METRICS")
print("="*60)

async def run_correctness_check():
    """Run correctness metrics using Playwright"""
    from playwright.async_api import async_playwright

    correctness_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        html_files = list(Path(PREDICTIONS_DIR).glob('*.html'))
        print(f"Checking {len(html_files)} HTML files...")

        for file_path in html_files:
            context = await browser.new_context()
            page = await context.new_page()

            errors = []
            render_success = False

            try:
                # Set up error listeners
                page.on('console', lambda msg: errors.append(f"[Console]: {msg.text}") if msg.type.lower() == 'error' else None)
                page.on('pageerror', lambda err: errors.append(f"[Page Error]: {err.message}"))

                # Load HTML
                file_url = f'file://{file_path.resolve()}'
                await page.goto(file_url, wait_until='load')

                # Check render success
                body_content = await page.evaluate("() => document.body.innerHTML")
                render_success = bool(body_content and len(body_content.strip()) > 0)

            except Exception as e:
                errors.append(f"[Critical]: {e}")

            finally:
                await context.close()

            correctness_results.append({
                "file": file_path.name,
                "render_success": render_success,
                "error_count": len(errors),
                "errors": errors
            })

        await browser.close()

    return correctness_results

try:
    correctness_data = asyncio.run(run_correctness_check())
    correctness_df = pd.DataFrame(correctness_data)
    correctness_df.to_excel(f"{RESULTS_DIR}/correctness_metrics.xlsx", index=False)
    print(f"✓ Saved correctness metrics to {RESULTS_DIR}/correctness_metrics.xlsx")

    # Print summary
    success_rate = correctness_df['render_success'].sum() / len(correctness_df) * 100
    print(f"  Render Success Rate: {success_rate:.1f}%")
    print(f"  Total Errors: {correctness_df['error_count'].sum()}")
except Exception as e:
    print(f"⚠ Correctness metrics skipped: {e}")
    print("  (Install with: pip install playwright && playwright install)")

# Generate summary report
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

summary = {
    "model": "WebSight (HuggingFaceM4/VLM_WebSight_finetuned)",
    "test_samples": TEST_SAMPLES if TEST_SAMPLES else len(df),
    "metrics": {
        "efficiency": {
            "avg_tokens_per_second": float(df['tokens_per_second'].mean()),
            "avg_latency_seconds": float(df['latency'].mean()),
            "avg_peak_vram_gb": float(df['peak_vram_gb'].mean()),
            "total_tokens_generated": int(df['candidates_token_count'].sum())
        },
        "correctness": {
            "render_success_rate": f"{success_rate:.1f}%" if 'correctness_df' in locals() else "N/A",
            "total_errors": int(correctness_df['error_count'].sum()) if 'correctness_df' in locals() else "N/A"
        },
        "structural_alignment": {
            "avg_tree_edit_similarity": float(df['tree_edit_similarity'].mean()),
            "avg_semantic_html_ratio": float(df['semantic_html_ratio'].mean()),
            "avg_accessibility_score": float(df['accessibility_score'].mean())
        }
    },
    "output_directory": OUTPUT_DIR
}

# Save summary
with open(f"{RESULTS_DIR}/summary_report.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n📊 EFFICIENCY METRICS:")
print(f"  • Average Speed: {summary['metrics']['efficiency']['avg_tokens_per_second']:.1f} tokens/sec")
print(f"  • Average Latency: {summary['metrics']['efficiency']['avg_latency_seconds']:.2f} seconds")
print(f"  • Average VRAM: {summary['metrics']['efficiency']['avg_peak_vram_gb']:.2f} GB")
print(f"  • Total Tokens: {summary['metrics']['efficiency']['total_tokens_generated']}")

if 'correctness_df' in locals():
    print(f"\n✅ CORRECTNESS METRICS:")
    print(f"  • Render Success: {summary['metrics']['correctness']['render_success_rate']}")
    print(f"  • Total Errors: {summary['metrics']['correctness']['total_errors']}")

print(f"\n🏗️  STRUCTURAL ALIGNMENT METRICS:")
print(f"  • Tree Edit Similarity: {summary['metrics']['structural_alignment']['avg_tree_edit_similarity']:.3f}")
print(f"  • Semantic HTML Ratio: {summary['metrics']['structural_alignment']['avg_semantic_html_ratio']:.3f}")
print(f"  • Accessibility Score: {summary['metrics']['structural_alignment']['avg_accessibility_score']:.3f}")

print(f"\n📁 ALL OUTPUTS SAVED TO: {OUTPUT_DIR}/")
print(f"  • HTML files: {PREDICTIONS_DIR}/")
print(f"  • Results: {RESULTS_DIR}/")
print(f"  • Summary: {RESULTS_DIR}/summary_report.json")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
