"""
Complete Robustness Testing Pipeline

This script runs the full robustness evaluation workflow:
1. Generate perturbed images from original dataset
2. Run inference on perturbed images (requires model loading)
3. Render perturbed predictions to PNG
4. Calculate visual fidelity (CLIP) and structural alignment for clean vs perturbed
5. Compute degradation rates using metrics/robustness.py

Usage:
    python test_robustness.py --model qwen
    python test_robustness.py --model websight
    python test_robustness.py --model qwen --skip-inference  # If predictions already exist
    python test_robustness.py --model qwen --samples 50      # Test on subset

Output saved to: robustness_results/{model}/
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup

# DOMNode wrapper for structural alignment metrics
class DOMNode:
    """Wrapper around BeautifulSoup to match structural_alignment interface"""
    def __init__(self, soup_element):
        self.element = soup_element
        self.tag = getattr(soup_element, 'name', "") or ""
        self.name = self.tag  # Alias for tree_edit_similarity compatibility
        self.attrs = dict(soup_element.attrs) if hasattr(soup_element, 'attrs') else {}
        self.children = [DOMNode(child) for child in soup_element.children
                        if hasattr(child, 'name') and child.name] if hasattr(soup_element, 'children') else []


def parse_html_to_dom(html_string):
    """Parse HTML string to DOM tree wrapped in DOMNode"""
    try:
        soup = BeautifulSoup(html_string, 'lxml')
        body = soup.find('body')
        return DOMNode(body) if body else DOMNode(soup)
    except Exception:
        return None


def parse_html_to_soup(html_string):
    """Parse HTML string to BeautifulSoup element (for tree_edit_similarity)"""
    try:
        soup = BeautifulSoup(html_string, 'lxml')
        body = soup.find('body')
        return body if body else soup
    except Exception:
        return None


# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent / "metrics"))
from perturb_image import perturb_image
from robustness import compute_robustness_metrics
from structural_alignment import (
    semantic_html_usage,
    accessibility_score,
    tree_edit_similarity,
    compute_structural_alignment_scores,
)
from CLIP import calculate_clip_score

# Required imports
try:
    from PIL import Image
except ImportError:
    print("Error: PIL not available (pip install Pillow)")
    sys.exit(1)

try:
    import torch
except ImportError:
    raise ImportError("PyTorch not available. Install: pip install torch")

try:
    from playwright.async_api import async_playwright
except ImportError:
    raise ImportError("Playwright not available. Install: pip install playwright && playwright install")

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets not available (pip install datasets)")
    sys.exit(1)


def setup_directories(model_name, output_dir=None):
    """Create output directories."""
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(f"robustness_results/{model_name}")
    dirs = {
        "base": base_dir,
        "reference_images": base_dir / "reference_images",
        "perturbed_images": base_dir / "perturbed_images",
        "perturbed_predictions": base_dir / "perturbed_predictions",
        "perturbed_rendered": base_dir / "perturbed_rendered",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def step1_generate_perturbed_images(dirs, dataset, sample_indices, strength=0.05):
    """Generate perturbed versions of dataset images."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Perturbed Images")
    print("=" * 60)

    for idx in tqdm(sample_indices, desc="Perturbing"):
        ref_path = dirs["reference_images"] / f"{idx}.png"
        perturbed_path = dirs["perturbed_images"] / f"{idx}.png"

        # Skip if already exists
        if perturbed_path.exists() and ref_path.exists():
            continue

        # Get image from dataset
        sample = dataset[idx]
        image = sample["image"]

        # Save reference image
        if not ref_path.exists():
            image.save(ref_path)

        # Generate and save perturbed image
        perturbed = perturb_image(image, strength=strength)
        perturbed.save(perturbed_path)

    print(f"  Reference images: {dirs['reference_images']}")
    print(f"  Perturbed images: {dirs['perturbed_images']}")
    print(f"  Total: {len(sample_indices)} images")


def step2_run_inference_qwen(dirs, sample_indices):
    """Run Qwen inference on perturbed images."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Qwen Inference on Perturbed Images")
    print("=" * 60)

    # Check for existing predictions
    existing = [f for f in dirs["perturbed_predictions"].glob("*.html")]
    if len(existing) >= len(sample_indices):
        print(f"  Found {len(existing)} existing predictions. Skipping inference.")
        return

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Qwen dependencies not available. Install: pip install transformers qwen-vl-utils")

    # Load model
    print("  Loading Qwen3-VL-8B-Thinking model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Thinking",
        torch_dtype="auto",
        cache_dir="/root/"
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Thinking",
        cache_dir="/root/"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load prompt
    with open("prompt.txt", 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    # Run inference
    for idx in tqdm(sample_indices, desc="Inference"):
        output_path = dirs["perturbed_predictions"] / f"{idx}.html"
        if output_path.exists():
            continue

        perturbed_path = dirs["perturbed_images"] / f"{idx}.png"
        if not perturbed_path.exists():
            continue

        image = Image.open(perturbed_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=9000,
                temperature=0,
                do_sample=False
            )

        generated_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_text)

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Predictions saved to: {dirs['perturbed_predictions']}")


def step2_run_inference_websight(dirs, sample_indices):
    """Run WebSight inference on perturbed images."""
    print("\n" + "=" * 60)
    print("STEP 2: Running WebSight Inference on Perturbed Images")
    print("=" * 60)

    # Check for existing predictions
    existing = [f for f in dirs["perturbed_predictions"].glob("*.html")]
    if len(existing) >= len(sample_indices):
        print(f"  Found {len(existing)} existing predictions. Skipping inference.")
        return

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
        from transformers.image_transforms import resize, to_channel_dimension_format
    except ImportError:
        raise ImportError("WebSight dependencies not available. Install: pip install transformers")

    # Load model
    print("  Loading WebSight model...")
    LOCAL_MODEL_DIR = "/root/models--HuggingFaceM4--VLM_WebSight_finetuned/snapshots/a5c2b06bfee0bd713cf2a6b3e4d46f94dd8fe839/"

    if not Path(LOCAL_MODEL_DIR).exists():
        raise FileNotFoundError(f"Model not found at {LOCAL_MODEL_DIR}")

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

    # Run inference
    image_seq_len = model.config.perceiver_config.resampler_n_latents
    BOS = processor.tokenizer.bos_token
    bad_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"],
        add_special_tokens=False
    ).input_ids

    for idx in tqdm(sample_indices, desc="Inference"):
        output_path = dirs["perturbed_predictions"] / f"{idx}.html"
        if output_path.exists():
            continue

        perturbed_path = dirs["perturbed_images"] / f"{idx}.png"
        if not perturbed_path.exists():
            continue

        image = Image.open(perturbed_path)

        pixel_values = custom_transform(image).unsqueeze(0).to(device)
        inputs = processor.tokenizer(
            f"{BOS}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
            return_tensors="pt",
            add_special_tokens=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = pixel_values

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                bad_words_ids=bad_ids,
                max_length=9000,
                use_cache=False
            )

        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_text)

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Predictions saved to: {dirs['perturbed_predictions']}")


async def step3_render_predictions(dirs, sample_indices):
    """Render HTML predictions to PNG images."""
    print("\n" + "=" * 60)
    print("STEP 3: Rendering Predictions to PNG")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Render perturbed predictions
        print("  Rendering perturbed predictions...")
        for idx in tqdm(sample_indices, desc="  Perturbed"):
            html_path = dirs["perturbed_predictions"] / f"{idx}.html"
            output_path = dirs["perturbed_rendered"] / f"{idx}.png"

            if output_path.exists() or not html_path.exists():
                continue

            # Get viewport size from reference
            ref_path = dirs["reference_images"] / f"{idx}.png"
            if ref_path.exists():
                with Image.open(ref_path) as img:
                    width, height = img.size
            else:
                width, height = 1280, 720

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            await page.set_viewport_size({"width": width, "height": height})
            await page.set_content(html_content)
            await page.screenshot(path=str(output_path))

        await browser.close()

    print(f"  Perturbed rendered: {dirs['perturbed_rendered']}")


async def step4_calculate_scores(dirs, sample_indices, clean_predictions_dir, clean_rendered_dir):
    """Calculate visual fidelity (CLIP) and structural alignment scores using metrics/."""
    print("\n" + "=" * 60)
    print("STEP 4: Calculating Visual Fidelity & Structural Alignment")
    print("=" * 60)
    print("  Using metrics/CLIP.py for CLIP score")
    print("  Using metrics/IOU.py for IoU score")
    print("  Using metrics/structural_alignment.py for structural alignment")

    # Load dataset for ground truth HTML (needed for IoU)
    print("  Loading dataset for ground truth HTML...")
    from datasets import load_dataset
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
    print(f"  Dataset loaded: {len(dataset)} samples")

    # Initialize IoU benchmark
    from IOU import LayoutBenchmark

    benchmark = LayoutBenchmark()
    await benchmark.start()

    # Storage for all scores
    scores = {
        "clean_clip": [],
        "perturbed_clip": [],
        "clean_iou": [],
        "perturbed_iou": [],
        "clean_structural": [],
        "perturbed_structural": [],
        "clean_semantic_ratio": [],
        "perturbed_semantic_ratio": [],
        "clean_accessibility": [],
        "perturbed_accessibility": [],
        "clean_tree_edit": [],
        "perturbed_tree_edit": [],
    }

    for idx in tqdm(sample_indices, desc="  Calculating"):
        ref_path = dirs["reference_images"] / f"{idx}.png"
        clean_rendered_path = clean_rendered_dir / f"{idx}.png"
        perturbed_rendered_path = dirs["perturbed_rendered"] / f"{idx}.png"
        clean_html_path = clean_predictions_dir / f"{idx}.html"
        perturbed_html_path = dirs["perturbed_predictions"] / f"{idx}.html"

        # --- Visual Fidelity (CLIP) using metrics/CLIP.py ---
        if ref_path.exists():
            try:
                img_ref = Image.open(ref_path)

                # Clean CLIP score
                if clean_rendered_path.exists():
                    img_clean = Image.open(clean_rendered_path)
                    clean_clip = calculate_clip_score(img_ref, img_clean)
                    scores["clean_clip"].append(clean_clip)
                else:
                    scores["clean_clip"].append(None)

                # Perturbed CLIP score
                if perturbed_rendered_path.exists():
                    img_perturbed = Image.open(perturbed_rendered_path)
                    perturbed_clip = calculate_clip_score(img_ref, img_perturbed)
                    scores["perturbed_clip"].append(perturbed_clip)
                else:
                    scores["perturbed_clip"].append(None)
            except Exception:
                scores["clean_clip"].append(None)
                scores["perturbed_clip"].append(None)
        else:
            scores["clean_clip"].append(None)
            scores["perturbed_clip"].append(None)

        # --- Visual Fidelity (IoU) using metrics/IOU.py ---
        # Get ground truth HTML from dataset
        gt_html = dataset[idx]['text']
        gt_dom = parse_html_to_dom(gt_html)
        gt_soup = parse_html_to_soup(gt_html)  # For tree_edit_similarity

        async def calc_iou(pred_html_path, gt_html, sample_idx):
            """Calculate IoU score between prediction and ground truth"""
            try:
                if not pred_html_path.exists():
                    return None
                with open(pred_html_path, 'r', encoding='utf-8') as f:
                    pred_html = f.read()
                gt_boxes = await benchmark.get_element_bboxes(gt_html, sample_idx, "GT")
                pred_boxes = await benchmark.get_element_bboxes(pred_html, sample_idx, "PRED")
                return benchmark.compare_layouts(gt_boxes, pred_boxes)
            except Exception:
                return None

        # Clean IoU score
        clean_iou = await calc_iou(clean_html_path, gt_html, idx)
        scores["clean_iou"].append(clean_iou)

        # Perturbed IoU score
        perturbed_iou = await calc_iou(perturbed_html_path, gt_html, idx)
        scores["perturbed_iou"].append(perturbed_iou)

        # --- Structural Alignment using metrics/structural_alignment.py ---
        # Clean structural (using DOMNode wrapper)
        if clean_html_path.exists():
            try:
                with open(clean_html_path, 'r', encoding='utf-8') as f:
                    clean_html = f.read()
                clean_dom = parse_html_to_dom(clean_html)
                clean_soup = parse_html_to_soup(clean_html)  # For tree_edit_similarity
                if clean_dom:
                    sem = semantic_html_usage(clean_dom)
                    acc = accessibility_score(clean_dom)
                    # Tree Edit Similarity (compare prediction vs ground truth, needs soup elements)
                    tree_sim = tree_edit_similarity(gt_soup, clean_soup) if gt_soup and clean_soup else None
                    # Combined score (average of semantic + accessibility + tree_edit)
                    if tree_sim is not None:
                        struct = (sem + acc + tree_sim) / 3.0
                    else:
                        struct = (sem + acc) / 2.0
                    scores["clean_structural"].append(struct)
                    scores["clean_semantic_ratio"].append(sem)
                    scores["clean_accessibility"].append(acc)
                    scores["clean_tree_edit"].append(tree_sim)
                else:
                    scores["clean_structural"].append(None)
                    scores["clean_semantic_ratio"].append(None)
                    scores["clean_accessibility"].append(None)
                    scores["clean_tree_edit"].append(None)
            except Exception:
                scores["clean_structural"].append(None)
                scores["clean_semantic_ratio"].append(None)
                scores["clean_accessibility"].append(None)
                scores["clean_tree_edit"].append(None)
        else:
            scores["clean_structural"].append(None)
            scores["clean_semantic_ratio"].append(None)
            scores["clean_accessibility"].append(None)
            scores["clean_tree_edit"].append(None)

        # Perturbed structural (using DOMNode wrapper)
        if perturbed_html_path.exists():
            try:
                with open(perturbed_html_path, 'r', encoding='utf-8') as f:
                    perturbed_html = f.read()
                perturbed_dom = parse_html_to_dom(perturbed_html)
                perturbed_soup = parse_html_to_soup(perturbed_html)  # For tree_edit_similarity
                if perturbed_dom:
                    sem = semantic_html_usage(perturbed_dom)
                    acc = accessibility_score(perturbed_dom)
                    # Tree Edit Similarity (compare prediction vs ground truth, needs soup elements)
                    tree_sim = tree_edit_similarity(gt_soup, perturbed_soup) if gt_soup and perturbed_soup else None
                    # Combined score (average of semantic + accessibility + tree_edit)
                    if tree_sim is not None:
                        struct = (sem + acc + tree_sim) / 3.0
                    else:
                        struct = (sem + acc) / 2.0
                    scores["perturbed_structural"].append(struct)
                    scores["perturbed_semantic_ratio"].append(sem)
                    scores["perturbed_accessibility"].append(acc)
                    scores["perturbed_tree_edit"].append(tree_sim)
                else:
                    scores["perturbed_structural"].append(None)
                    scores["perturbed_semantic_ratio"].append(None)
                    scores["perturbed_accessibility"].append(None)
                    scores["perturbed_tree_edit"].append(None)
            except Exception:
                scores["perturbed_structural"].append(None)
                scores["perturbed_semantic_ratio"].append(None)
                scores["perturbed_accessibility"].append(None)
                scores["perturbed_tree_edit"].append(None)
        else:
            scores["perturbed_structural"].append(None)
            scores["perturbed_semantic_ratio"].append(None)
            scores["perturbed_accessibility"].append(None)
            scores["perturbed_tree_edit"].append(None)

    # Close IoU benchmark
    await benchmark.stop()

    return scores


def step5_generate_report(dirs, sample_indices, scores, model_name, strength=0.05):
    """Generate robustness evaluation report using metrics/robustness.py."""
    print("\n" + "=" * 60)
    print("STEP 5: Generating Report")
    print("=" * 60)

    def calc_mean(values):
        valid = [v for v in values if v is not None]
        return np.mean(valid) if valid else None

    def calc_std(values):
        valid = [v for v in values if v is not None]
        return np.std(valid) if valid else None

    def count_valid(values):
        return len([v for v in values if v is not None])

    # Calculate means for degradation rate
    clean_clip_mean = calc_mean(scores["clean_clip"])
    perturbed_clip_mean = calc_mean(scores["perturbed_clip"])
    clean_iou_mean = calc_mean(scores["clean_iou"])
    perturbed_iou_mean = calc_mean(scores["perturbed_iou"])
    clean_structural_mean = calc_mean(scores["clean_structural"])
    perturbed_structural_mean = calc_mean(scores["perturbed_structural"])
    clean_tree_edit_mean = calc_mean(scores["clean_tree_edit"])
    perturbed_tree_edit_mean = calc_mean(scores["perturbed_tree_edit"])

    # Combine sub-metrics into Visual Fidelity: (CLIP + IoU) / 2
    def safe_avg(*values):
        valid = [v for v in values if v is not None]
        return np.mean(valid) if valid else 0.0

    clean_visual_fidelity = safe_avg(clean_clip_mean, clean_iou_mean)
    perturbed_visual_fidelity = safe_avg(perturbed_clip_mean, perturbed_iou_mean)

    # Structural Alignment already combines (semantic + accessibility + tree_edit) / 3
    # Use the pre-computed structural means

    # Use metrics/robustness.py to compute degradation rates
    clean_metrics = {
        "visual_fidelity": clean_visual_fidelity,
        "structural_alignment": clean_structural_mean if clean_structural_mean else 0.0,
    }
    perturbed_metrics = {
        "visual_fidelity": perturbed_visual_fidelity,
        "structural_alignment": perturbed_structural_mean if perturbed_structural_mean else 0.0,
    }

    robustness_results = compute_robustness_metrics(clean_metrics, perturbed_metrics)

    # Generate report
    report = []
    report.append("=" * 60)
    report.append(f"{model_name.upper()} MODEL - ROBUSTNESS EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    report.append("CONFIGURATION:")
    report.append(f"  Model:              {model_name}")
    report.append(f"  Total Samples:      {len(sample_indices)}")
    report.append(f"  Perturbation:       Crop + Brightness + Contrast + Blur")
    report.append(f"  Perturbation Strength: {strength}")

    # Visual Fidelity Section
    report.append("")
    report.append("=" * 60)
    report.append("1. VISUAL FIDELITY (CLIP + IoU)")
    report.append("=" * 60)
    report.append("")
    report.append("Combined Visual Fidelity = (CLIP + IoU) / 2")
    report.append(f"  Clean:     {clean_visual_fidelity:.4f}")
    report.append(f"  Perturbed: {perturbed_visual_fidelity:.4f}")
    report.append("")
    report.append("1.1 CLIP Score (range: [0, 1], higher is better):")
    report.append("  Clean Predictions:")
    if clean_clip_mean is not None:
        report.append(f"    Mean:   {clean_clip_mean:.4f}")
        report.append(f"    Std:    {calc_std(scores['clean_clip']):.4f}")
        report.append(f"    Count:  {count_valid(scores['clean_clip'])}")
    else:
        report.append("    (Not computed)")
    report.append("  Perturbed Predictions:")
    if perturbed_clip_mean is not None:
        report.append(f"    Mean:   {perturbed_clip_mean:.4f}")
        report.append(f"    Std:    {calc_std(scores['perturbed_clip']):.4f}")
        report.append(f"    Count:  {count_valid(scores['perturbed_clip'])}")
    else:
        report.append("    (Not computed)")
    report.append("")
    report.append("1.2 IoU Score (range: [0, 1], higher is better):")
    report.append("  Clean Predictions:")
    if clean_iou_mean is not None:
        report.append(f"    Mean:   {clean_iou_mean:.4f}")
        report.append(f"    Std:    {calc_std(scores['clean_iou']):.4f}")
        report.append(f"    Count:  {count_valid(scores['clean_iou'])}")
    else:
        report.append("    (Not computed)")
    report.append("  Perturbed Predictions:")
    if perturbed_iou_mean is not None:
        report.append(f"    Mean:   {perturbed_iou_mean:.4f}")
        report.append(f"    Std:    {calc_std(scores['perturbed_iou']):.4f}")
        report.append(f"    Count:  {count_valid(scores['perturbed_iou'])}")
    else:
        report.append("    (Not computed)")

    # Structural Alignment Section
    report.append("")
    report.append("=" * 60)
    report.append("2. STRUCTURAL ALIGNMENT (Semantic + Accessibility + Tree Edit)")
    report.append("=" * 60)
    report.append("")
    report.append("Combined Score = (Semantic + Accessibility + Tree Edit) / 3")
    report.append("")
    report.append("  Clean Predictions:")
    if clean_structural_mean is not None:
        report.append(f"    Combined Mean:   {clean_structural_mean:.4f}")
        report.append(f"    Std:             {calc_std(scores['clean_structural']):.4f}")
        report.append(f"    Count:           {count_valid(scores['clean_structural'])}")
        report.append(f"    - Semantic HTML: {calc_mean(scores['clean_semantic_ratio']):.4f}" if calc_mean(scores['clean_semantic_ratio']) is not None else "    - Semantic HTML: N/A")
        report.append(f"    - Accessibility: {calc_mean(scores['clean_accessibility']):.4f}" if calc_mean(scores['clean_accessibility']) is not None else "    - Accessibility: N/A")
        report.append(f"    - Tree Edit Sim: {clean_tree_edit_mean:.4f}" if clean_tree_edit_mean is not None else "    - Tree Edit Sim: N/A")
    else:
        report.append("    (Not computed)")
    report.append("")
    report.append("  Perturbed Predictions:")
    if perturbed_structural_mean is not None:
        report.append(f"    Combined Mean:   {perturbed_structural_mean:.4f}")
        report.append(f"    Std:             {calc_std(scores['perturbed_structural']):.4f}")
        report.append(f"    Count:           {count_valid(scores['perturbed_structural'])}")
        report.append(f"    - Semantic HTML: {calc_mean(scores['perturbed_semantic_ratio']):.4f}" if calc_mean(scores['perturbed_semantic_ratio']) is not None else "    - Semantic HTML: N/A")
        report.append(f"    - Accessibility: {calc_mean(scores['perturbed_accessibility']):.4f}" if calc_mean(scores['perturbed_accessibility']) is not None else "    - Accessibility: N/A")
        report.append(f"    - Tree Edit Sim: {perturbed_tree_edit_mean:.4f}" if perturbed_tree_edit_mean is not None else "    - Tree Edit Sim: N/A")
    else:
        report.append("    (Not computed)")

    # Robustness Metrics Section
    report.append("")
    report.append("=" * 60)
    report.append("3. ROBUSTNESS METRICS (Performance Degradation Rate)")
    report.append("=" * 60)
    report.append("")
    report.append("Degradation Rate (range: [0%, 100%], lower is better):")
    report.append("  Formula: (clean - perturbed) / clean")
    report.append("")

    vf_drop = robustness_results["visual_fidelity_drop"]
    sa_drop = robustness_results["structural_alignment_drop"]

    report.append(f"  Visual Fidelity Drop:      {vf_drop*100:.2f}%")
    report.append(f"  Structural Alignment Drop: {sa_drop*100:.2f}%")
    report.append("")

    # Overall rating
    avg_drop = (vf_drop + sa_drop) / 2
    report.append(f"  Average Degradation:       {avg_drop*100:.2f}%")
    report.append("")
    if avg_drop < 0.05:
        report.append("  Rating: EXCELLENT (< 5% degradation)")
    elif avg_drop < 0.10:
        report.append("  Rating: GOOD (5-10% degradation)")
    elif avg_drop < 0.20:
        report.append("  Rating: MODERATE (10-20% degradation)")
    else:
        report.append("  Rating: POOR (> 20% degradation)")

    report.append("")
    report.append("=" * 60)
    report.append("OUTPUT FILES")
    report.append("=" * 60)
    report.append(f"  Reference images:      {dirs['reference_images']}")
    report.append(f"  Perturbed images:      {dirs['perturbed_images']}")
    report.append(f"  Perturbed predictions: {dirs['perturbed_predictions']}")
    report.append(f"  Perturbed rendered:    {dirs['perturbed_rendered']}")
    report.append("")
    report.append("=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)

    # Save report
    report_text = "\n".join(report)
    print(report_text)

    report_path = dirs["base"] / "robustness_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    # Save JSON
    json_output = {
        "model": model_name,
        "generated": datetime.now().isoformat(),
        "samples": len(sample_indices),
        "strength": strength,
        "visual_fidelity": {
            "combined": {
                "clean": clean_visual_fidelity,
                "perturbed": perturbed_visual_fidelity,
                "formula": "(CLIP + IoU) / 2"
            },
            "clip": {
                "clean": {
                    "mean": clean_clip_mean,
                    "std": calc_std(scores["clean_clip"]),
                    "count": count_valid(scores["clean_clip"])
                },
                "perturbed": {
                    "mean": perturbed_clip_mean,
                    "std": calc_std(scores["perturbed_clip"]),
                    "count": count_valid(scores["perturbed_clip"])
                }
            },
            "iou": {
                "clean": {
                    "mean": clean_iou_mean,
                    "std": calc_std(scores["clean_iou"]),
                    "count": count_valid(scores["clean_iou"])
                },
                "perturbed": {
                    "mean": perturbed_iou_mean,
                    "std": calc_std(scores["perturbed_iou"]),
                    "count": count_valid(scores["perturbed_iou"])
                }
            }
        },
        "structural_alignment": {
            "combined": {
                "clean": clean_structural_mean,
                "perturbed": perturbed_structural_mean,
                "formula": "(Semantic + Accessibility + Tree Edit) / 3"
            },
            "clean": {
                "mean": clean_structural_mean,
                "std": calc_std(scores["clean_structural"]),
                "count": count_valid(scores["clean_structural"]),
                "semantic_ratio_mean": calc_mean(scores["clean_semantic_ratio"]),
                "accessibility_mean": calc_mean(scores["clean_accessibility"]),
                "tree_edit_mean": clean_tree_edit_mean
            },
            "perturbed": {
                "mean": perturbed_structural_mean,
                "std": calc_std(scores["perturbed_structural"]),
                "count": count_valid(scores["perturbed_structural"]),
                "semantic_ratio_mean": calc_mean(scores["perturbed_semantic_ratio"]),
                "accessibility_mean": calc_mean(scores["perturbed_accessibility"]),
                "tree_edit_mean": perturbed_tree_edit_mean
            }
        },
        "degradation_rate": {
            "visual_fidelity_drop": vf_drop,
            "structural_alignment_drop": sa_drop,
            "average": avg_drop
        },
        "per_sample": {
            "clean_clip": scores["clean_clip"],
            "perturbed_clip": scores["perturbed_clip"],
            "clean_iou": scores["clean_iou"],
            "perturbed_iou": scores["perturbed_iou"],
            "clean_structural": scores["clean_structural"],
            "perturbed_structural": scores["perturbed_structural"],
            "clean_tree_edit": scores["clean_tree_edit"],
            "perturbed_tree_edit": scores["perturbed_tree_edit"]
        }
    }
    json_path = dirs["base"] / "robustness_report.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to: {json_path}")


async def main():
    parser = argparse.ArgumentParser(description="Complete Robustness Testing Pipeline")
    parser.add_argument("--model", choices=["qwen", "websight"], required=True,
                        help="Model to test")
    parser.add_argument("--samples", type=int, default=484,
                        help="Number of samples to test (default: 484 = full dataset)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference step (use existing predictions)")
    parser.add_argument("--strength", type=float, default=0.05,
                        help="Perturbation strength (default: 0.05)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: robustness_results/{model})")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ROBUSTNESS TESTING - {args.model.upper()}")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"Perturbation strength: {args.strength}")
    print(f"Skip inference: {args.skip_inference}")
    print(f"Output directory: {args.output_dir or f'robustness_results/{args.model}'}")

    # Setup
    dirs = setup_directories(args.model, args.output_dir)
    sample_indices = list(range(min(args.samples, 484)))

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
    print(f"  Dataset size: {len(dataset)}")

    # Step 1: Generate perturbed images
    step1_generate_perturbed_images(dirs, dataset, sample_indices, args.strength)

    # Step 2: Run inference
    if not args.skip_inference:
        if args.model == "qwen":
            step2_run_inference_qwen(dirs, sample_indices)
        else:
            step2_run_inference_websight(dirs, sample_indices)
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Skipping Inference (--skip-inference)")
        print("=" * 60)

    # Step 3: Render predictions
    await step3_render_predictions(dirs, sample_indices)

    # Step 4: Calculate scores
    # Get clean directories from existing results
    if args.model == "qwen":
        clean_predictions_dir = Path("results_Qwen/predictions")
        clean_rendered_dir = Path("results_Qwen/rendered_imgs")
    else:
        clean_predictions_dir = Path("results_WebSight/predictions")
        clean_rendered_dir = Path("results_WebSight/rendered_imgs")

    scores = await step4_calculate_scores(
        dirs, sample_indices, clean_predictions_dir, clean_rendered_dir
    )

    # Step 5: Generate report
    step5_generate_report(dirs, sample_indices, scores, args.model, args.strength)

    print("\n" + "=" * 60)
    print("ROBUSTNESS TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
