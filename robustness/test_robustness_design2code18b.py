"""
Robustness Testing for Design2Code-18B

This script runs the robustness evaluation workflow:
1. Generate perturbed images from original dataset
2. Run Design2Code-18B inference on perturbed images
3. Render perturbed predictions to PNG
4. Calculate visual fidelity (CLIP + IoU) and structural alignment
5. Generate comprehensive report

Usage:
    python test_robustness_design2code18b.py --strength 0.05 --samples 50
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

# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent / "metrics"))
from perturb_image import perturb_image
from robustness import compute_robustness_metrics
from structural_alignment import (
    semantic_html_usage,
    tree_edit_similarity,
)
from CLIP import calculate_clip_score
from IOU import LayoutBenchmark

from PIL import Image
import torch

try:
    from playwright.async_api import async_playwright
except ImportError:
    raise ImportError("Playwright not available. Install: pip install playwright && playwright install")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets not available. Install: pip install datasets")


class DOMNode:
    """Wrapper around BeautifulSoup to match structural_alignment interface"""
    def __init__(self, soup_element):
        self.element = soup_element
        self.tag = getattr(soup_element, 'name', "") or ""
        self.name = self.tag
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


def setup_directories(output_dir):
    """Create output directories."""
    base_dir = Path(output_dir)
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

        if perturbed_path.exists() and ref_path.exists():
            continue

        sample = dataset[idx]
        image = sample["image"]

        if not ref_path.exists():
            image.save(ref_path)

        perturbed = perturb_image(image, strength=strength)
        perturbed.save(perturbed_path)

    print(f"  Reference images: {dirs['reference_images']}")
    print(f"  Perturbed images: {dirs['perturbed_images']}")
    print(f"  Total: {len(sample_indices)} images")


def step2_run_inference_design2code18b(dirs, sample_indices):
    """Run Design2Code-18B inference on perturbed images."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Design2Code-18B Inference on Perturbed Images")
    print("=" * 60)

    # Check for existing predictions
    existing = list(dirs["perturbed_predictions"].glob("*.html"))
    if len(existing) >= len(sample_indices):
        print(f"  Found {len(existing)} existing predictions. Skipping inference.")
        return

    # Add CogVLM to path
    cogvlm_path = "/root/Design2Code_official/CogVLM"
    sys.path.insert(1, cogvlm_path)

    try:
        from sat.model import AutoModel
        from sat.model.mixins import CachedAutoregressiveMixin
        from utils.models import FineTuneTestCogAgentModel
        from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
    except ImportError as e:
        print(f"Error importing SAT/CogVLM modules: {e}")
        print("Please ensure SwissArmyTransformer==0.4.9 is installed")
        sys.exit(1)

    # Model configuration
    model_path = "/root/models--design2code-18b-v0/design2code-18b-v0"
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    print("  Loading Design2Code-18B model...")

    model_args_namespace = argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=1,
        model_parallel_size=1,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
        bf16=True,
        fp16=None,
        stream_chat=False
    )

    # Change to model directory for loading
    original_cwd = os.getcwd()
    os.chdir(model_dir)

    model, model_args = FineTuneTestCogAgentModel.from_pretrained(
        model_name,
        args=model_args_namespace,
        home_path=model_dir,
        overwrite_args={'model_parallel_size': 1}
    )
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    os.chdir(original_cwd)
    print("  Model loaded successfully!")

    # Initialize processors
    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else "chat"
    tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    text_processor_infer = llama2_text_processor_inference(tokenizer, 2048, model.image_length)

    # Inference function
    def get_html(image_path):
        with torch.no_grad():
            history = None
            cache_image = None
            query = ''

            response, history, cache_image = chat(
                image_path,
                model,
                text_processor_infer,
                image_processor,
                query,
                history=history,
                cross_img_processor=cross_image_processor,
                image=cache_image,
                max_length=4096,
                top_p=1.0,
                temperature=0.5,
                top_k=1,
                invalid_slices=text_processor_infer.invalid_slices,
                repetition_penalty=1.1,
                args=model_args_namespace
            )
        return response

    # Run inference on perturbed images
    for idx in tqdm(sample_indices, desc="Inference"):
        output_path = dirs["perturbed_predictions"] / f"{idx}.html"
        if output_path.exists():
            continue

        perturbed_path = dirs["perturbed_images"] / f"{idx}.png"
        if not perturbed_path.exists():
            continue

        try:
            html_output = get_html(str(perturbed_path))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
        except Exception as e:
            print(f"\nError processing {idx}: {e}")
            continue

    # Cleanup
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

        for idx in tqdm(sample_indices, desc="  Rendering"):
            html_path = dirs["perturbed_predictions"] / f"{idx}.html"
            output_path = dirs["perturbed_rendered"] / f"{idx}.png"

            if output_path.exists() or not html_path.exists():
                continue

            ref_path = dirs["reference_images"] / f"{idx}.png"
            if ref_path.exists():
                with Image.open(ref_path) as img:
                    width, height = img.size
            else:
                width, height = 1280, 720

            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                await page.set_viewport_size({"width": width, "height": height})
                await page.set_content(html_content)
                await page.screenshot(path=str(output_path))
            except Exception as e:
                print(f"\nError rendering {idx}: {e}")
                continue

        await browser.close()

    print(f"  Perturbed rendered: {dirs['perturbed_rendered']}")


async def step4_calculate_scores(dirs, sample_indices, clean_predictions_dir, clean_rendered_dir, dataset):
    """Calculate visual fidelity (CLIP + IoU) and structural alignment scores."""
    print("\n" + "=" * 60)
    print("STEP 4: Calculating Visual Fidelity & Structural Alignment")
    print("=" * 60)
    print("  Using metrics/CLIP.py for CLIP score")
    print("  Using metrics/IOU.py for IoU score")
    print("  Using metrics/structural_alignment.py for structural alignment")

    # Initialize IoU benchmark
    benchmark = LayoutBenchmark()
    await benchmark.start()

    scores = {
        "clean_clip": [],
        "perturbed_clip": [],
        "clean_iou": [],
        "perturbed_iou": [],
        "clean_structural": [],
        "perturbed_structural": [],
        "clean_semantic_ratio": [],
        "perturbed_semantic_ratio": [],
        "clean_tree_edit": [],
        "perturbed_tree_edit": [],
    }

    for idx in tqdm(sample_indices, desc="  Calculating"):
        ref_path = dirs["reference_images"] / f"{idx}.png"
        clean_rendered_path = clean_rendered_dir / f"{idx}.png"
        perturbed_rendered_path = dirs["perturbed_rendered"] / f"{idx}.png"
        clean_html_path = clean_predictions_dir / f"{idx}.html"
        perturbed_html_path = dirs["perturbed_predictions"] / f"{idx}.html"

        # --- Visual Fidelity (CLIP) ---
        if ref_path.exists():
            try:
                img_ref = Image.open(ref_path)

                if clean_rendered_path.exists():
                    img_clean = Image.open(clean_rendered_path)
                    clean_clip = calculate_clip_score(img_ref, img_clean)
                    scores["clean_clip"].append(clean_clip)
                else:
                    scores["clean_clip"].append(None)

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

        # --- Visual Fidelity (IoU) ---
        gt_html = dataset[idx]['text']
        gt_dom = parse_html_to_dom(gt_html)
        gt_soup = parse_html_to_soup(gt_html)

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

        # --- Structural Alignment ---
        # Clean structural
        if clean_html_path.exists():
            try:
                with open(clean_html_path, 'r', encoding='utf-8') as f:
                    clean_html = f.read()
                clean_dom = parse_html_to_dom(clean_html)
                clean_soup = parse_html_to_soup(clean_html)

                if clean_dom:
                    sem = semantic_html_usage(clean_dom)
                    tree_sim = tree_edit_similarity(gt_soup, clean_soup) if gt_soup and clean_soup else None

                    if tree_sim is not None:
                        struct = (sem + tree_sim) / 2.0
                    else:
                        struct = sem

                    scores["clean_structural"].append(struct)
                    scores["clean_semantic_ratio"].append(sem)
                    scores["clean_tree_edit"].append(tree_sim)
                else:
                    scores["clean_structural"].append(None)
                    scores["clean_semantic_ratio"].append(None)
                    scores["clean_tree_edit"].append(None)
            except Exception:
                scores["clean_structural"].append(None)
                scores["clean_semantic_ratio"].append(None)
                scores["clean_tree_edit"].append(None)
        else:
            scores["clean_structural"].append(None)
            scores["clean_semantic_ratio"].append(None)
            scores["clean_tree_edit"].append(None)

        # Perturbed structural
        if perturbed_html_path.exists():
            try:
                with open(perturbed_html_path, 'r', encoding='utf-8') as f:
                    perturbed_html = f.read()
                perturbed_dom = parse_html_to_dom(perturbed_html)
                perturbed_soup = parse_html_to_soup(perturbed_html)

                if perturbed_dom:
                    sem = semantic_html_usage(perturbed_dom)
                    tree_sim = tree_edit_similarity(gt_soup, perturbed_soup) if gt_soup and perturbed_soup else None

                    if tree_sim is not None:
                        struct = (sem + tree_sim) / 2.0
                    else:
                        struct = sem

                    scores["perturbed_structural"].append(struct)
                    scores["perturbed_semantic_ratio"].append(sem)
                    scores["perturbed_tree_edit"].append(tree_sim)
                else:
                    scores["perturbed_structural"].append(None)
                    scores["perturbed_semantic_ratio"].append(None)
                    scores["perturbed_tree_edit"].append(None)
            except Exception:
                scores["perturbed_structural"].append(None)
                scores["perturbed_semantic_ratio"].append(None)
                scores["perturbed_tree_edit"].append(None)
        else:
            scores["perturbed_structural"].append(None)
            scores["perturbed_semantic_ratio"].append(None)
            scores["perturbed_tree_edit"].append(None)

    # Close IoU benchmark
    await benchmark.stop()

    return scores


def step5_generate_report(dirs, sample_indices, scores, model_name, strength=0.05):
    """Generate robustness evaluation report."""
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

    # Calculate means
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
    report.append("2. STRUCTURAL ALIGNMENT (Semantic + Tree Edit)")
    report.append("=" * 60)
    report.append("")
    report.append("Combined Score = (Semantic + Tree Edit) / 2")
    report.append("")
    report.append("  Clean Predictions:")
    if clean_structural_mean is not None:
        report.append(f"    Combined Mean:   {clean_structural_mean:.4f}")
        report.append(f"    Std:             {calc_std(scores['clean_structural']):.4f}")
        report.append(f"    Count:           {count_valid(scores['clean_structural'])}")
        sem_mean = calc_mean(scores['clean_semantic_ratio'])
        report.append(f"    - Semantic HTML: {sem_mean:.4f}" if sem_mean else "    - Semantic HTML: N/A")
        report.append(f"    - Tree Edit Sim: {clean_tree_edit_mean:.4f}" if clean_tree_edit_mean else "    - Tree Edit Sim: N/A")
    else:
        report.append("    (Not computed)")
    report.append("")
    report.append("  Perturbed Predictions:")
    if perturbed_structural_mean is not None:
        report.append(f"    Combined Mean:   {perturbed_structural_mean:.4f}")
        report.append(f"    Std:             {calc_std(scores['perturbed_structural']):.4f}")
        report.append(f"    Count:           {count_valid(scores['perturbed_structural'])}")
        sem_mean = calc_mean(scores['perturbed_semantic_ratio'])
        report.append(f"    - Semantic HTML: {sem_mean:.4f}" if sem_mean else "    - Semantic HTML: N/A")
        report.append(f"    - Tree Edit Sim: {perturbed_tree_edit_mean:.4f}" if perturbed_tree_edit_mean else "    - Tree Edit Sim: N/A")
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

    # Helper to convert None to 0 for per-sample lists
    def safe_list(values):
        return [v if v is not None else 0.0 for v in values]

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
            },
            "clip": {
                "clean_mean": clean_clip_mean,
                "perturbed_mean": perturbed_clip_mean,
            },
            "iou": {
                "clean_mean": clean_iou_mean,
                "perturbed_mean": perturbed_iou_mean,
            }
        },
        "structural_alignment": {
            "clean_mean": clean_structural_mean,
            "perturbed_mean": perturbed_structural_mean,
        },
        "degradation": {
            "visual_fidelity_drop": vf_drop,
            "structural_alignment_drop": sa_drop,
            "average_drop": avg_drop,
        },
        "per_sample": {
            "clean_clip": safe_list(scores["clean_clip"]),
            "perturbed_clip": safe_list(scores["perturbed_clip"]),
            "clean_iou": safe_list(scores["clean_iou"]),
            "perturbed_iou": safe_list(scores["perturbed_iou"]),
            "clean_structural": safe_list(scores["clean_structural"]),
            "perturbed_structural": safe_list(scores["perturbed_structural"]),
            "clean_tree_edit": safe_list(scores["clean_tree_edit"]),
            "perturbed_tree_edit": safe_list(scores["perturbed_tree_edit"]),
        }
    }

    json_path = dirs["base"] / "robustness_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to: {json_path}")

    return robustness_results


async def main_async(args):
    """Main async function."""
    print("=" * 60)
    print("DESIGN2CODE-18B ROBUSTNESS TESTING")
    print("=" * 60)
    print(f"Perturbation strength: {args.strength}")
    print(f"Samples: {args.samples}")
    print(f"Output directory: {args.output_dir}")

    # Setup directories
    dirs = setup_directories(args.output_dir)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
    sample_indices = list(range(min(args.samples, len(dataset))))
    print(f"Using {len(sample_indices)} samples")

    # Step 1: Generate perturbed images
    step1_generate_perturbed_images(dirs, dataset, sample_indices, args.strength)

    # Step 2: Run inference on perturbed images
    if not args.skip_inference:
        step2_run_inference_design2code18b(dirs, sample_indices)
    else:
        print("\n[STEP 2] Skipping inference (--skip-inference flag)")

    # Step 3: Render predictions
    await step3_render_predictions(dirs, sample_indices)

    # Step 4: Calculate scores
    clean_predictions_dir = Path(args.clean_predictions_dir)
    clean_rendered_dir = Path(args.clean_rendered_dir)
    scores = await step4_calculate_scores(dirs, sample_indices, clean_predictions_dir, clean_rendered_dir, dataset)

    # Step 5: Generate report
    results = step5_generate_report(dirs, sample_indices, scores, "design2code-18b", args.strength)

    print("\n" + "=" * 60)
    print("ROBUSTNESS TESTING COMPLETE!")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Design2Code-18B Robustness Testing")
    parser.add_argument("--strength", type=float, default=0.05,
                        help="Perturbation strength (default: 0.05)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to test (default: 50)")
    parser.add_argument("--output-dir", type=str, default="robustness_results/design2code18b",
                        help="Output directory for robustness results")
    parser.add_argument("--clean-predictions-dir", type=str,
                        default="results_Design2Code18B/predictions",
                        help="Directory containing clean HTML predictions")
    parser.add_argument("--clean-rendered-dir", type=str,
                        default="results_Design2Code18B/rendered",
                        help="Directory containing clean rendered PNG files")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference step (use existing predictions)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
