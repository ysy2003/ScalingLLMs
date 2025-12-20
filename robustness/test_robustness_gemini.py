"""
Robustness Testing for Gemini API Model

This script evaluates robustness for Gemini predictions that were generated externally.
Since Gemini is an API model, inference is done externally - this script only:
1. Renders perturbed predictions to PNG
2. Calculates visual fidelity (CLIP + IoU) and structural alignment
3. Computes degradation rates

Folder structure expected in robustness_results/gemini_{strength}/:
- reference_images/     : Original clean PNG images (e.g., 10018.png)
- perturbed_predictions/: HTML generated from perturbed images
- clean_pred_htmls/     : HTML generated from clean images (for comparison)
- clean_htmls/          : Ground truth HTML (optional)

Usage:
    python test_robustness_gemini.py --strength 0.2
    python test_robustness_gemini.py --strength 0.2 --skip-render  # If renders exist
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


# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent / "metrics"))
from robustness import compute_robustness_metrics
from structural_alignment import (
    semantic_html_usage,
    tree_edit_similarity,
)
from CLIP import calculate_clip_score

# Required imports
try:
    from PIL import Image
except ImportError:
    print("Error: PIL not available (pip install Pillow)")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright
except ImportError:
    raise ImportError("Playwright not available. Install: pip install playwright && playwright install")


def get_sample_ids(dirs, overlap_only=False):
    """Get sample IDs from reference_images folder (filename without extension).

    If overlap_only=True, only return samples that exist in both clean_predictions
    and perturbed_predictions folders.
    """
    ref_dir = dirs["reference_images"]
    sample_ids = set()
    for f in ref_dir.glob("*.png"):
        sample_id = f.stem  # e.g., "10018" from "10018.png"
        sample_ids.add(sample_id)

    if overlap_only:
        # Get IDs from clean predictions
        clean_ids = set()
        for f in dirs["clean_predictions"].glob("*.html"):
            clean_ids.add(f.stem)

        # Get IDs from perturbed predictions
        perturbed_ids = set()
        for f in dirs["perturbed_predictions"].glob("*.html"):
            perturbed_ids.add(f.stem)

        # Intersection of all three
        sample_ids = sample_ids & clean_ids & perturbed_ids
        print(f"  Using overlap-only mode: {len(sample_ids)} samples with both clean and perturbed predictions")

    return sorted(list(sample_ids))


def setup_directories(output_dir):
    """Setup directory paths for Gemini evaluation."""
    base_dir = Path(output_dir)
    dirs = {
        "base": base_dir,
        "reference_images": base_dir / "reference_images",
        "perturbed_predictions": base_dir / "perturbed_predictions",
        "perturbed_rendered": base_dir / "perturbed_rendered",
        "clean_predictions": base_dir / "clean_pred_htmls",
        "clean_rendered": base_dir / "clean_rendered",
        "clean_htmls": base_dir / "clean_htmls",  # Ground truth HTML
    }
    # Create output directories
    dirs["perturbed_rendered"].mkdir(parents=True, exist_ok=True)
    dirs["clean_rendered"].mkdir(parents=True, exist_ok=True)
    return dirs


async def step1_render_predictions(dirs, sample_ids):
    """Render HTML predictions to PNG images."""
    print("\n" + "=" * 60)
    print("STEP 1: Rendering Predictions to PNG")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Render clean predictions
        print("  Rendering clean predictions...")
        for sample_id in tqdm(sample_ids, desc="  Clean"):
            html_path = dirs["clean_predictions"] / f"{sample_id}.html"
            output_path = dirs["clean_rendered"] / f"{sample_id}.png"

            if output_path.exists() or not html_path.exists():
                continue

            ref_path = dirs["reference_images"] / f"{sample_id}.png"
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
                print(f"\nError rendering clean {sample_id}: {e}")
                continue

        # Render perturbed predictions
        print("  Rendering perturbed predictions...")
        for sample_id in tqdm(sample_ids, desc="  Perturbed"):
            html_path = dirs["perturbed_predictions"] / f"{sample_id}.html"
            output_path = dirs["perturbed_rendered"] / f"{sample_id}.png"

            if output_path.exists() or not html_path.exists():
                continue

            ref_path = dirs["reference_images"] / f"{sample_id}.png"
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
                print(f"\nError rendering perturbed {sample_id}: {e}")
                continue

        await browser.close()

    print(f"  Clean rendered: {dirs['clean_rendered']}")
    print(f"  Perturbed rendered: {dirs['perturbed_rendered']}")


async def step2_calculate_scores(dirs, sample_ids):
    """Calculate visual fidelity (CLIP + IoU) and structural alignment scores."""
    print("\n" + "=" * 60)
    print("STEP 2: Calculating Visual Fidelity & Structural Alignment")
    print("=" * 60)
    print("  Using metrics/CLIP.py for CLIP score")
    print("  Using metrics/IOU.py for IoU score")
    print("  Using metrics/structural_alignment.py for structural alignment")

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
        "clean_tree_edit": [],
        "perturbed_tree_edit": [],
    }

    for sample_id in tqdm(sample_ids, desc="  Calculating"):
        ref_path = dirs["reference_images"] / f"{sample_id}.png"
        clean_rendered_path = dirs["clean_rendered"] / f"{sample_id}.png"
        perturbed_rendered_path = dirs["perturbed_rendered"] / f"{sample_id}.png"
        clean_html_path = dirs["clean_predictions"] / f"{sample_id}.html"
        perturbed_html_path = dirs["perturbed_predictions"] / f"{sample_id}.html"
        gt_html_path = dirs["clean_htmls"] / f"{sample_id}.html"

        # --- Visual Fidelity (CLIP) ---
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

        # --- Get ground truth HTML for IoU and Tree Edit ---
        gt_html = None
        gt_soup = None
        if gt_html_path.exists():
            try:
                with open(gt_html_path, 'r', encoding='utf-8') as f:
                    gt_html = f.read()
                gt_soup = parse_html_to_soup(gt_html)
            except Exception:
                pass

        # --- Visual Fidelity (IoU) ---
        async def calc_iou(pred_html_path, gt_html, sample_id):
            """Calculate IoU score between prediction and ground truth"""
            try:
                if not pred_html_path.exists() or gt_html is None:
                    return None
                with open(pred_html_path, 'r', encoding='utf-8') as f:
                    pred_html = f.read()
                gt_boxes = await benchmark.get_element_bboxes(gt_html, sample_id, "GT")
                pred_boxes = await benchmark.get_element_bboxes(pred_html, sample_id, "PRED")
                return benchmark.compare_layouts(gt_boxes, pred_boxes)
            except Exception:
                return None

        # Clean IoU score
        clean_iou = await calc_iou(clean_html_path, gt_html, sample_id)
        scores["clean_iou"].append(clean_iou)

        # Perturbed IoU score
        perturbed_iou = await calc_iou(perturbed_html_path, gt_html, sample_id)
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


def step3_generate_report(dirs, sample_ids, scores, model_name, strength):
    """Generate robustness evaluation report."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating Report")
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

    # Compute degradation rates
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
    report.append(f"  Total Samples:      {len(sample_ids)}")
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
        report.append(f"    - Semantic HTML: {calc_mean(scores['clean_semantic_ratio']):.4f}" if calc_mean(scores['clean_semantic_ratio']) is not None else "    - Semantic HTML: N/A")
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
    report.append(f"  Clean predictions:     {dirs['clean_predictions']}")
    report.append(f"  Clean rendered:        {dirs['clean_rendered']}")
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
        "samples": len(sample_ids),
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
                "formula": "(Semantic + Tree Edit) / 2"
            },
            "clean": {
                "mean": clean_structural_mean,
                "std": calc_std(scores["clean_structural"]),
                "count": count_valid(scores["clean_structural"]),
                "semantic_ratio_mean": calc_mean(scores["clean_semantic_ratio"]),
                "tree_edit_mean": clean_tree_edit_mean
            },
            "perturbed": {
                "mean": perturbed_structural_mean,
                "std": calc_std(scores["perturbed_structural"]),
                "count": count_valid(scores["perturbed_structural"]),
                "semantic_ratio_mean": calc_mean(scores["perturbed_semantic_ratio"]),
                "tree_edit_mean": perturbed_tree_edit_mean
            }
        },
        "degradation_rate": {
            "visual_fidelity_drop": vf_drop,
            "structural_alignment_drop": sa_drop,
            "average": avg_drop
        },
        "sample_ids": sample_ids,
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

    return robustness_results


async def main():
    parser = argparse.ArgumentParser(description="Gemini Robustness Testing")
    parser.add_argument("--strength", type=float, default=0.2,
                        help="Perturbation strength (default: 0.2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: robustness_results/gemini_{strength})")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip rendering step (use existing renders)")
    parser.add_argument("--overlap-only", action="store_true",
                        help="Only use samples that exist in both clean and perturbed predictions")
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"robustness_results/gemini_{args.strength}"

    print("=" * 60)
    print("GEMINI ROBUSTNESS TESTING")
    print("=" * 60)
    print(f"Perturbation strength: {args.strength}")
    print(f"Output directory: {output_dir}")
    print(f"Skip render: {args.skip_render}")
    print(f"Overlap only: {args.overlap_only}")

    # Setup directories
    dirs = setup_directories(output_dir)

    # Get sample IDs from reference images
    sample_ids = get_sample_ids(dirs, overlap_only=args.overlap_only)
    print(f"\nFound {len(sample_ids)} samples")

    if len(sample_ids) == 0:
        print("ERROR: No reference images found!")
        print(f"Expected PNG files in: {dirs['reference_images']}")
        return

    # Step 1: Render predictions
    if not args.skip_render:
        await step1_render_predictions(dirs, sample_ids)
    else:
        print("\n" + "=" * 60)
        print("STEP 1: Skipping Render (--skip-render)")
        print("=" * 60)

    # Step 2: Calculate scores
    scores = await step2_calculate_scores(dirs, sample_ids)

    # Step 3: Generate report
    step3_generate_report(dirs, sample_ids, scores, "gemini", args.strength)

    print("\n" + "=" * 60)
    print("ROBUSTNESS TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
