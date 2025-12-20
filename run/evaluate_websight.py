"""
Comprehensive Metrics Evaluation for WebSight Model
Evaluates all metrics on existing predictions in results_WebSight/

Metrics:
1. Visual Fidelity: CLIP Score, IoU
2. Code Correctness: Render Success Rate, Error Count
3. Structural Alignment: Tree Edit Similarity, Semantic HTML Ratio, Accessibility
4. Robustness: Performance Degradation Rate (if perturbed results exist)
5. Computational Efficiency: Tokens/Second, Latency, Peak VRAM
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
import pandas as pd

# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent / "metrics"))

from structural_alignment import (
    semantic_html_usage,
    accessibility_score,
    tree_edit_similarity,
)

# Configuration
PREDICTIONS_DIR = Path("results_WebSight/predictions")
RESULTS_DIR = Path("results_WebSight/results")
RENDERED_PRED_DIR = Path("results_WebSight/rendered_imgs")
RENDERED_GT_DIR = Path("results_WebSight/reference_imgs")
OUTPUT_FILE = Path("results_WebSight/websight_metrics_report.txt")
SAMPLE_RANGE = range(0, 484)

# Import CLIP from metrics
try:
    from CLIP import calculate_clip_score
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available (pip install playwright && playwright install)")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available (pip install datasets)")


# ============================================================
# Helper Functions
# ============================================================
class DOMNode:
    """Wrapper around BeautifulSoup for structural alignment"""
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
    except:
        return None


def load_prediction_html(idx):
    """Load prediction HTML file"""
    html_path = PREDICTIONS_DIR / f"{idx}.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def load_sample_results(idx):
    """Load individual sample result JSON"""
    json_path = RESULTS_DIR / f"{idx}.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


# Load all results from Excel file (for efficiency metrics)
_ALL_RESULTS_DF = None

def get_all_results_df():
    """Load and cache the detailed_results.xlsx DataFrame"""
    global _ALL_RESULTS_DF
    if _ALL_RESULTS_DF is None:
        excel_path = RESULTS_DIR / "detailed_results.xlsx"
        if excel_path.exists():
            try:
                _ALL_RESULTS_DF = pd.read_excel(excel_path)
                _ALL_RESULTS_DF = _ALL_RESULTS_DF.set_index('number')
            except Exception as e:
                print(f"Warning: Could not load {excel_path}: {e}")
                _ALL_RESULTS_DF = pd.DataFrame()
    return _ALL_RESULTS_DF


def load_sample_from_excel(idx):
    """Load sample result from detailed_results.xlsx"""
    df = get_all_results_df()
    if df is not None and not df.empty and idx in df.index:
        return df.loc[idx].to_dict()
    return None


def compute_stats(values):
    """Compute statistics for a list of values"""
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values)
    }


# ============================================================
# CLIP Score Calculation (using metrics/CLIP.py)
# ============================================================
def calculate_clip_scores():
    """Calculate CLIP scores using metrics/CLIP.py"""
    if not CLIP_AVAILABLE:
        print("  Skipping CLIP (dependencies not available)")
        return []

    if not RENDERED_GT_DIR.exists() or not RENDERED_PRED_DIR.exists():
        print("  Skipping CLIP (rendered images not found)")
        return []

    print("  Using metrics/CLIP.py for CLIP scores...")

    scores = []
    for idx in tqdm(SAMPLE_RANGE, desc="  CLIP"):
        ref_path = RENDERED_GT_DIR / f"{idx}.png"
        pred_path = RENDERED_PRED_DIR / f"{idx}.png"

        if not ref_path.exists() or not pred_path.exists():
            scores.append(None)
            continue

        try:
            img_ref = Image.open(ref_path)
            img_pred = Image.open(pred_path)
            score = calculate_clip_score(img_ref, img_pred)
            scores.append(score)
        except Exception as e:
            scores.append(None)

    return scores


# ============================================================
# IoU Calculation
# ============================================================
async def calculate_iou_scores(ground_truths):
    """Calculate IoU scores for all samples"""
    if not PLAYWRIGHT_AVAILABLE:
        print("  Skipping IoU (Playwright not available)")
        return []

    scores = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for idx in tqdm(SAMPLE_RANGE, desc="  IoU"):
            pred_html = load_prediction_html(idx)
            if pred_html is None or idx not in ground_truths:
                scores.append(None)
                continue

            gt_html = ground_truths[idx]

            try:
                gt_boxes = await get_element_bboxes(browser, gt_html)
                pred_boxes = await get_element_bboxes(browser, pred_html)
                iou = compute_layout_iou(gt_boxes, pred_boxes)
                scores.append(iou)
            except:
                scores.append(None)

        await browser.close()

    return scores


async def get_element_bboxes(browser, html_content):
    """Extract bounding boxes from HTML"""
    page = await browser.new_page()
    await page.set_viewport_size({"width": 1280, "height": 800})

    try:
        await page.set_content(html_content, wait_until='load', timeout=5000)
    except:
        pass

    elements = await page.evaluate('''() => {
        const results = [];
        const allElements = document.querySelectorAll('body *');
        allElements.forEach((el) => {
            const invalidTags = ['SCRIPT', 'STYLE', 'NOSCRIPT', 'HEAD', 'META', 'TITLE', 'LINK', 'BR'];
            if (invalidTags.includes(el.tagName)) return;
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                results.push({x: rect.x, y: rect.y, width: rect.width, height: rect.height});
            }
        });
        return results;
    }''')

    await page.close()
    return elements


def compute_layout_iou(gt_boxes, pred_boxes):
    """Compute average IoU between layouts"""
    if not gt_boxes or not pred_boxes:
        return 0.0

    total_iou = 0.0
    for gt_box in gt_boxes:
        max_iou = 0.0
        for pred_box in pred_boxes:
            iou = box_iou(gt_box, pred_box)
            max_iou = max(max_iou, iou)
        total_iou += max_iou

    return total_iou / len(gt_boxes)


def box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ============================================================
# Correctness Check
# ============================================================
async def check_correctness():
    """Check render success and errors for all samples"""
    if not PLAYWRIGHT_AVAILABLE:
        print("  Skipping correctness (Playwright not available)")
        return [], [], []

    render_success = []
    error_counts = []
    critical_errors = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for idx in tqdm(SAMPLE_RANGE, desc="  Correctness"):
            html_path = PREDICTIONS_DIR / f"{idx}.html"
            if not html_path.exists():
                render_success.append(None)
                error_counts.append(None)
                critical_errors.append(None)
                continue

            context = await browser.new_context()
            page = await context.new_page()
            errors = []
            success = False

            try:
                page.on('console', lambda msg: errors.append(msg.text) if msg.type.lower() == 'error' else None)

                file_url = f'file://{html_path.resolve()}'
                await page.goto(file_url, wait_until='load', timeout=10000)

                is_empty = await page.evaluate('''() => {
                    const body = document.body;
                    if (!body) return true;
                    if (body.innerText.trim().length > 0) return false;
                    const allElements = body.querySelectorAll('*');
                    for (const el of allElements) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) return false;
                    }
                    return true;
                }''')
                success = not is_empty
            except Exception as e:
                errors.append(str(e))

            await context.close()

            render_success.append(1 if success else 0)
            error_counts.append(len(errors))
            critical_errors.append(0 if success else 1)

        await browser.close()

    return render_success, error_counts, critical_errors


# ============================================================
# Main Evaluation
# ============================================================
async def main():
    print("=" * 60)
    print("WEBSIGHT MODEL - COMPREHENSIVE METRICS EVALUATION")
    print("=" * 60)
    print(f"Predictions: {PREDICTIONS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Output: {OUTPUT_FILE}")

    # Initialize metrics storage
    metrics = {
        "clip_score": [],
        "iou_score": [],
        "render_success": [],
        "error_count": [],
        "critical_error_count": [],
        "tree_edit_similarity": [],
        "semantic_html_ratio": [],
        "accessibility_score": [],
        "tokens_per_second": [],
        "latency": [],
        "peak_vram_gb": [],
    }

    samples_found = 0

    # Load dataset for ground truth
    ground_truths = {}
    if DATASETS_AVAILABLE:
        print("\nLoading ground truth dataset...")
        dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
        for idx in SAMPLE_RANGE:
            if idx < len(dataset):
                ground_truths[idx] = dataset[idx]["text"]
        print(f"  Loaded {len(ground_truths)} ground truth samples")

    # 1. CLIP Scores
    print("\n[1/5] Visual Fidelity - CLIP Score")
    metrics["clip_score"] = calculate_clip_scores()

    # 2. IoU Scores
    print("\n[2/5] Visual Fidelity - IoU Score")
    metrics["iou_score"] = await calculate_iou_scores(ground_truths)

    # 3. Correctness
    print("\n[3/5] Code Correctness")
    render_success, error_counts, critical_errors = await check_correctness()
    metrics["render_success"] = render_success
    metrics["error_count"] = error_counts
    metrics["critical_error_count"] = critical_errors

    # 4. Structural Alignment
    print("\n[4/5] Structural Alignment")
    for idx in tqdm(SAMPLE_RANGE, desc="  Structural"):
        pred_html = load_prediction_html(idx)

        if pred_html:
            samples_found += 1
            pred_dom = parse_html_to_dom(pred_html)

            if pred_dom:
                metrics["semantic_html_ratio"].append(semantic_html_usage(pred_dom))
                metrics["accessibility_score"].append(accessibility_score(pred_dom))

                if idx in ground_truths:
                    try:
                        gt_soup = BeautifulSoup(ground_truths[idx], 'lxml')
                        pred_soup = BeautifulSoup(pred_html, 'lxml')
                        gt_body = gt_soup.find('body') or gt_soup
                        pred_body = pred_soup.find('body') or pred_soup
                        sim = tree_edit_similarity(gt_body, pred_body)
                        metrics["tree_edit_similarity"].append(sim)
                    except:
                        metrics["tree_edit_similarity"].append(None)

    # 5. Efficiency (from saved results)
    print("\n[5/5] Computational Efficiency")
    for idx in tqdm(SAMPLE_RANGE, desc="  Efficiency"):
        # Try JSON first, fall back to Excel
        saved_result = load_sample_results(idx)
        if not saved_result:
            saved_result = load_sample_from_excel(idx)
        if saved_result:
            if saved_result.get("tokens_per_second"):
                metrics["tokens_per_second"].append(saved_result["tokens_per_second"])
            if saved_result.get("latency"):
                metrics["latency"].append(saved_result["latency"])
            if saved_result.get("peak_vram_gb"):
                metrics["peak_vram_gb"].append(saved_result["peak_vram_gb"])

    # Compute statistics
    print("\nComputing statistics...")
    stats = {name: compute_stats(values) for name, values in metrics.items()}

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("WEBSIGHT MODEL - FULL METRICS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    report.append("DATASET COVERAGE:")
    report.append(f"  Total Dataset Size:     484")
    report.append(f"  Evaluation Range:       {SAMPLE_RANGE.start}-{SAMPLE_RANGE.stop-1}")
    report.append(f"  Samples Found:          {samples_found}")

    # 1. Visual Fidelity
    report.append("\n" + "=" * 60)
    report.append("1. USER INTERFACE VISUAL FIDELITY")
    report.append("=" * 60)

    report.append("\nCLIP Score (range: [0, 1], higher is better):")
    s = stats["clip_score"]
    if s["count"] > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (Not computed - requires rendered images)")

    report.append("\nIoU Score (range: [0, 1], higher is better):")
    s = stats["iou_score"]
    if s["count"] > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (Not computed - requires Playwright)")

    # 2. Code Correctness
    report.append("\n" + "=" * 60)
    report.append("2. CODE CORRECTNESS")
    report.append("=" * 60)

    report.append("\nRender Success Rate (range: [0%, 100%], higher is better):")
    s = stats["render_success"]
    if s["count"] > 0:
        report.append(f"  Success Rate: {s['mean']*100:.2f}%")
        report.append(f"  Total Tested: {s['count']}")
    else:
        report.append("  (Not computed - requires Playwright)")

    report.append("\nError Count (range: [0, ∞), lower is better):")
    s = stats["error_count"]
    if s["count"] > 0:
        total_errors = sum(v for v in metrics['error_count'] if v is not None)
        files_with_errors = sum(1 for v in metrics['error_count'] if v is not None and v > 0)
        report.append(f"  Total Errors:       {int(total_errors)}")
        report.append(f"  Files with errors:  {files_with_errors}")
        report.append(f"  Mean per file:      {s['mean']:.2f}")
        report.append(f"  Max in single file: {int(s['max'])}")
    else:
        report.append("  (Not computed)")

    report.append("\nCritical Error Count (range: [0, ∞), lower is better):")
    s = stats["critical_error_count"]
    if s["count"] > 0:
        total_critical = sum(v for v in metrics['critical_error_count'] if v is not None)
        report.append(f"  Total Critical:     {int(total_critical)}")
    else:
        report.append("  (Not computed)")

    # 3. Structural Alignment
    report.append("\n" + "=" * 60)
    report.append("3. STRUCTURAL ALIGNMENT")
    report.append("=" * 60)

    for metric_name, display_name in [
        ("tree_edit_similarity", "Tree Edit Similarity"),
        ("semantic_html_ratio", "Semantic HTML Ratio"),
        ("accessibility_score", "Accessibility Score")
    ]:
        report.append(f"\n{display_name} (range: [0, 1], higher is better):")
        s = stats[metric_name]
        if s["count"] > 0:
            report.append(f"  Mean:  {s['mean']:.4f}")
            report.append(f"  Std:   {s['std']:.4f}")
            report.append(f"  Min:   {s['min']:.4f}")
            report.append(f"  Max:   {s['max']:.4f}")
            report.append(f"  Count: {s['count']}")
        else:
            report.append("  (Not computed)")

    # 4. Robustness
    report.append("\n" + "=" * 60)
    report.append("4. ROBUSTNESS")
    report.append("=" * 60)
    report.append("\nPerformance Degradation Rate (range: [0%, 100%], lower is better):")

    # Check for robustness results
    robustness_report = Path("robustness_results/websight/robustness_report.json")
    if robustness_report.exists():
        with open(robustness_report) as f:
            rob_data = json.load(f)
        deg = rob_data.get("degradation_rate")
        if deg is not None:
            # New format: deg is a dict with visual_fidelity_drop, structural_alignment_drop, average
            if isinstance(deg, dict):
                report.append(f"  Visual Fidelity Drop:      {deg['visual_fidelity_drop']*100:.2f}%")
                report.append(f"  Structural Alignment Drop: {deg['structural_alignment_drop']*100:.2f}%")
                report.append(f"  Average Degradation:       {deg['average']*100:.2f}%")
                report.append("")
                report.append(f"  Visual Fidelity - Clean Mean:     {rob_data['visual_fidelity']['clean']['mean']:.4f}")
                report.append(f"  Visual Fidelity - Perturbed Mean: {rob_data['visual_fidelity']['perturbed']['mean']:.4f}")
                report.append(f"  Structural Align - Clean Mean:    {rob_data['structural_alignment']['clean']['mean']:.4f}")
                report.append(f"  Structural Align - Perturbed Mean:{rob_data['structural_alignment']['perturbed']['mean']:.4f}")
            # Old format: deg is a float (CLIP-only degradation rate)
            else:
                report.append(f"  Degradation Rate: {deg*100:.2f}% (CLIP-only, re-run test_robustness.py for full metrics)")
                report.append(f"  Clean CLIP Mean:  {rob_data['clean_clip']['mean']:.4f}")
                report.append(f"  Perturbed CLIP Mean: {rob_data['perturbed_clip']['mean']:.4f}")
            report.append(f"  Samples Tested: {rob_data['samples']}")
        else:
            report.append("  (Robustness test incomplete)")
    else:
        report.append("  (Requires perturbed image testing - run test_robustness.py)")

    # 5. Computational Efficiency
    report.append("\n" + "=" * 60)
    report.append("5. COMPUTATIONAL EFFICIENCY")
    report.append("=" * 60)

    report.append("\nTokens/Second (range: [0, ∞), higher is better):")
    s = stats["tokens_per_second"]
    if s["count"] > 0:
        report.append(f"  Mean:  {s['mean']:.2f} tok/s")
        report.append(f"  Std:   {s['std']:.2f}")
        report.append(f"  Min:   {s['min']:.2f}")
        report.append(f"  Max:   {s['max']:.2f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data in saved results)")

    report.append("\nAverage Latency (range: [0, ∞) seconds, lower is better):")
    s = stats["latency"]
    if s["count"] > 0:
        report.append(f"  Mean:  {s['mean']:.2f} seconds")
        report.append(f"  Std:   {s['std']:.2f}")
        report.append(f"  Min:   {s['min']:.2f}")
        report.append(f"  Max:   {s['max']:.2f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data in saved results)")

    report.append("\nPeak VRAM (range: [0, ∞) GB, lower is better):")
    s = stats["peak_vram_gb"]
    if s["count"] > 0:
        report.append(f"  Mean:  {s['mean']:.2f} GB")
        report.append(f"  Std:   {s['std']:.2f}")
        report.append(f"  Min:   {s['min']:.2f}")
        report.append(f"  Max:   {s['max']:.2f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data in saved results)")

    report.append("\n" + "=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)

    # Write report
    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {OUTPUT_FILE}")

    # Also save as JSON
    json_output = {
        "model": "WebSight (HuggingFaceM4/VLM_WebSight_finetuned)",
        "generated": datetime.now().isoformat(),
        "samples_evaluated": samples_found,
        "metrics": stats
    }
    json_file = OUTPUT_FILE.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to: {json_file}")


if __name__ == "__main__":
    asyncio.run(main())
