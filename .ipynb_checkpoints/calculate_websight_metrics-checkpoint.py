"""
Comprehensive Metrics Evaluation for WebSight Model
Evaluates samples SAMPLE_RANGE from results_WebSight

Uses metrics from Design2code/metrics:
- structural_alignment: Tree Edit Similarity, Semantic HTML Ratio, Accessibility Score
- efficiency: Tokens/Second, Latency, Peak VRAM
- robustness: Performance Degradation Rate
- correctness: Render Success Rate, DOM/Console Error Count
- visual_fidelity: CLIP Score

Metrics Categories:
1. User Interface Visual Fidelity: CLIP, IoU, Content & Color Consistency
2. Code Correctness: Render Success Rate, DOM/Console Error Count
3. Structural Alignment: Tree Edit Similarity, Semantic HTML Ratio, Accessibility Score
4. Robustness: Performance Degradation Rate
5. Computational Efficiency: Tokens/Second, Average Latency, Peak VRAM
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
from typing import Optional, List, Dict, Any

# Add metrics to path - import directly from files to avoid naming conflict
sys.path.insert(0, str(Path(__file__).parent / "metrics"))

# Import from Design2code/metrics (direct file imports)
from structural_alignment import (
    SEMANTIC_TAGS,
    semantic_html_usage,
    accessibility_score,
)
from efficiency import (
    EfficiencyScores,
    compute_efficiency_local,
)

# Configuration
PREDICTIONS_DIR = Path("results_WebSight/predictions")
RESULTS_DIR = Path("results_WebSight/results")
OUTPUT_FILE = Path("results_WebSight/websight_metrics_report.txt")
SAMPLE_RANGE = range(316, 484)

# Try to import optional dependencies
try:
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from torch.nn.functional import cosine_similarity
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available (torch/transformers not installed)")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available for correctness metrics")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available")

# ============================================================
# DOM Node Wrapper for Structural Alignment
# ============================================================
class DOMNode:
    """Wrapper around BeautifulSoup to match structural_alignment interface"""
    def __init__(self, soup_element):
        self.element = soup_element
        self.tag = getattr(soup_element, 'name', "") or ""
        self.attrs = dict(soup_element.attrs) if hasattr(soup_element, 'attrs') else {}
        self.children = [DOMNode(child) for child in soup_element.children
                        if hasattr(child, 'name') and child.name] if hasattr(soup_element, 'children') else []

def parse_html_to_dom(html_string: str) -> Optional[DOMNode]:
    """Parse HTML string to DOM tree"""
    try:
        soup = BeautifulSoup(html_string, 'lxml')
        body = soup.find('body')
        if body:
            return DOMNode(body)
        return DOMNode(soup) if soup else None
    except Exception as e:
        return None

# ============================================================
# Simple Tree Edit Similarity (placeholder for full implementation)
# ============================================================
def _count_nodes(node: Any) -> int:
    """Total number of nodes in a DOM tree."""
    if node is None:
        return 0
    total = 1
    for child in getattr(node, "children", []):
        total += _count_nodes(child)
    return total

def simple_tree_edit_similarity(ref_dom: Any, pred_dom: Any) -> float:
    """Simple tree similarity based on node count and tag matching."""
    if ref_dom is None and pred_dom is None:
        return 1.0
    if ref_dom is None or pred_dom is None:
        return 0.0

    ref_nodes = _count_nodes(ref_dom)
    pred_nodes = _count_nodes(pred_dom)
    max_nodes = max(ref_nodes, pred_nodes)
    if max_nodes == 0:
        return 1.0

    # Simple similarity based on node count difference
    diff = abs(ref_nodes - pred_nodes)
    return max(0.0, 1.0 - diff / max_nodes)

# ============================================================
# Code Correctness Metrics (from metrics/correctness.py logic)
# ============================================================
async def check_render_success(html_content: str) -> Dict[str, Any]:
    """Check if HTML renders successfully using Playwright."""
    if not PLAYWRIGHT_AVAILABLE:
        return {"render_success": None, "error_count": None, "critical_error": None}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        errors = []
        render_success = False

        def handle_console(msg):
            if msg.type.lower() == 'error':
                errors.append(msg.text)

        page.on('console', handle_console)

        try:
            await page.set_content(html_content, timeout=10000)

            # Check if visually empty (from metrics/correctness.py)
            is_visually_empty = await page.evaluate("""() => {
                const body = document.body;
                if (!body) return true;
                if (body.innerText.trim().length > 0) return false;
                const allElements = body.querySelectorAll('*');
                for (const el of allElements) {
                    if (['SCRIPT', 'STYLE', 'META', 'LINK', 'HEAD', 'TITLE'].includes(el.tagName)) continue;
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        const style = window.getComputedStyle(el);
                        if (style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0') {
                            return false;
                        }
                    }
                }
                return true;
            }""")

            render_success = not is_visually_empty

        except Exception as e:
            errors.append(str(e))

        await browser.close()

        return {
            "render_success": render_success,
            "error_count": len(errors),
            "critical_error": not render_success
        }

# ============================================================
# Visual Fidelity Metrics (from metrics/visual_fidelity/CLIP.py logic)
# ============================================================
def init_clip_model():
    """Initialize CLIP model."""
    if not CLIP_AVAILABLE:
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    try:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor, device
    except:
        return None, None, None

def calculate_clip_score(image1_path: str, image2_path: str, model, processor, device) -> Optional[float]:
    """Calculate CLIP similarity between two images."""
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")

        inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=pixel_values)

        emb1 = outputs[0].unsqueeze(0)
        emb2 = outputs[1].unsqueeze(0)

        similarity = cosine_similarity(emb1, emb2)
        return similarity.item()
    except Exception as e:
        return None

# ============================================================
# Helper Functions
# ============================================================
def load_sample_results(sample_idx: int) -> Optional[Dict]:
    """Load individual sample result JSON."""
    json_path = RESULTS_DIR / f"{sample_idx}.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def load_prediction_html(sample_idx: int) -> Optional[str]:
    """Load prediction HTML file."""
    html_path = PREDICTIONS_DIR / f"{sample_idx}.html"
    if html_path.exists():
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            pass
    return None

def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
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
# Main Evaluation
# ============================================================
async def main():
    print("=" * 60)
    print("WebSight Metrics Evaluation")
    print("=" * 60)

    # Load dataset for ground truth
    ground_truths = {}
    dataset_total_size = 484  # Default if dataset loading fails
    if DATASETS_AVAILABLE:
        print("\nLoading ground truth dataset...")
        try:
            dataset = load_dataset("SALT-NLP/Design2Code-hf", split="train")
            dataset_total_size = len(dataset)
            for idx in SAMPLE_RANGE:
                if idx < len(dataset):
                    ground_truths[idx] = dataset[idx]["text"]
            print(f"  Loaded {len(ground_truths)} ground truth samples")
            print(f"  Total dataset size: {dataset_total_size}")
        except Exception as e:
            print(f"  Failed to load dataset: {e}")

    # Initialize metrics storage
    metrics = {
        # Efficiency
        "tokens_per_second": [],
        "latency": [],
        "peak_vram_gb": [],
        # Structural Alignment
        "tree_edit_similarity": [],
        "semantic_html_ratio": [],
        "accessibility_score": [],
        # Correctness
        "render_success": [],
        "error_count": [],
        "critical_error_count": [],
        # Visual Fidelity
        "clip_score": [],
    }

    samples_found = 0
    samples_missing = 0

    print("\nProcessing samples...")
    for idx in tqdm(SAMPLE_RANGE, desc="Evaluating"):
        # Load prediction HTML
        pred_html = load_prediction_html(idx)
        if pred_html is None:
            samples_missing += 1
            continue

        samples_found += 1

        # Load saved results if available (for efficiency metrics)
        saved_result = load_sample_results(idx)

        # Efficiency metrics from saved results
        if saved_result:
            if saved_result.get("tokens_per_second"):
                metrics["tokens_per_second"].append(saved_result["tokens_per_second"])
            if saved_result.get("latency"):
                metrics["latency"].append(saved_result["latency"])
            if saved_result.get("peak_vram_gb"):
                metrics["peak_vram_gb"].append(saved_result["peak_vram_gb"])

        # Structural Alignment metrics using imported functions
        pred_dom = parse_html_to_dom(pred_html)
        if pred_dom:
            # Use imported semantic_html_usage from metrics/structural_alignment.py
            metrics["semantic_html_ratio"].append(semantic_html_usage(pred_dom))
            # Use imported accessibility_score from metrics/structural_alignment.py
            metrics["accessibility_score"].append(accessibility_score(pred_dom))

            # Tree edit similarity (requires ground truth)
            if idx in ground_truths:
                ref_dom = parse_html_to_dom(ground_truths[idx])
                if ref_dom:
                    metrics["tree_edit_similarity"].append(
                        simple_tree_edit_similarity(ref_dom, pred_dom)
                    )

    # Run correctness checks on ALL samples
    if PLAYWRIGHT_AVAILABLE:
        print("\nRunning correctness checks on all samples...")
        for idx in tqdm(SAMPLE_RANGE, desc="Checking renders"):
            pred_html = load_prediction_html(idx)
            if pred_html:
                try:
                    result = await check_render_success(pred_html)
                    if result["render_success"] is not None:
                        metrics["render_success"].append(1 if result["render_success"] else 0)
                        metrics["error_count"].append(result["error_count"])
                        metrics["critical_error_count"].append(1 if result["critical_error"] else 0)
                except Exception as e:
                    pass

    # Compute statistics
    print("\nComputing statistics...")
    stats = {}
    for metric_name, values in metrics.items():
        stats[metric_name] = compute_stats(values)

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("WEBSIGHT MODEL METRICS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    report.append("DATASET COVERAGE:")
    report.append(f"  Total Dataset Size: {dataset_total_size}")
    report.append(f"  Evaluation Range:   0-315 ({len(SAMPLE_RANGE)} samples)")
    report.append(f"  Samples Found:      {samples_found} / {len(SAMPLE_RANGE)}")
    report.append(f"  Samples Missing:    {samples_missing}")
    report.append(f"  Coverage:           {samples_found} / {dataset_total_size} ({100*samples_found/dataset_total_size:.1f}%)")
    report.append("=" * 60)

    # 1. User Interface Visual Fidelity
    report.append("\n" + "=" * 60)
    report.append("1. USER INTERFACE VISUAL FIDELITY")
    report.append("=" * 60)
    report.append("\nCLIP Score (range: [0, 1], higher is better):")
    s = stats.get("clip_score", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (Not computed - requires rendering pipeline)")

    report.append("\nIoU (range: [0, 1], higher is better):")
    report.append("  (Not computed - requires rendered images)")

    report.append("\nContent & Color Consistency (range: [0, 1], higher is better):")
    report.append("  (Not computed - requires rendered images)")

    # 2. Code Correctness
    report.append("\n" + "=" * 60)
    report.append("2. CODE CORRECTNESS")
    report.append("=" * 60)

    report.append("\nRender Success Rate (range: [0%, 100%], higher is better):")
    s = stats.get("render_success", {})
    if s.get("count", 0) > 0:
        report.append(f"  Success Rate: {s['mean']*100:.2f}%")
        report.append(f"  Total Tested: {s['count']}")
    else:
        report.append("  (Not computed - requires Playwright)")

    report.append("\nDOM/Console Error Count:")
    s = stats.get("error_count", {})
    if s.get("count", 0) > 0:
        total_errors = sum(metrics['error_count'])
        files_with_errors = sum(1 for v in metrics['error_count'] if v > 0)
        report.append(f"  Total Errors:       {int(total_errors)}")
        report.append(f"  Files with errors:  {files_with_errors}")
        report.append(f"  Mean per file:      {s['mean']:.2f}")
        report.append(f"  Max in single file: {int(s['max'])}")
        report.append(f"  Files tested:       {s['count']}")
    else:
        report.append("  (Not computed - requires Playwright)")

    report.append("\nCritical Error Count:")
    s = stats.get("critical_error_count", {})
    if s.get("count", 0) > 0:
        total_critical = sum(metrics['critical_error_count'])
        report.append(f"  Total Critical Errors:       {int(total_critical)}")
        report.append(f"  Files with critical errors:  {int(total_critical)}")
        report.append(f"  Files tested:                {s['count']}")
    else:
        report.append("  (Not computed - requires Playwright)")

    # 3. Structural Alignment
    report.append("\n" + "=" * 60)
    report.append("3. STRUCTURAL ALIGNMENT")
    report.append("=" * 60)

    report.append("\nTree Edit Similarity (range: [0, 1], higher is better):")
    s = stats.get("tree_edit_similarity", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (Not computed - requires ground truth)")

    report.append("\nSemantic HTML Ratio (range: [0, 1], higher is better):")
    s = stats.get("semantic_html_ratio", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data)")

    report.append("\nAccessibility Score (range: [0, 1], higher is better):")
    s = stats.get("accessibility_score", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.4f}")
        report.append(f"  Std:   {s['std']:.4f}")
        report.append(f"  Min:   {s['min']:.4f}")
        report.append(f"  Max:   {s['max']:.4f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data)")

    # 4. Robustness
    report.append("\n" + "=" * 60)
    report.append("4. ROBUSTNESS")
    report.append("=" * 60)
    report.append("\nPerformance Degradation Rate (range: [0, 1], lower is better):")
    report.append("  (Not computed - requires perturbed image testing)")

    # 5. Computational Efficiency
    report.append("\n" + "=" * 60)
    report.append("5. COMPUTATIONAL EFFICIENCY")
    report.append("=" * 60)

    report.append("\nTokens/Second (range: [0, +∞), higher is better):")
    s = stats.get("tokens_per_second", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.2f}")
        report.append(f"  Std:   {s['std']:.2f}")
        report.append(f"  Min:   {s['min']:.2f}")
        report.append(f"  Max:   {s['max']:.2f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data in saved results)")

    report.append("\nAverage Latency in seconds (range: [0, +∞), lower is better):")
    s = stats.get("latency", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.2f}")
        report.append(f"  Std:   {s['std']:.2f}")
        report.append(f"  Min:   {s['min']:.2f}")
        report.append(f"  Max:   {s['max']:.2f}")
        report.append(f"  Count: {s['count']}")
    else:
        report.append("  (No data in saved results)")

    report.append("\nPeak VRAM in GB (range: [0, +∞), lower is better):")
    s = stats.get("peak_vram_gb", {})
    if s.get("count", 0) > 0:
        report.append(f"  Mean:  {s['mean']:.2f}")
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
    print(report_text)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
