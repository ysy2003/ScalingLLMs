"""
metrics.py (aggregator)

This file re-exports metrics grouped into three modules:
- structural_alignment.py (@ysy)
- robustness.py
- efficiency.py

Visual fidelity placeholders remain here for future implementation.
"""

from typing import Any, Dict

from metrics.structural_alignment import (
    StructuralAlignmentScores,
    SEMANTIC_TAGS,
    tree_edit_distance,
    tree_edit_similarity,
    semantic_html_usage,
    accessibility_score,
    compute_structural_alignment_scores,
    compute_overall_structural_alignment,
)

from metrics.robustness import (
    RobustnessScores,
    performance_degradation_rate,
    compute_robustness_metrics,
)

from metrics.efficiency import (
    EfficiencyScores,
    compute_efficiency_local,
    compute_efficiency_api,
)


# ---------------------------------------------------------
# Visual fidelity placeholders (to be implemented later)
# ---------------------------------------------------------

def compute_visual_fidelity_scores(
    ref_image_path: str,
    pred_image_path: str,
) -> Dict[str, float]:
    """
    TODO: implement visual fidelity metrics:
      - "clip_similarity"
      - "layout_iou"
      - "text_consistency"
      - "color_consistency"
      - "block_matching"
    Return a dict of named scores in [0, 1].
    """
    raise NotImplementedError


__all__ = [
    # structural alignment
    "StructuralAlignmentScores",
    "SEMANTIC_TAGS",
    "tree_edit_distance",
    "tree_edit_similarity",
    "semantic_html_usage",
    "accessibility_score",
    "compute_structural_alignment_scores",
    "compute_overall_structural_alignment",
    # robustness
    "RobustnessScores",
    "performance_degradation_rate",
    "compute_robustness_metrics",
    # efficiency
    "EfficiencyScores",
    "compute_efficiency_local",
    "compute_efficiency_api",
    # visual fidelity placeholder
    "compute_visual_fidelity_scores",
]
