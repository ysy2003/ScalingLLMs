"""
Metrics package for ScalingLLMs evaluation.

Contains:
- structural_alignment: Tree edit distance, semantic HTML, accessibility (@ysy)
- robustness: Performance degradation rate under perturbations
- efficiency: Tokens/sec, VRAM, latency, and API cost
"""

from .structural_alignment import (
    StructuralAlignmentScores,
    SEMANTIC_TAGS,
    tree_edit_distance,
    tree_edit_similarity,
    semantic_html_usage,
    accessibility_score,
    compute_structural_alignment_scores,
    compute_overall_structural_alignment,
)

from .robustness import (
    RobustnessScores,
    performance_degradation_rate,
    compute_robustness_metrics,
)

from .efficiency import (
    EfficiencyScores,
    compute_efficiency_local,
    compute_efficiency_api,
)


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
]
