"""
Robustness metrics

Currently implements Performance Degradation Rate for:
- visual_fidelity
- structural_alignment
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RobustnessScores:
    """Robustness metrics under perturbations."""
    visual_fidelity_drop: float      # percentage drop, e.g. 0.25 = 25%
    structural_alignment_drop: float # percentage drop


def performance_degradation_rate(
    clean_scores: Dict[str, float],
    perturbed_scores: Dict[str, float],
) -> RobustnessScores:
    """
    Compute degradation rates for:
      - visual_fidelity
      - structural_alignment

    Each rate is:
        (clean - perturbed) / clean
    and is capped at [0, 1] for interpretability.
    `clean_scores` and `perturbed_scores` should at least contain:
        - "visual_fidelity"
        - "structural_alignment"
    """
    vf_clean = clean_scores.get("visual_fidelity", 0.0)
    vf_pert  = perturbed_scores.get("visual_fidelity", 0.0)
    sa_clean = clean_scores.get("structural_alignment", 0.0)
    sa_pert  = perturbed_scores.get("structural_alignment", 0.0)

    def _rate(clean: float, pert: float) -> float:
        if clean <= 0:
            return 0.0
        r = (clean - pert) / clean
        # clamp to [0, 1] for readability
        return max(0.0, min(1.0, r))

    vf_drop = _rate(vf_clean, vf_pert)
    sa_drop = _rate(sa_clean, sa_pert)

    return RobustnessScores(
        visual_fidelity_drop=vf_drop,
        structural_alignment_drop=sa_drop,
    )


def compute_robustness_metrics(
    clean_metrics: Dict[str, float],
    perturbed_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    High-level hook to compute robustness-related metrics.
    For now, we only implement Performance Degradation Rate.
    You can extend this with:
      - AURC (area under robustness curve)
      - worst-case score, etc.
    """
    degr = performance_degradation_rate(clean_metrics, perturbed_metrics)
    return {
        "visual_fidelity_drop": degr.visual_fidelity_drop,
        "structural_alignment_drop": degr.structural_alignment_drop,
    }


__all__ = [
    "RobustnessScores",
    "performance_degradation_rate",
    "compute_robustness_metrics",
]
