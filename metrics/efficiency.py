"""
Computational Efficiency metrics

- For Local Models: tokens-per-second and peak VRAM usage
- For API Models: end-to-end latency and relative API cost ($ per 1,000 tokens)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EfficiencyScores:
    """Computational efficiency metrics."""
    tokens_per_second: Optional[float] = None   # for local models
    latency_seconds: Optional[float] = None     # for both local & API
    cost_per_1k_tokens: Optional[float] = None  # for API models


def compute_efficiency_local(
    generated_tokens: int,
    wall_time_sec: float,
    latency_sec: Optional[float] = None,
) -> EfficiencyScores:
    """
    Efficiency metrics for local models.
    - generated_tokens: number of output tokens
    - wall_time_sec: total generation time
    - peak_vram_bytes: torch.cuda.max_memory_allocated() or similar
    - latency_sec: optional; if None, defaults to wall_time_sec
    """
    if wall_time_sec <= 0:
        tps = 0.0
    else:
        tps = generated_tokens / wall_time_sec


    return EfficiencyScores(
        tokens_per_second=tps,
        latency_seconds=latency_sec if latency_sec is not None else wall_time_sec,
        cost_per_1k_tokens=None,
    )


def compute_efficiency_api(
    latency_sec: float,
    total_tokens: int,
    price_per_1k_tokens: float,
) -> EfficiencyScores:
    """
    Efficiency metrics for API models.
    - latency_sec: end-to-end latency for the request
    - total_tokens: prompt + completion tokens billed
    - price_per_1k_tokens: USD per 1,000 tokens (model-specific)
    """
    cost = (total_tokens / 1000.0) * price_per_1k_tokens

    return EfficiencyScores(
        tokens_per_second=None,
        latency_seconds=latency_sec,
        cost_per_1k_tokens=cost,
    )


__all__ = [
    "EfficiencyScores",
    "compute_efficiency_local",
    "compute_efficiency_api",
]
