"""Parameters for Stryktipset coupon optimization.

Leakage UNKNOWN: odds and public % have no timestamps in storage;
regCloseTime exists in the Svenska Spel API but is not persisted on STRoundModel.
Treat stored snapshots as operable coupon-time data; do not claim closing-line safety.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerParams:
    """Tunable knobs for row scoring, search, and portfolio selection."""

    beta: float = 1.0
    lambda_diversity: float = 0.5
    banker_value_weight: float = 1.0
    candidate_count: int = 500
    public_epsilon: float = 1e-6
    coupon_size: int = 13
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.lambda_diversity < 0:
            raise ValueError(
                f"lambda_diversity must be >= 0, got {self.lambda_diversity}"
            )
        if self.candidate_count < 1:
            raise ValueError(
                f"candidate_count must be >= 1, got {self.candidate_count}"
            )
        if self.public_epsilon <= 0:
            raise ValueError(
                f"public_epsilon must be > 0, got {self.public_epsilon}"
            )
        if self.coupon_size < 1:
            raise ValueError(f"coupon_size must be >= 1, got {self.coupon_size}")
