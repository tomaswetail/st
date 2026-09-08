"""Parameters for Stryktipset coupon optimization.

Modes:
- PREDICTION (default): market Pm only; public % diagnostic.
- VALUE: EV path Σ log Pm − β Σ log Pp + Hamming diversity.

Leakage UNKNOWN: odds and public % have no timestamps in storage;
regCloseTime exists in the Svenska Spel API but is not persisted on STRoundModel.
Treat stored snapshots as operable coupon-time data; do not claim closing-line safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptimizerMode = Literal["PREDICTION", "VALUE"]
PredictionObjective = Literal[
    "MAX_P13",
    "MAX_P12_OR_BETTER",
    "MAX_P11_OR_BETTER",
    "MAX_EXPECTED_CORRECT",
]

VALID_MODES: frozenset[str] = frozenset({"PREDICTION", "VALUE"})
VALID_OBJECTIVES: frozenset[str] = frozenset(
    {
        "MAX_P13",
        "MAX_P12_OR_BETTER",
        "MAX_P11_OR_BETTER",
        "MAX_EXPECTED_CORRECT",
    }
)


@dataclass(frozen=True)
class OptimizerParams:
    """Tunable knobs for mode, objective, VALUE scoring, and search."""

    mode: OptimizerMode = "PREDICTION"
    objective: PredictionObjective = "MAX_P13"
    # VALUE-only knobs (ignored for PREDICTION row selection):
    beta: float = 1.0
    lambda_diversity: float = 0.5
    banker_value_weight: float = 1.0
    candidate_count: int = 500
    # PREDICTION approx candidate pool size:
    prediction_candidate_count: int = 2000
    public_epsilon: float = 1e-6
    coupon_size: int = 13
    seed: int | None = None
    # When True, build doubles/triples reduced system instead of free rows.
    reduced_system: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(VALID_OBJECTIVES)}"
            )
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
        if self.prediction_candidate_count < 1:
            raise ValueError(
                "prediction_candidate_count must be >= 1, "
                f"got {self.prediction_candidate_count}"
            )
        if self.public_epsilon <= 0:
            raise ValueError(
                f"public_epsilon must be > 0, got {self.public_epsilon}"
            )
        if self.coupon_size < 1:
            raise ValueError(f"coupon_size must be >= 1, got {self.coupon_size}")
        if self.reduced_system and self.mode != "PREDICTION":
            raise ValueError("reduced_system is only valid in PREDICTION mode")
