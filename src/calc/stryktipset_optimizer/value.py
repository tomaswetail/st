"""Value ratios and log-leverage between market (Pm) and public (Pp)."""

from __future__ import annotations

import math
from typing import Mapping

from src.utils.common import OUTCOMES, Outcome


def value_ratio(pm: float, pp: float) -> float:
    """Pm / Pp. Rejects non-positive Pp."""
    if pp <= 0:
        raise ValueError(f"Pp must be > 0 for value_ratio, got {pp}")
    if pm < 0:
        raise ValueError(f"Pm must be >= 0 for value_ratio, got {pm}")
    return pm / pp


def log_leverage(pm: float, pp: float) -> float:
    """log(Pm / Pp)."""
    return math.log(value_ratio(pm, pp))


def value_ratios(
    pm: Mapping[str, float],
    pp: Mapping[str, float],
) -> dict[Outcome, float]:
    return {
        outcome: value_ratio(float(pm[outcome]), float(pp[outcome]))
        for outcome in OUTCOMES
    }


def log_leverages(
    pm: Mapping[str, float],
    pp: Mapping[str, float],
) -> dict[Outcome, float]:
    return {
        outcome: log_leverage(float(pm[outcome]), float(pp[outcome]))
        for outcome in OUTCOMES
    }
