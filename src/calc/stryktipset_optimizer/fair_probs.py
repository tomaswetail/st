"""Strict market odds → fair probabilities (Pm).

Do NOT use or modify ``src.utils.common.odds_to_probabilities`` — that path
serves production residual-ML scoring. This helper rejects odds <= 1,
null/NaN/inf, and non-finite values.
"""

from __future__ import annotations

import math
from typing import Mapping

from src.utils.common import OUTCOMES, Outcome


class InvalidOddsError(ValueError):
    """Raised when odds cannot form a valid fair-probability triple."""


def _validate_odds_value(label: str, odds: float | None) -> float:
    if odds is None:
        raise InvalidOddsError(f"{label} odds is null")
    try:
        value = float(odds)
    except (TypeError, ValueError) as exc:
        raise InvalidOddsError(f"{label} odds is not numeric: {odds!r}") from exc
    if math.isnan(value) or math.isinf(value):
        raise InvalidOddsError(f"{label} odds is not finite: {odds!r}")
    if value <= 1.0:
        raise InvalidOddsError(f"{label} odds must be > 1, got {value}")
    return value


def fair_probabilities_from_odds(
    odds_1: float,
    odds_x: float,
    odds_2: float,
) -> dict[Outcome, float]:
    """Convert decimal 1X2 odds to de-vigged fair probs: raw=1/odds; p=raw/sum."""
    home = _validate_odds_value("1", odds_1)
    draw = _validate_odds_value("X", odds_x)
    away = _validate_odds_value("2", odds_2)

    raw_home = 1.0 / home
    raw_draw = 1.0 / draw
    raw_away = 1.0 / away
    total = raw_home + raw_draw + raw_away
    if total <= 0 or not math.isfinite(total):
        raise InvalidOddsError(f"odds implied mass is invalid: {total}")

    return {
        "1": raw_home / total,
        "X": raw_draw / total,
        "2": raw_away / total,
    }


def fair_probabilities_from_mapping(
    odds: Mapping[str, float],
) -> dict[Outcome, float]:
    """Same as fair_probabilities_from_odds from an odds mapping."""
    try:
        return fair_probabilities_from_odds(odds["1"], odds["X"], odds["2"])
    except KeyError as exc:
        raise InvalidOddsError(f"odds mapping missing outcome {exc}") from exc


def outcome_vector(probs: Mapping[str, float]) -> tuple[float, float, float]:
    """Return (p1, pX, p2) in OUTCOMES order."""
    return tuple(float(probs[outcome]) for outcome in OUTCOMES)  # type: ignore[return-value]
