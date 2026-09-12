"""Strict market odds → fair probabilities (Pm).

Shares ``src.utils.common.validate_decimal_odds`` / ``odds_to_probabilities``
with production residual-ML scoring. This module raises ``InvalidOddsError``
so optimizer callers keep a dedicated exception type.
"""

from __future__ import annotations

from typing import Mapping

from src.utils.common import OUTCOMES, Outcome, odds_to_probabilities


class InvalidOddsError(ValueError):
    """Raised when odds cannot form a valid fair-probability triple."""


def fair_probabilities_from_odds(
    odds_1: float,
    odds_x: float,
    odds_2: float,
) -> dict[Outcome, float]:
    """Convert decimal 1X2 odds to de-vigged fair probs: raw=1/odds; p=raw/sum."""
    try:
        return odds_to_probabilities(odds_1, odds_x, odds_2)
    except ValueError as exc:
        if isinstance(exc, InvalidOddsError):
            raise
        raise InvalidOddsError(str(exc)) from exc


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
