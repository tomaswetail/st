"""Market baseline + residual logit-delta math for the 1X2 ML model."""

from __future__ import annotations

import math
from typing import Mapping

from src.calc.probability_metrics import PROB_EPSILON
from src.utils.common import OUTCOMES, Outcome, ensure_unit_probabilities

__all__ = [
    "OUTCOMES",
    "apply_residual_deltas",
    "inv_logit",
    "logit",
    "market_baseline",
    "normalize_probabilities",
    "apply_large_move_or_identity",
    "shrink_toward_market",
    "target_logit_deltas",
]


def _clip_prob(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    return min(1.0 - epsilon, max(epsilon, probability))


def logit(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    """Natural logit of a probability, clipped away from 0/1."""
    clipped = _clip_prob(probability, epsilon=epsilon)
    return math.log(clipped / (1.0 - clipped))


def inv_logit(value: float) -> float:
    """Sigmoid: inverse of :func:`logit`."""
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def normalize_probabilities(
    probabilities: Mapping[str, float | None],
) -> dict[str, float] | None:
    """Renormalize {1, X, 2} to sum to 1; return None on missing or non-positive input."""
    present = {
        key: float(value)
        for key, value in probabilities.items()
        if value is not None and key in OUTCOMES
    }
    if len(present) != 3:
        return None
    total = sum(present.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in present.items()}


def market_baseline(
    market_probabilities: Mapping[str, float | None],
) -> dict[str, float] | None:
    """Normalized market baseline from raw 1X2 probabilities (overround-free)."""
    unit = ensure_unit_probabilities(dict(market_probabilities))
    return normalize_probabilities(unit)


def target_logit_deltas(
    label: Outcome,
    baseline: Mapping[str, float],
    *,
    label_smoothing: float = 0.05,
) -> dict[str, float]:
    """Soft one-hot label mapped to per-outcome logit deltas vs baseline."""
    if label not in OUTCOMES:
        raise ValueError(f"Invalid label: {label}")
    smoothing = min(max(label_smoothing, 0.0), 1.0 / 3.0)
    soft_label = {outcome: smoothing for outcome in OUTCOMES}
    soft_label[label] = 1.0 - 2.0 * smoothing
    return {
        outcome: logit(soft_label[outcome]) - logit(baseline[outcome])
        for outcome in OUTCOMES
    }


def apply_residual_deltas(
    baseline: Mapping[str, float],
    deltas: Mapping[str, float],
) -> dict[str, float]:
    """Apply per-outcome logit adjustments and renormalize to sum to 1."""
    adjusted = {
        outcome: inv_logit(logit(baseline[outcome]) + float(deltas[outcome]))
        for outcome in OUTCOMES
    }
    normalized = normalize_probabilities(adjusted)
    if normalized is None:
        return dict(baseline)
    return normalized


def shrink_toward_market(
    ml_probabilities: Mapping[str, float],
    market_probabilities: Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """Mix ML toward market: final = (1 - alpha) * ml + alpha * market."""
    alpha = min(1.0, max(0.0, alpha))
    mixed = {
        outcome: (1.0 - alpha) * ml_probabilities[outcome]
        + alpha * market_probabilities[outcome]
        for outcome in OUTCOMES
    }
    normalized = normalize_probabilities(mixed)
    if normalized is None:
        return dict(ml_probabilities)
    return normalized


def apply_large_move_or_identity(
    ml_probabilities: Mapping[str, float],
    market_probabilities: Mapping[str, float],
    *,
    alpha: float,
    threshold: float,
) -> dict[str, float]:
    """Identity market if the ML move is below ``threshold``; else shrink.

    ``large = max_k |p_k^{ml} - p_k^{market}|``. Below threshold returns the
    same market object (not a no-op shrink). Above threshold applies
    ``shrink_toward_market``. Outputs are normalized over ``{1, X, 2}``.
    """
    large = max(
        abs(ml_probabilities[outcome] - market_probabilities[outcome])
        for outcome in OUTCOMES
    )
    if large < threshold:
        return market_probabilities  # type: ignore[return-value]
    return shrink_toward_market(
        ml_probabilities, market_probabilities, alpha=alpha
    )
