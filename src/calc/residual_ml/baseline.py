"""Market + Dixon–Coles baseline blend and residual probability math."""

from __future__ import annotations

import math
from typing import Mapping

from calc.draw_adjustment import (
    DrawAdjustmentConfig,
    apply_draw_adjustment,
    load_draw_adjustment_config,
)
from utils.common import OUTCOMES, Outcome, ensure_unit_probabilities
from calc.probability_metrics import PROB_EPSILON

__all__ = [
    "DrawAdjustmentConfig",
    "apply_draw_adjustment",
    "apply_market_only_baseline",
    "apply_residual_deltas",
    "blend_baselines",
    "blend_logit_vector",
    "engine_baseline",
    "inv_logit",
    "is_market_only_weights",
    "load_draw_adjustment_config",
    "logit",
    "market_baseline",
    "normalize_probabilities",
    "shrink_toward_market",
    "target_logit_deltas",
]


def _clip_prob(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    return min(1.0 - epsilon, max(epsilon, probability))


def logit(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    clipped = _clip_prob(probability, epsilon=epsilon)
    return math.log(clipped / (1.0 - clipped))


def inv_logit(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def normalize_probabilities(
    probabilities: Mapping[str, float | None],
) -> dict[str, float] | None:
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
    unit = ensure_unit_probabilities(dict(market_probabilities))
    return normalize_probabilities(unit)


def engine_baseline(
    *,
    p_home_dc: float | None,
    p_draw_dc: float | None,
    p_away_dc: float | None,
) -> dict[str, float] | None:
    return normalize_probabilities(
        {"1": p_home_dc, "X": p_draw_dc, "2": p_away_dc}
    )


def blend_baselines(
    market: Mapping[str, float] | None,
    engine: Mapping[str, float] | None,
    *,
    market_weight: float = 0.7,
    dc_weight: float | None = None,
) -> dict[str, float] | None:
    """Weighted linear blend in probability space; falls back to whichever baseline exists."""
    if market is None and engine is None:
        return None
    if market is None:
        return dict(engine)  # type: ignore[arg-type]
    if engine is None:
        return dict(market)

    weight_market = market_weight
    weight_dc = dc_weight if dc_weight is not None else (1.0 - market_weight)
    total_weight = weight_market + weight_dc
    if total_weight <= 0:
        return None
    weight_market /= total_weight
    weight_dc /= total_weight

    blended: dict[str, float] = {}
    for outcome in OUTCOMES:
        blended[outcome] = (
            weight_market * market[outcome] + weight_dc * engine[outcome]
        )
    return normalize_probabilities(blended)


def apply_market_only_baseline(rows: list[dict]) -> list[dict]:
    """Set p_*_blend from p_*_market_norm so residuals train/predict vs market only."""
    for row in rows:
        home = row.get("p_home_market_norm")
        draw = row.get("p_draw_market_norm")
        away = row.get("p_away_market_norm")
        if home is None or draw is None or away is None:
            continue
        if home == "" or draw == "" or away == "":
            continue
        row["p_home_blend"] = float(home)
        row["p_draw_blend"] = float(draw)
        row["p_away_blend"] = float(away)
    return rows


def is_market_only_weights(market_weight: float, dc_weight: float) -> bool:
    return market_weight >= 1.0 - 1e-9 and dc_weight <= 1e-9


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


def blend_logit_vector(baseline: Mapping[str, float]) -> dict[str, float]:
    return {outcome: logit(baseline[outcome]) for outcome in OUTCOMES}


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
