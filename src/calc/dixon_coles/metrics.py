"""Probability metrics helpers for classic Dixon–Coles evaluation."""

from __future__ import annotations

from typing import Sequence

from calc.probability_metrics import (
    PROB_EPSILON,
    clip_and_normalize_probs,
    log_loss_one,
    mean_log_loss,
    multiclass_log_loss,
)

__all__ = [
    "PROB_EPSILON",
    "clip_and_normalize_probs",
    "log_loss_one",
    "mean_log_loss",
    "multiclass_log_loss",
    "ranked_probability_score",
    "mean_rps",
]


def ranked_probability_score(
    label: str,
    p_home: float,
    p_draw: float,
    p_away: float,
) -> float:
    """RPS for ordered outcomes Home -> Draw -> Away."""
    p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(p_home, p_draw, p_away)
    observed = {"1": (1.0, 0.0, 0.0), "X": (0.0, 1.0, 0.0), "2": (0.0, 0.0, 1.0)}
    if label not in observed:
        raise ValueError(f"Unknown label {label!r}")
    o_home, o_draw, _o_away = observed[label]
    forecast_cdf = (p_home_n, p_home_n + p_draw_n)
    observed_cdf = (o_home, o_home + o_draw)
    return 0.5 * (
        (forecast_cdf[0] - observed_cdf[0]) ** 2
        + (forecast_cdf[1] - observed_cdf[1]) ** 2
    )


def mean_rps(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> float:
    if not labels:
        return 0.0
    if len(labels) != len(probability_rows):
        raise ValueError("labels and probability_rows length mismatch")
    total = 0.0
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        total += ranked_probability_score(label, p_home, p_draw, p_away)
    return total / len(labels)
