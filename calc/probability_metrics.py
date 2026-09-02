"""Shared 1X2 probability metrics and log-loss helpers."""

from __future__ import annotations

import math
from typing import Sequence

PROB_EPSILON = 1e-15


def clip_and_normalize_probs(
    p_home: float,
    p_draw: float,
    p_away: float,
    *,
    epsilon: float = PROB_EPSILON,
) -> tuple[float, float, float]:
    """Clip to [eps, 1-eps] and renormalize so probabilities sum to 1."""
    clipped = [
        min(max(value, epsilon), 1.0 - epsilon)
        for value in (p_home, p_draw, p_away)
    ]
    total = sum(clipped)
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return clipped[0] / total, clipped[1] / total, clipped[2] / total


def log_loss_one(
    label: str,
    p_home: float,
    p_draw: float,
    p_away: float,
) -> float:
    """Negative log probability of the observed 1/X/2 label."""
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    if label not in probs:
        raise ValueError(f"Unknown label {label!r}")
    p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(p_home, p_draw, p_away)
    normalized = {"1": p_home_n, "X": p_draw_n, "2": p_away_n}
    return -math.log(normalized[label])


def mean_log_loss(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> float:
    if not labels:
        return 0.0
    if len(labels) != len(probability_rows):
        raise ValueError("labels and probability_rows length mismatch")
    total = 0.0
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        total += log_loss_one(label, p_home, p_draw, p_away)
    return total / len(labels)


def multiclass_log_loss(
    y_true: Sequence[int],
    y_prob: Sequence[Sequence[float]],
) -> float:
    """Mean multiclass log loss with clipped, renormalized 3-way probabilities."""
    if not y_true:
        return 0.0
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")
    total = 0.0
    for label_index, probs in zip(y_true, y_prob):
        if len(probs) != 3:
            raise ValueError("expected 3-class probability vector")
        p_home, p_draw, p_away = clip_and_normalize_probs(
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
        )
        normalized = [p_home, p_draw, p_away]
        probability = normalized[label_index]
        total += -math.log(probability)
    return total / len(y_true)
