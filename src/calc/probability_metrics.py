"""Shared 1X2 probability metrics and log-loss helpers."""

from __future__ import annotations

import math
from typing import Sequence

from src.utils.common import OUTCOMES

PROB_EPSILON = 1e-15
DEFAULT_ECE_BINS = 10
_LABEL_INDEX = {"1": 0, "X": 1, "2": 2}


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


def clip_unit(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    """Clip a single probability to ``[epsilon, 1 - epsilon]``."""
    return min(1.0 - epsilon, max(epsilon, probability))


def binary_log_loss(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    epsilon: float = PROB_EPSILON,
) -> float | None:
    """Mean one-vs-rest binary log loss.

    ``-(1/N) Σ [y log p + (1-y) log(1-p)]`` after clipping each ``p`` to
    ``[epsilon, 1 - epsilon]``. Returns ``None`` when ``y_true`` is empty.
    """
    if not y_true:
        return None
    total = 0.0
    for label, probability in zip(y_true, probabilities):
        clipped = clip_unit(float(probability), epsilon=epsilon)
        if label:
            total += -math.log(clipped)
        else:
            total += -math.log(1.0 - clipped)
    return total / len(y_true)


def predicted_label(p_home: float, p_draw: float, p_away: float) -> str:
    """Argmax over ``{1, X, 2}`` after clip/normalize.

    Ties break by ``OUTCOMES`` order: start at ``1``, replace only on a
    strictly greater probability (same as ``market_favorite``).
    """
    p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(p_home, p_draw, p_away)
    return _argmax_outcome(p_home_n, p_draw_n, p_away_n)


def _argmax_outcome(p_home: float, p_draw: float, p_away: float) -> str:
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    best_outcome = OUTCOMES[0]
    best_prob = probs[best_outcome]
    for outcome in OUTCOMES[1:]:
        if probs[outcome] > best_prob:
            best_outcome = outcome
            best_prob = probs[outcome]
    return best_outcome


def _one_hot(label: str) -> tuple[float, float, float]:
    if label not in _LABEL_INDEX:
        raise ValueError(f"Unknown label {label!r}")
    index = _LABEL_INDEX[label]
    return (float(index == 0), float(index == 1), float(index == 2))


def _paired_rows(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> None:
    if len(labels) != len(probability_rows):
        raise ValueError("labels and probability_rows length mismatch")


def multiclass_brier(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> float:
    """Mean multiclass Brier: ``(1/N) Σ_i Σ_k (p_ik - y_ik)^2`` for ``{1, X, 2}``."""
    if not labels:
        return 0.0
    _paired_rows(labels, probability_rows)
    total = 0.0
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(
            p_home, p_draw, p_away
        )
        y_home, y_draw, y_away = _one_hot(label)
        total += (
            (p_home_n - y_home) ** 2
            + (p_draw_n - y_draw) ** 2
            + (p_away_n - y_away) ** 2
        )
    return total / len(labels)


def mean_rps(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> float:
    """Mean ranked probability score for ordered outcomes ``1``, ``X``, ``2``.

    ``RPS = 1/(K-1) Σ_{k=1}^{K-1} (F_k - O_k)^2`` with ``K=3``,
    ``F_k = Σ_{j≤k} p_j``, ``O_k = Σ_{j≤k} y_j``.
    """
    if not labels:
        return 0.0
    _paired_rows(labels, probability_rows)
    total = 0.0
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(
            p_home, p_draw, p_away
        )
        y_home, y_draw, _y_away = _one_hot(label)
        forecast_1 = p_home_n
        forecast_x = p_home_n + p_draw_n
        observed_1 = y_home
        observed_x = y_home + y_draw
        total += 0.5 * (
            (forecast_1 - observed_1) ** 2 + (forecast_x - observed_x) ** 2
        )
    return total / len(labels)


def mean_accuracy(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> float:
    """Fraction of matches where clip/normalized argmax equals the label."""
    if not labels:
        return 0.0
    _paired_rows(labels, probability_rows)
    correct = 0
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        if predicted_label(p_home, p_draw, p_away) == label:
            correct += 1
    return correct / len(labels)


def expected_calibration_error(
    confidences: Sequence[float],
    observations: Sequence[float],
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Reliability-diagram ECE with equal-width bins on ``[0, 1]``.

    Default is **10** bins. The last bin includes ``1.0``. Empty bins are
    ignored.

    ``ECE = Σ_b (n_b / N) |acc_b - conf_b|``

    Two definitions use this same formula:

    - **Top-label ECE** (``top_label_ece``): predicted class is argmax with
      ``OUTCOMES`` tie-break; confidence is that class's probability;
      observation is 1 if the predicted class equals the label.
    - **Per-class ECE** (``per_class_ece``): confidence is that class's
      probability; observation is the one-hot of that class.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not confidences:
        raise ValueError("confidences is empty")
    if len(confidences) != len(observations):
        raise ValueError("confidences and observations length mismatch")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for confidence, observation in zip(confidences, observations):
        bin_index = min(int(float(confidence) * n_bins), n_bins - 1)
        if bin_index < 0:
            bin_index = 0
        buckets[bin_index].append((float(confidence), float(observation)))
    row_count = len(confidences)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        bin_count = len(bucket)
        accuracy = sum(observed for _conf, observed in bucket) / bin_count
        mean_confidence = sum(confidence for confidence, _obs in bucket) / bin_count
        ece += (bin_count / row_count) * abs(accuracy - mean_confidence)
    return ece


def top_label_ece(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Top-label ECE: predicted-class confidence vs correctness."""
    if not labels:
        return 0.0
    _paired_rows(labels, probability_rows)
    confidences: list[float] = []
    observations: list[float] = []
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(
            p_home, p_draw, p_away
        )
        predicted = _argmax_outcome(p_home_n, p_draw_n, p_away_n)
        predicted_probability = {"1": p_home_n, "X": p_draw_n, "2": p_away_n}[
            predicted
        ]
        confidences.append(predicted_probability)
        observations.append(1.0 if predicted == label else 0.0)
    return expected_calibration_error(confidences, observations, n_bins=n_bins)


def per_class_ece(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
    class_label: str,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Per-class ECE: class probability vs one-hot of that class."""
    if class_label not in _LABEL_INDEX:
        raise ValueError(f"Unknown label {class_label!r}")
    if not labels:
        return 0.0
    _paired_rows(labels, probability_rows)
    class_index = _LABEL_INDEX[class_label]
    confidences: list[float] = []
    observations: list[float] = []
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        normalized = clip_and_normalize_probs(p_home, p_draw, p_away)
        confidences.append(normalized[class_index])
        observations.append(1.0 if label == class_label else 0.0)
    return expected_calibration_error(confidences, observations, n_bins=n_bins)


def per_class_log_loss(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> tuple[float | None, float | None, float | None]:
    """One-vs-rest binary log loss for home, draw, and away."""
    if not labels:
        return None, None, None
    _paired_rows(labels, probability_rows)
    y_home: list[int] = []
    y_draw: list[int] = []
    y_away: list[int] = []
    p_home_values: list[float] = []
    p_draw_values: list[float] = []
    p_away_values: list[float] = []
    for label, (p_home, p_draw, p_away) in zip(labels, probability_rows):
        p_home_n, p_draw_n, p_away_n = clip_and_normalize_probs(
            p_home, p_draw, p_away
        )
        y_home.append(1 if label == "1" else 0)
        y_draw.append(1 if label == "X" else 0)
        y_away.append(1 if label == "2" else 0)
        p_home_values.append(p_home_n)
        p_draw_values.append(p_draw_n)
        p_away_values.append(p_away_n)
    return (
        binary_log_loss(y_home, p_home_values),
        binary_log_loss(y_draw, p_draw_values),
        binary_log_loss(y_away, p_away_values),
    )
