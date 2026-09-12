"""Tests for shared probability metrics."""

from __future__ import annotations

import math

import pytest

from src.calc.probability_metrics import (
    binary_log_loss,
    mean_accuracy,
    mean_log_loss,
    mean_rps,
    multiclass_brier,
    multiclass_log_loss,
    predicted_label,
    top_label_ece,
)


def test_multiclass_log_loss_matches_sklearn_on_fixed_rows():
    pytest.importorskip("sklearn")
    from sklearn.metrics import log_loss

    y_true = [0, 1, 2, 0]
    y_prob = [
        [0.6, 0.25, 0.15],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
        [0.45, 0.30, 0.25],
    ]
    expected = log_loss(y_true, y_prob)
    actual = multiclass_log_loss(y_true, y_prob)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_hand_computed_brier_rps_logloss_accuracy():
    probabilities = (0.5, 0.3, 0.2)
    assert multiclass_brier(["1"], [probabilities]) == pytest.approx(0.38)
    assert mean_rps(["1"], [probabilities]) == pytest.approx(0.145)
    assert mean_rps(["2"], [probabilities]) == pytest.approx(0.445)
    assert mean_log_loss(["1"], [probabilities]) == pytest.approx(-math.log(0.5))
    assert predicted_label(0.4, 0.4, 0.2) == "1"
    assert mean_accuracy(["1"], [(0.4, 0.4, 0.2)]) == 1.0


def test_top_label_ece_perfectly_calibrated_constant_predictions():
    probabilities = (0.7, 0.2, 0.1)
    labels = ["1"] * 70 + ["X"] * 20 + ["2"] * 10
    probability_rows = [probabilities] * len(labels)
    assert top_label_ece(labels, probability_rows) == pytest.approx(0.0, abs=1e-12)


def test_binary_log_loss_empty_returns_none():
    assert binary_log_loss([], []) is None
