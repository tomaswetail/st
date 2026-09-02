"""Tests for shared probability metrics."""

from __future__ import annotations

import pytest

from calc.probability_metrics import multiclass_log_loss


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
