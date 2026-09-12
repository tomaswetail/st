"""Tests for the market baseline + residual logit-delta math."""

from __future__ import annotations

import math

import pytest

from src.calc.residual_ml.baseline import (
    apply_large_move_or_identity,
    apply_residual_deltas,
    inv_logit,
    logit,
    market_baseline,
    normalize_probabilities,
    shrink_toward_market,
    target_logit_deltas,
)


def _approx_sum_one(probabilities: dict[str, float]) -> bool:
    return math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9)


def test_normalize_probabilities_renormalizes_to_one():
    normalized = normalize_probabilities({"1": 0.2, "X": 0.2, "2": 0.2})
    assert normalized is not None
    assert _approx_sum_one(normalized)
    assert normalized == pytest.approx({"1": 1 / 3, "X": 1 / 3, "2": 1 / 3})


def test_market_baseline_removes_overround():
    baseline = market_baseline({"1": 0.5, "X": 0.3, "2": 0.4})
    assert baseline is not None
    assert _approx_sum_one(baseline)
    assert baseline["1"] == pytest.approx(0.5 / 1.2)


def test_market_baseline_returns_none_on_missing_key():
    assert market_baseline({"1": 0.5, "X": 0.3}) is None


def test_logit_inv_logit_round_trip():
    for probability in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert inv_logit(logit(probability)) == pytest.approx(probability, abs=1e-9)


def test_target_logit_deltas_shift_baseline_toward_label():
    baseline = {"1": 0.3, "X": 0.3, "2": 0.4}
    deltas = target_logit_deltas("1", baseline, label_smoothing=0.05)
    updated = apply_residual_deltas(baseline, deltas)
    assert updated["1"] > baseline["1"]
    assert _approx_sum_one(updated)


def test_shrink_toward_market_moves_probabilities_toward_market():
    ml_probabilities = {"1": 0.6, "X": 0.2, "2": 0.2}
    market_probabilities = {"1": 0.3, "X": 0.3, "2": 0.4}
    shrunk = shrink_toward_market(ml_probabilities, market_probabilities, alpha=0.5)
    assert shrunk["1"] == pytest.approx(0.45)
    assert shrunk["X"] == pytest.approx(0.25)
    assert shrunk["2"] == pytest.approx(0.30)
    assert _approx_sum_one(shrunk)


def test_shrink_toward_market_zero_alpha_is_identity():
    ml_probabilities = {"1": 0.6, "X": 0.2, "2": 0.2}
    shrunk = shrink_toward_market(
        ml_probabilities,
        {"1": 0.3, "X": 0.3, "2": 0.4},
        alpha=0.0,
    )
    assert shrunk == pytest.approx(ml_probabilities)


def test_threshold_below_returns_same_market_object():
    market = {"1": 0.50, "X": 0.28, "2": 0.22}
    ml_probabilities = {"1": 0.51, "X": 0.275, "2": 0.215}
    result = apply_large_move_or_identity(
        ml_probabilities, market, alpha=0.7, threshold=0.04
    )
    assert result is market
    assert result == {"1": 0.50, "X": 0.28, "2": 0.22}


def test_threshold_at_or_above_applies_shrink():
    market = {"1": 0.50, "X": 0.28, "2": 0.22}
    ml_probabilities = {"1": 0.60, "X": 0.22, "2": 0.18}
    result = apply_large_move_or_identity(
        ml_probabilities, market, alpha=0.7, threshold=0.04
    )
    assert result is not market
    expected = shrink_toward_market(ml_probabilities, market, alpha=0.7)
    assert result["1"] == pytest.approx(expected["1"])
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_residual_deltas_returns_baseline_on_all_zero_deltas():
    baseline = {"1": 0.3, "X": 0.3, "2": 0.4}
    zero_deltas = {"1": 0.0, "X": 0.0, "2": 0.0}
    result = apply_residual_deltas(baseline, zero_deltas)
    assert result == pytest.approx(baseline)
