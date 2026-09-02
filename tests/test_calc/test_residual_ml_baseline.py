"""Tests for residual ML baseline blending."""

from __future__ import annotations

import pytest

from calc.residual_ml.baseline import (
    apply_residual_deltas,
    blend_baselines,
    engine_baseline,
    market_baseline,
    shrink_toward_market,
    target_logit_deltas,
)


def test_market_baseline_sums_to_one():
    probs = market_baseline({"1": 0.50, "X": 0.28, "2": 0.22})
    assert probs is not None
    assert sum(probs.values()) == pytest.approx(1.0)


def test_blend_baselines_sums_to_one():
    market = {"1": 0.55, "X": 0.25, "2": 0.20}
    engine = {"1": 0.45, "X": 0.30, "2": 0.25}
    blend = blend_baselines(market, engine, market_weight=0.5, dc_weight=0.5)
    assert blend is not None
    assert sum(blend.values()) == pytest.approx(1.0)


def test_blend_baselines_linear_70_30():
    market = {"1": 0.60, "X": 0.25, "2": 0.15}
    engine = {"1": 0.40, "X": 0.30, "2": 0.30}
    blend = blend_baselines(market, engine, market_weight=0.7, dc_weight=0.3)
    assert blend is not None
    assert blend["1"] == pytest.approx(0.54)
    assert blend["X"] == pytest.approx(0.265)
    assert blend["2"] == pytest.approx(0.195)
    assert sum(blend.values()) == pytest.approx(1.0)


def test_target_logit_deltas_round_trip():
    baseline = {"1": 0.45, "X": 0.28, "2": 0.27}
    deltas = target_logit_deltas("1", baseline)
    adjusted = apply_residual_deltas(baseline, deltas)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["1"] > baseline["1"]


def test_blend_falls_back_to_market_when_engine_missing():
    market = {"1": 0.55, "X": 0.25, "2": 0.20}
    blend = blend_baselines(market, None)
    assert blend == market


def test_apply_residual_deltas_renormalizes():
    baseline = {"1": 0.50, "X": 0.28, "2": 0.22}
    adjusted = apply_residual_deltas(
        baseline,
        {"1": 0.2, "X": -0.1, "2": -0.1},
    )
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["1"] > baseline["1"]


def test_engine_baseline_requires_all_three():
    assert engine_baseline(p_home_dc=0.5, p_draw_dc=0.3, p_away_dc=None) is None


def test_shrink_toward_market():
    ml = {"1": 0.60, "X": 0.20, "2": 0.20}
    market = {"1": 0.50, "X": 0.28, "2": 0.22}
    shrunk = shrink_toward_market(ml, market, alpha=0.5)
    assert shrunk["1"] == pytest.approx(0.55)
    toward_market = shrink_toward_market(ml, market, alpha=1.0)
    assert toward_market["1"] == pytest.approx(0.50)
    no_shrink = shrink_toward_market(ml, market, alpha=0.0)
    assert no_shrink["1"] == pytest.approx(0.60)


def test_apply_market_only_baseline():
    from calc.residual_ml.baseline import apply_market_only_baseline, is_market_only_weights

    rows = [
        {
            "p_home_market_norm": 0.5,
            "p_draw_market_norm": 0.3,
            "p_away_market_norm": 0.2,
            "p_home_blend": 0.4,
            "p_draw_blend": 0.3,
            "p_away_blend": 0.3,
        },
        {
            "p_home_market_norm": None,
            "p_draw_market_norm": 0.3,
            "p_away_market_norm": 0.2,
            "p_home_blend": 0.1,
            "p_draw_blend": 0.1,
            "p_away_blend": 0.8,
        },
    ]
    apply_market_only_baseline(rows)
    assert rows[0]["p_home_blend"] == pytest.approx(0.5)
    assert rows[0]["p_draw_blend"] == pytest.approx(0.3)
    assert rows[0]["p_away_blend"] == pytest.approx(0.2)
    assert rows[1]["p_home_blend"] == pytest.approx(0.1)
    assert is_market_only_weights(1.0, 0.0)
    assert not is_market_only_weights(0.7, 0.3)
