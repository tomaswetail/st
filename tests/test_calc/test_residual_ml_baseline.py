"""Tests for residual ML baseline blending."""

from __future__ import annotations

import pytest

from src.calc.residual_ml.baseline import (
    apply_count_differential_logit_shift,
    apply_residual_deltas,
    blend_baselines,
    coverage_aware_shrink_alpha,
    engine_baseline,
    inv_logit,
    logit,
    market_baseline,
    shrink_toward_market,
    shrink_toward_market_by_coverage,
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


def test_coverage_aware_shrink_alpha_uses_covered_and_uncovered():
    assert coverage_aware_shrink_alpha(1) == 0.7
    assert coverage_aware_shrink_alpha(1.0) == 0.7
    assert coverage_aware_shrink_alpha("1") == 0.7
    assert coverage_aware_shrink_alpha(0) == 0.85
    assert coverage_aware_shrink_alpha(None) == 0.85
    assert coverage_aware_shrink_alpha("") == 0.85
    assert coverage_aware_shrink_alpha("x") == 0.85


def test_coverage_aware_shrink_renormalizes_and_leaves_global_shrink():
    ml = {"1": 0.60, "X": 0.20, "2": 0.20}
    market = {"1": 0.50, "X": 0.28, "2": 0.22}
    covered = shrink_toward_market_by_coverage(ml, market, 1)
    uncovered = shrink_toward_market_by_coverage(ml, market, 0)
    missing = shrink_toward_market_by_coverage(ml, market, None)
    global_ship = shrink_toward_market(ml, market, alpha=0.7)
    assert sum(covered.values()) == pytest.approx(1.0)
    assert sum(uncovered.values()) == pytest.approx(1.0)
    assert covered["1"] == pytest.approx(global_ship["1"])
    assert uncovered["1"] == pytest.approx(
        shrink_toward_market(ml, market, alpha=0.85)["1"]
    )
    assert missing["1"] == pytest.approx(uncovered["1"])
    assert shrink_toward_market(ml, market, alpha=0.7)["1"] == pytest.approx(
        0.60 * 0.3 + 0.50 * 0.7
    )


def test_count_differential_logit_shift_moves_home_away_keeps_draw_logit():
    engine = {"1": 0.45, "X": 0.30, "2": 0.25}
    # c_a=4, c_h=2 → Δ = 0.04 * 2 = 0.08
    shifted = apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=1,
        k=0.04,
    )
    assert sum(shifted.values()) == pytest.approx(1.0)
    assert shifted["1"] == pytest.approx(
        apply_residual_deltas(engine, {"1": 0.08, "X": 0.0, "2": -0.08})["1"]
    )
    assert shifted["1"] > engine["1"]
    assert shifted["2"] < engine["2"]
    pre_renorm_draw = inv_logit(logit(engine["X"]) + 0.0)
    assert pre_renorm_draw == pytest.approx(engine["X"])


def test_count_differential_logit_shift_leaves_uncovered_and_nulls():
    engine = {"1": 0.45, "X": 0.30, "2": 0.25}
    assert apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=0,
    ) == engine
    assert apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=None,
    ) == engine
    assert apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=None,
        away_unavailable_count=4,
        has_availability=1,
    ) == engine
    assert apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=2,
        away_unavailable_count="",
        has_availability=1,
    ) == engine


def test_apply_market_only_baseline():
    from src.calc.residual_ml.baseline import apply_market_only_baseline, is_market_only_weights

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
