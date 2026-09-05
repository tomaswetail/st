"""Tests for explicit draw logit adjustment."""

from __future__ import annotations

import pytest

from src.calc.draw_adjustment import (
    DrawAdjustmentConfig,
    apply_draw_adjustment,
)
from src.calc.residual_ml.baseline import apply_draw_adjustment as baseline_apply


def test_enabled_adjust_changes_draw_and_renormalizes():
    blend = {"1": 0.45, "X": 0.25, "2": 0.30}
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=0.5,
        features={"away_npxg_for": -0.2},
    )
    features = {"away_npxg_for": 1.0}
    adjusted = apply_draw_adjustment(blend, features, config)
    assert adjusted is not None
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["X"] != pytest.approx(blend["X"])
    # Positive intercept with mild negative feature still moves mass; check simplex.
    assert all(value > 0.0 for value in adjusted.values())


def test_disabled_config_is_noop():
    blend = {"1": 0.50, "X": 0.28, "2": 0.22}
    config = DrawAdjustmentConfig(
        enabled=False,
        intercept=10.0,
        features={"away_npxg_for": 5.0},
    )
    adjusted = apply_draw_adjustment(blend, {"away_npxg_for": 1.0}, config)
    assert adjusted == blend


def test_missing_feature_uses_safe_default():
    blend = {"1": 0.40, "X": 0.30, "2": 0.30}
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=0.0,
        features={"away_npxg_for": 0.5},
        missing_feature_default=0.0,
    )
    # Missing key → 0.0 contribution → identical to intercept-only (0).
    adjusted = apply_draw_adjustment(blend, {}, config)
    assert adjusted is not None
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["X"] == pytest.approx(blend["X"])


def test_extreme_near_zero_one_probs_still_unit_simplex():
    blend = {"1": 1e-12, "X": 1.0 - 2e-12, "2": 1e-12}
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=5.0,
        features={"congestion_difference": 2.0},
    )
    adjusted = apply_draw_adjustment(
        blend,
        {"congestion_difference": 10.0},
        config,
    )
    assert adjusted is not None
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert all(0.0 < value < 1.0 for value in adjusted.values())


def test_baseline_module_exports_apply_draw_adjustment():
    blend = {"1": 0.5, "X": 0.25, "2": 0.25}
    config = DrawAdjustmentConfig(enabled=False)
    assert baseline_apply(blend, {}, config) == blend
