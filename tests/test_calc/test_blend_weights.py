"""Tests for conditional blend weight selection."""

from __future__ import annotations

import pytest

from src.calc.residual_ml.blend_weights import (
    BlendWeightRule,
    BlendWeightsConfig,
    select_blend_weights,
)


def _config(**overrides) -> BlendWeightsConfig:
    base = BlendWeightsConfig(
        enabled=True,
        default_market_weight=0.7,
        default_dc_weight=0.3,
        league_overrides={},
        market_vs_dc=BlendWeightRule(
            enabled=True,
            threshold=0.15,
            market_weight=0.85,
            dc_weight=0.15,
        ),
        dc_quality=BlendWeightRule(
            enabled=True,
            missing_league_market_weight=0.9,
            missing_league_dc_weight=0.1,
            max_acceptable_log_loss=1.05,
            poor_fit_market_weight=0.88,
            poor_fit_dc_weight=0.12,
        ),
        injury_uncertainty=BlendWeightRule(
            enabled=True,
            missing_availability_market_weight=0.8,
            missing_availability_dc_weight=0.2,
        ),
    )
    return BlendWeightsConfig(**{**base.__dict__, **overrides})


def test_disabled_uses_fallback_defaults():
    config = _config(enabled=False)
    market, dc = select_blend_weights(
        {},
        config,
        fallback_market_weight=0.7,
        fallback_dc_weight=0.3,
    )
    assert market == pytest.approx(0.7)
    assert dc == pytest.approx(0.3)


def test_missing_league_and_injury_keeps_defaults():
    """Absent league_external_id / has_availability → safe 0.7/0.3."""
    config = _config()
    market, dc = select_blend_weights(
        {"market_vs_dc_home": 0.01},
        config,
        dc_league_log_loss={"39": 0.95},
    )
    assert market == pytest.approx(0.7)
    assert dc == pytest.approx(0.3)


def test_market_vs_dc_rule_fires():
    config = _config()
    market, dc = select_blend_weights(
        {"market_vs_dc_draw": 0.25},
        config,
        dc_league_log_loss={},
    )
    assert market == pytest.approx(0.85)
    assert dc == pytest.approx(0.15)


def test_league_override_and_missing_dc_quality():
    config = _config(
        league_overrides={"180": (0.95, 0.05)},
    )
    market, dc = select_blend_weights(
        {},
        config,
        league_external_id="180",
        dc_league_log_loss={"39": 0.9},  # 180 missing → also DC quality bump
    )
    # Highest market among override (0.95) and missing-league (0.9)
    assert market == pytest.approx(0.95)
    assert dc == pytest.approx(0.05)


def test_explicit_zero_availability_bumps_market():
    config = _config()
    market, dc = select_blend_weights(
        {"has_availability": 0},
        config,
        dc_league_log_loss={},
    )
    assert market == pytest.approx(0.8)
    assert dc == pytest.approx(0.2)


def test_league_override_applies_below_market_only_default():
    """Allowlist 0.7/0.3 must win over default 1.0/0.0 when other rules are off."""
    config = BlendWeightsConfig(
        enabled=True,
        default_market_weight=1.0,
        default_dc_weight=0.0,
        league_overrides={"39": (0.7, 0.3)},
        market_vs_dc=BlendWeightRule(enabled=False),
        dc_quality=BlendWeightRule(enabled=False),
        injury_uncertainty=BlendWeightRule(enabled=False),
    )
    allowlisted = select_blend_weights(
        {},
        config,
        league_external_id="39",
    )
    other_league = select_blend_weights(
        {},
        config,
        league_external_id="41",
    )
    missing_league = select_blend_weights({}, config)
    assert allowlisted == pytest.approx((0.7, 0.3))
    assert other_league == pytest.approx((1.0, 0.0))
    assert missing_league == pytest.approx((1.0, 0.0))
