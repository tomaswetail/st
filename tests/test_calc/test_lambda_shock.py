"""Tests for T2 Dixon–Coles λ shock."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.calc.residual_ml.lambda_shock import (
    dixon_coles_1x2,
    rho_for_league,
    shock_expected_goals,
)


def test_lambda_shock_formula_and_uncovered_nulls():
    shocked = shock_expected_goals(
        1.4,
        1.1,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=1,
        k=0.02,
    )
    assert shocked[0] == pytest.approx(1.4 * 0.96 * 1.08)
    assert shocked[1] == pytest.approx(1.1 * 0.92 * 1.04)
    assert shock_expected_goals(
        1.4,
        1.1,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=0,
    ) == (1.4, 1.1)
    assert shock_expected_goals(
        1.4,
        1.1,
        home_unavailable_count=None,
        away_unavailable_count=4,
        has_availability=1,
    ) == (1.4, 1.1)
    assert shock_expected_goals(
        1.4,
        1.1,
        home_unavailable_count=2,
        away_unavailable_count="",
        has_availability=None,
    ) == (1.4, 1.1)


def test_lambda_shock_clips_each_multiplier():
    shocked = shock_expected_goals(
        1.0,
        1.0,
        home_unavailable_count=20,
        away_unavailable_count=20,
        has_availability=1,
        k=0.02,
    )
    assert shocked[0] == pytest.approx(0.70 * 1.30)
    assert shocked[1] == pytest.approx(0.70 * 1.30)


def test_paired_recompute_uses_same_rho_and_sums_to_one():
    rho = -0.13
    unadjusted = dixon_coles_1x2(1.4, 1.1, rho)
    lambda_home, lambda_away = shock_expected_goals(
        1.4,
        1.1,
        home_unavailable_count=2,
        away_unavailable_count=4,
        has_availability=1,
    )
    adjusted = dixon_coles_1x2(lambda_home, lambda_away, rho)
    assert sum(unadjusted.values()) == pytest.approx(1.0)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert rho_for_league(
        39,
        default_rho=-0.13,
        league_params={39: SimpleNamespace(rho=-0.0145)},
    ) == pytest.approx(-0.0145)
    assert rho_for_league(99, default_rho=-0.13, league_params={}) == pytest.approx(
        -0.13
    )
    assert rho_for_league(None, default_rho=-0.13, league_params={}) == pytest.approx(
        -0.13
    )
