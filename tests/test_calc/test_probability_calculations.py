"""Worked-example tests for 1X2 probability formulas.

Numbers here must stay in sync with docs/probability_calculations.md.
"""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.calc.dixon_coles.metrics import ranked_probability_score
from src.calc.dixon_coles.model import match_weight
from src.calc.draw_adjustment import DrawAdjustmentConfig, apply_draw_adjustment
from src.calc.probability_manager import ProbabilityManager
from src.calc.probability_metrics import log_loss_one, mean_log_loss, multiclass_log_loss
from src.calc.residual_ml.baseline import (
    apply_residual_deltas,
    blend_baselines,
    inv_logit,
    logit,
    market_baseline,
    shrink_toward_market,
    target_logit_deltas,
)
from src.calc.residual_ml.blend_weights import (
    BlendWeightRule,
    BlendWeightsConfig,
    market_vs_dc_magnitude,
    select_blend_weights,
)
from src.calc.strength_calculator import (
    _dixon_coles_tau,
    _poisson_pmf,
    _scoreline_probability,
    dixon_coles_matrix,
)
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures
from src.utils.common import odds_to_probabilities


def test_odds_to_probabilities_example():
    probs = odds_to_probabilities(2.00, 3.50, 4.00)
    assert probs["1"] == pytest.approx(0.48275862069)
    assert probs["X"] == pytest.approx(0.275862068966)
    assert probs["2"] == pytest.approx(0.241379310345)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_market_baseline_renormalizes_overround_free_vector():
    raw = {"1": 0.50, "X": 0.28, "2": 0.22}
    assert sum(raw.values()) == pytest.approx(1.0)
    assert market_baseline(raw) == pytest.approx(raw)


def test_blend_70_30_worked_example():
    market = {"1": 0.60, "X": 0.25, "2": 0.15}
    engine = {"1": 0.40, "X": 0.30, "2": 0.30}
    blend = blend_baselines(market, engine, market_weight=0.7, dc_weight=0.3)
    assert blend == {
        "1": pytest.approx(0.54),
        "X": pytest.approx(0.265),
        "2": pytest.approx(0.195),
    }


def test_logit_inv_logit_round_trip():
    assert logit(0.25) == pytest.approx(math.log(0.25 / 0.75))
    assert inv_logit(logit(0.25)) == pytest.approx(0.25)
    assert logit(0.5) == pytest.approx(0.0)
    assert inv_logit(0.0) == pytest.approx(0.5)


def test_draw_adjustment_worked_example():
    blend = {"1": 0.45, "X": 0.25, "2": 0.30}
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=0.5,
        features={"away_npxg_for": -0.2},
    )
    adjusted = apply_draw_adjustment(blend, {"away_npxg_for": 1.0}, config)
    assert adjusted is not None
    assert adjusted["1"] == pytest.approx(0.424399203647856)
    assert adjusted["X"] == pytest.approx(0.29266799392024)
    assert adjusted["2"] == pytest.approx(0.282932802431904)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["X"] > blend["X"]


def test_shrink_toward_market_worked_examples():
    ml = {"1": 0.60, "X": 0.20, "2": 0.20}
    market = {"1": 0.50, "X": 0.28, "2": 0.22}
    assert shrink_toward_market(ml, market, alpha=0.0)["1"] == pytest.approx(0.60)
    mid = shrink_toward_market(ml, market, alpha=0.5)
    assert mid["1"] == pytest.approx(0.55)
    assert mid["X"] == pytest.approx(0.24)
    assert mid["2"] == pytest.approx(0.21)
    near = shrink_toward_market(ml, market, alpha=0.9)
    assert near["1"] == pytest.approx(0.51)
    assert near["X"] == pytest.approx(0.272)
    assert near["2"] == pytest.approx(0.218)
    assert shrink_toward_market(ml, market, alpha=1.0)["1"] == pytest.approx(0.50)


def test_label_smoothing_and_target_deltas_recover_soft_label():
    baseline = {"1": 0.50, "X": 0.28, "2": 0.22}
    deltas = target_logit_deltas("1", baseline, label_smoothing=0.05)
    assert deltas["1"] == pytest.approx(logit(0.90) - logit(0.50))
    assert deltas["X"] == pytest.approx(logit(0.05) - logit(0.28))
    assert deltas["2"] == pytest.approx(logit(0.05) - logit(0.22))
    recovered = apply_residual_deltas(baseline, deltas)
    assert recovered["1"] == pytest.approx(0.90)
    assert recovered["X"] == pytest.approx(0.05)
    assert recovered["2"] == pytest.approx(0.05)


def test_apply_residual_deltas_independent_logits_then_renormalize():
    baseline = {"1": 0.50, "X": 0.28, "2": 0.22}
    adjusted = apply_residual_deltas(baseline, {"1": 0.2, "X": -0.1, "2": -0.1})
    expected_raw = {
        "1": inv_logit(logit(0.50) + 0.2),
        "X": inv_logit(logit(0.28) - 0.1),
        "2": inv_logit(logit(0.22) - 0.1),
    }
    total = sum(expected_raw.values())
    assert adjusted["1"] == pytest.approx(expected_raw["1"] / total)
    assert adjusted["X"] == pytest.approx(expected_raw["X"] / total)
    assert adjusted["2"] == pytest.approx(expected_raw["2"] / total)
    assert sum(adjusted.values()) == pytest.approx(1.0)


def test_log_loss_worked_examples():
    assert log_loss_one("1", 0.5, 0.3, 0.2) == pytest.approx(-math.log(0.5))
    assert mean_log_loss(
        ["1", "X"],
        [(0.5, 0.3, 0.2), (0.2, 0.5, 0.3)],
    ) == pytest.approx(-math.log(0.5))
    assert multiclass_log_loss([0], [[0.5, 0.3, 0.2]]) == pytest.approx(-math.log(0.5))


def test_ranked_probability_score_worked_example():
    assert ranked_probability_score("1", 0.5, 0.3, 0.2) == pytest.approx(0.145)
    assert ranked_probability_score("X", 0.5, 0.3, 0.2) == pytest.approx(0.145)
    assert ranked_probability_score("2", 0.5, 0.3, 0.2) == pytest.approx(0.445)


def test_dixon_coles_tau_and_scoreline_worked_example():
    lambda_home, lambda_away, rho = 1.4, 1.1, -0.13
    assert _poisson_pmf(1, lambda_home) == pytest.approx(
        math.exp(-1.4) * 1.4 / math.factorial(1)
    )
    assert _dixon_coles_tau(0, 0, lambda_home, lambda_away, rho) == pytest.approx(
        1.0 - 1.4 * 1.1 * (-0.13)
    )
    assert _dixon_coles_tau(0, 1, lambda_home, lambda_away, rho) == pytest.approx(
        1.0 + 1.4 * (-0.13)
    )
    assert _dixon_coles_tau(1, 0, lambda_home, lambda_away, rho) == pytest.approx(0.857)
    assert _dixon_coles_tau(1, 1, lambda_home, lambda_away, rho) == pytest.approx(
        1.0 - (-0.13)
    )
    assert _dixon_coles_tau(2, 1, lambda_home, lambda_away, rho) == pytest.approx(1.0)
    p_10 = _scoreline_probability(1, 0, lambda_home, lambda_away, rho)
    independent = _poisson_pmf(1, lambda_home) * _poisson_pmf(0, lambda_away)
    assert p_10 == pytest.approx(independent * 0.857)
    _matrix, p_home, p_draw, p_away = dixon_coles_matrix(
        lambda_home, lambda_away, rho=rho, max_goals=10
    )
    assert p_home == pytest.approx(0.4216091376335332)
    assert p_draw == pytest.approx(0.29921178018704997)
    assert p_away == pytest.approx(0.2791790821794171)
    assert p_home + p_draw + p_away == pytest.approx(1.0)


def test_match_weight_exponential_decay_example():
    assert match_weight(0, 0.005) == pytest.approx(1.0)
    assert match_weight(365, 0.005) == pytest.approx(math.exp(-0.005 * 365))
    assert match_weight(-1, 0.005) == 0.0


def test_blend_weight_rule_picks_highest_market_weight():
    config = BlendWeightsConfig(
        enabled=True,
        default_market_weight=0.7,
        default_dc_weight=0.3,
        market_vs_dc=BlendWeightRule(
            enabled=True,
            threshold=0.15,
            market_weight=0.85,
            dc_weight=0.15,
        ),
    )
    quiet = {"market_vs_dc_home": 0.05, "market_vs_dc_draw": 0.01, "market_vs_dc_away": 0.02}
    loud = {"market_vs_dc_home": 0.20, "market_vs_dc_draw": 0.01, "market_vs_dc_away": 0.02}
    assert market_vs_dc_magnitude(loud) == pytest.approx(0.20)
    assert select_blend_weights(quiet, config) == pytest.approx((0.7, 0.3))
    assert select_blend_weights(loud, config) == pytest.approx((0.85, 0.15))


def test_disabled_blend_config_uses_fallback_70_30():
    config = BlendWeightsConfig(enabled=False, default_market_weight=0.85)
    assert select_blend_weights({}, config) == pytest.approx((0.7, 0.3))


def _minimal_features() -> ResidualMLFeatures:
    return ResidualMLFeatures(
        match_id=1,
        draw_number=4950,
        feature_cutoff_date=date(2025, 1, 15),
        league_external_id=39,
        p_home_market=0.50,
        p_draw_market=0.28,
        p_away_market=0.22,
        home_npxg_for=1.2,
        home_npxg_against=1.0,
        away_npxg_for=1.1,
        away_npxg_against=1.0,
        home_opponent_adjusted_attack=1.0,
        home_opponent_adjusted_defence=1.0,
        away_opponent_adjusted_attack=1.0,
        away_opponent_adjusted_defence=1.0,
        home_shot_quality_for=0.1,
        home_shot_quality_conceded=0.1,
        away_shot_quality_for=0.1,
        away_shot_quality_conceded=0.1,
        home_set_piece_attack=1.0,
        home_set_piece_defence=1.0,
        away_set_piece_attack=1.0,
        away_set_piece_defence=1.0,
        home_goalkeeper_prevention=0.0,
        away_goalkeeper_prevention=0.0,
        expected_home_goals=1.4,
        expected_away_goals=1.1,
        p_home_dc=0.40,
        p_draw_dc=0.30,
        p_away_dc=0.30,
        market_vs_dc_home=0.10,
        market_vs_dc_draw=-0.02,
        market_vs_dc_away=-0.08,
        attack_strength_difference=0.1,
        expected_goal_difference=0.3,
        expected_goal_total=2.5,
        market_balance=0.05,
        defence_strength_difference=0.0,
        favourite_strength=0.05,
        combined_draw_rate=0.25,
        combined_one_goal_match_rate=0.2,
        combined_close_match_rate=0.4,
        combined_low_scoring_rate=0.3,
        league_draw_rate=0.25,
        league_home_win_rate=0.45,
        league_away_win_rate=0.30,
        league_avg_goals=2.6,
        league_goal_std=1.2,
        league_favourite_win_rate=0.55,
        league_competitive_balance=0.5,
        league_avg_npxg=1.2,
        league_upset_rate=0.45,
        league_prior_weight=0.5,
        home_rest_days=6,
        away_rest_days=5,
        rest_day_difference=1,
        home_matches_last_14_days=2,
        away_matches_last_14_days=2,
        home_short_rest=0,
        away_short_rest=0,
        home_congestion=0,
        away_congestion=0,
        congestion_difference=0,
        home_extra_time_in_previous_match=0,
        away_extra_time_in_previous_match=0,
        extra_time_x_short_rest=0,
        fatigue_difference=1,
        home_advantage_log=0.1,
        home_advantage_coefficient=0.25,
    )


def test_probability_manager_uses_blend_when_ml_disabled():
    config = DataSourceConfig(residual_ml_enabled=False)
    manager = ProbabilityManager(session=MagicMock(), config=config)
    assert manager.ml_model is None
    manager.assembler = MagicMock()
    manager.assembler.assemble.return_value = _minimal_features()
    match = SimpleNamespace(
        id=10,
        stryktipset_round_id=4950,
        start_time=date(2025, 1, 15),
        home_team=SimpleNamespace(name="Home FC"),
        away_team=SimpleNamespace(name="Away FC"),
    )
    result = manager.process_match(match, event_number=1)
    expected = blend_baselines(
        {"1": 0.50, "X": 0.28, "2": 0.22},
        {"1": 0.40, "X": 0.30, "2": 0.30},
        market_weight=0.7,
        dc_weight=0.3,
    )
    assert result.final_probabilities["1"] == pytest.approx(expected["1"])
    assert result.final_probabilities["X"] == pytest.approx(expected["X"])
    assert result.final_probabilities["2"] == pytest.approx(expected["2"])
    assert result.ml_probabilities is None
