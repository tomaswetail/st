"""Worked-example tests for 1X2 probability formulas."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.calc.probability_manager import ProbabilityManager
from src.calc.probability_metrics import log_loss_one, mean_log_loss, multiclass_log_loss
from src.calc.residual_ml.baseline import (
    apply_residual_deltas,
    inv_logit,
    logit,
    market_baseline,
    shrink_toward_market,
    target_logit_deltas,
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


def test_logit_inv_logit_round_trip():
    assert logit(0.25) == pytest.approx(math.log(0.25 / 0.75))
    assert inv_logit(logit(0.25)) == pytest.approx(0.25)
    assert logit(0.5) == pytest.approx(0.0)
    assert inv_logit(0.0) == pytest.approx(0.5)


def test_shrink_toward_market_worked_examples():
    ml_probabilities = {"1": 0.60, "X": 0.20, "2": 0.20}
    market_probabilities = {"1": 0.50, "X": 0.28, "2": 0.22}
    assert shrink_toward_market(ml_probabilities, market_probabilities, alpha=0.0)["1"] == pytest.approx(0.60)
    mid = shrink_toward_market(ml_probabilities, market_probabilities, alpha=0.5)
    assert mid["1"] == pytest.approx(0.55)
    assert mid["X"] == pytest.approx(0.24)
    assert mid["2"] == pytest.approx(0.21)
    assert shrink_toward_market(ml_probabilities, market_probabilities, alpha=1.0)["1"] == pytest.approx(0.50)


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


def test_probability_manager_returns_market_when_ml_disabled():
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
    expected = market_baseline({"1": 0.50, "X": 0.28, "2": 0.22})
    assert expected is not None
    assert result.final_probabilities["1"] == pytest.approx(expected["1"])
    assert result.final_probabilities["X"] == pytest.approx(expected["X"])
    assert result.final_probabilities["2"] == pytest.approx(expected["2"])
    assert result.ml_probabilities is None
