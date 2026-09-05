"""Phase 4 production alignment: league metadata + live shrink."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.calc.probability_manager import ProbabilityManager
from src.calc.residual_ml.baseline import shrink_toward_market
from src.calc.residual_ml.dataset import ResidualMLDatasetBuilder
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures


def _minimal_features(*, league_external_id: int | None = 39) -> ResidualMLFeatures:
    return ResidualMLFeatures(
        match_id=1,
        draw_number=4950,
        feature_cutoff_date=date(2025, 1, 15),
        league_external_id=league_external_id,
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
        p_home_dc=0.48,
        p_draw_dc=0.29,
        p_away_dc=0.23,
        market_vs_dc_home=0.02,
        market_vs_dc_draw=-0.01,
        market_vs_dc_away=-0.01,
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


def test_features_include_league_external_id_in_row_dict():
    features = _minimal_features(league_external_id=39)
    row = ResidualMLDatasetBuilder._features_to_row(features)
    assert row["league_external_id"] == 39
    assert row["draw_number"] == 4950
    # Metadata must not be treated as an HGB model feature.
    assert "league_external_id" not in features.model_feature_names()


def test_features_allow_missing_league_external_id():
    features = _minimal_features(league_external_id=None)
    row = ResidualMLDatasetBuilder._features_to_row(features)
    assert row["league_external_id"] is None


def test_probability_manager_applies_final_shrink_to_market():
    config = DataSourceConfig(
        residual_ml_enabled=False,
        residual_ml_final_shrink_to_market=0.5,
    )
    manager = ProbabilityManager(session=MagicMock(), config=config)
    manager.ml_model = MagicMock()
    manager.ml_model.version = "test"
    manager.ml_model.final_shrink_to_market = 0.0
    raw_ml = {"1": 0.60, "X": 0.20, "2": 0.20}
    manager.ml_model.predict_proba.return_value = raw_ml

    features = _minimal_features()
    manager.assembler = MagicMock()
    manager.assembler.assemble.return_value = features

    match = SimpleNamespace(
        id=10,
        stryktipset_round_id=4950,
        start_time=date(2025, 1, 15),
        home_team=SimpleNamespace(name="Home FC"),
        away_team=SimpleNamespace(name="Away FC"),
    )
    result = manager.process_match(match, event_number=1)
    expected = shrink_toward_market(
        raw_ml,
        {"1": 0.50, "X": 0.28, "2": 0.22},
        alpha=0.5,
    )
    assert result.final_probabilities is not None
    assert result.final_probabilities["1"] == pytest.approx(expected["1"])
    assert result.ml_probabilities == raw_ml
    assert sum(result.final_probabilities.values()) == pytest.approx(1.0)


def test_probability_manager_no_shrink_when_alpha_zero():
    config = DataSourceConfig(
        residual_ml_enabled=False,
        residual_ml_final_shrink_to_market=0.0,
    )
    manager = ProbabilityManager(session=MagicMock(), config=config)
    manager.ml_model = MagicMock()
    manager.ml_model.version = "test"
    raw_ml = {"1": 0.60, "X": 0.20, "2": 0.20}
    manager.ml_model.predict_proba.return_value = raw_ml
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
    assert result.final_probabilities == raw_ml
