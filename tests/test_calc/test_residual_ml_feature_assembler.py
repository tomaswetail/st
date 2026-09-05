"""Tests for ResidualMLFeatureAssembler."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.objects.schema.data_classes.balance_and_environment_features import (
    BalanceAndEnvironmentFeatures,
)
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.league_behavior_features import LeagueBehaviorFeatures
from src.objects.schema.data_classes.player_availability_features import (
    PlayerAvailabilityFeatures,
)
from src.objects.schema.data_classes.rest_congestion_features import RestCongestionFeatures
from src.objects.schema.data_classes.team_strength_features import MatchStrengthFeatures


def _empty_availability() -> PlayerAvailabilityFeatures:
    return PlayerAvailabilityFeatures(
        home_missing_player_value=None,
        away_missing_player_value=None,
        missing_value_difference=None,
        home_unavailable_count=None,
        away_unavailable_count=None,
        home_lineup_changes=None,
        away_lineup_changes=None,
        missing_value_x_favourite=None,
        short_rest_x_missing_value=None,
        has_availability=0,
    )


def _match():
    return SimpleNamespace(
        id=99,
        stryktipset_round_id=4750,
        start_time=datetime(2025, 8, 15, 18, 0, tzinfo=timezone.utc),
        home_team_id=1,
        away_team_id=2,
        home_team=SimpleNamespace(id=1, name="Arsenal", external_id=42),
        away_team=SimpleNamespace(id=2, name="Chelsea", external_id=43),
        league_name="Premier League",
        match_odds=SimpleNamespace(
            id=1,
            stryktipset_match_id=99,
            odds_1=2.0,
            odds_X=3.5,
            odds_2=3.8,
        ),
    )


def test_assembler_maps_feature_groups():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="strength"),
    )
    assembler.league_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=10, external_id=39)
    )
    assembler.fixture_repo.resolve_internal_league_id_for_team = MagicMock(return_value=10)
    assembler.fixture_repo.find_before_date_by_team = MagicMock(return_value=[])
    assembler.strength_calculator.get_fixture_features = MagicMock(
        return_value=MatchStrengthFeatures(
            match_id=99,
            home_team_id=1,
            away_team_id=2,
            home=None,
            away=None,
            home_npxg_for=1.4,
            home_npxg_against=1.0,
            away_npxg_for=1.2,
            away_npxg_against=1.1,
            home_opponent_adjusted_attack=1.1,
            home_opponent_adjusted_defence=0.9,
            away_opponent_adjusted_attack=1.0,
            away_opponent_adjusted_defence=1.0,
            home_shot_quality_for=0.12,
            home_shot_quality_conceded=0.10,
            away_shot_quality_for=0.11,
            away_shot_quality_conceded=0.09,
            home_set_piece_attack=1.05,
            home_set_piece_defence=0.95,
            away_set_piece_attack=1.0,
            away_set_piece_defence=1.0,
            home_goalkeeper_prevention=0.05,
            away_goalkeeper_prevention=0.02,
            expected_home_goals=1.55,
            expected_away_goals=1.10,
            dixon_coles_home_probability=0.46,
            dixon_coles_draw_probability=0.27,
            dixon_coles_away_probability=0.27,
        )
    )
    assembler.strength_calculator.league_averages_by_league_id = MagicMock(
        return_value={"npxg": 1.35}
    )
    assembler.balance_calculator.calculate = MagicMock(
        return_value=BalanceAndEnvironmentFeatures(
            attack_strength_difference=0.1,
            expected_goal_difference=0.45,
            expected_goal_total=2.65,
            market_balance=0.08,
            defence_strength_difference=0.05,
            favourite_strength=0.08,
            home_recent_draw_rate=0.2,
            away_recent_draw_rate=0.25,
            home_one_goal_match_rate=0.3,
            away_one_goal_match_rate=0.28,
            home_close_match_rate=0.4,
            away_close_match_rate=0.35,
            home_low_scoring_rate=0.2,
            away_low_scoring_rate=0.22,
            combined_draw_rate=0.225,
            combined_one_goal_match_rate=0.29,
            combined_close_match_rate=0.375,
            combined_low_scoring_rate=0.21,
        )
    )
    assembler.league_behavior_calculator.calculate = MagicMock(
        return_value=LeagueBehaviorFeatures(
            league_draw_rate=0.24,
            league_home_win_rate=0.45,
            league_away_win_rate=0.31,
            league_avg_goals=2.7,
            league_goal_std=1.4,
            league_favourite_win_rate=0.58,
            league_competitive_balance=0.5,
            league_promoted_team_effect=0.0,
            league_sample_size=200,
            league_data_quality=0.9,
            league_prior_weight=0.8,
        )
    )
    assembler.rest_calculator.calculate = MagicMock(
        return_value=RestCongestionFeatures(
            home_rest_days=6,
            away_rest_days=4,
            rest_day_difference=2,
            home_matches_last_14_days=2,
            away_matches_last_14_days=3,
            home_short_rest=0,
            away_short_rest=1,
            home_congestion=0,
            away_congestion=1,
            home_extra_time_in_previous_match=False,
            away_extra_time_in_previous_match=True,
            home_extra_time_short_rest=0,
            away_extra_time_short_rest=1,
            extra_time_x_short_rest=1,
            home_lineup_changes=None,
            away_lineup_changes=None,
            congestion_x_squad_depth=None,
            short_rest_x_rotation=None,
        )
    )
    assembler.rest_calculator.previous_fixtures = MagicMock(return_value=(None, None))
    assembler.availability_calculator.calculate = MagicMock(
        return_value=_empty_availability()
    )
    assembler.home_advantage_calculator.process = MagicMock(
        return_value=SimpleNamespace(home_advantage=0.12)
    )

    features = assembler.assemble(_match(), event_number=3)

    assert features.match_id == 99
    assert features.feature_cutoff_date == date(2025, 8, 15)
    assert features.p_home_market == pytest.approx(0.475, rel=1e-2)
    assert features.home_npxg_for == pytest.approx(1.4)
    assert features.p_draw_dc == pytest.approx(0.27)
    assert features.market_vs_dc_home == pytest.approx(
        features.p_home_market - features.p_home_dc
    )
    assert features.market_vs_dc_draw == pytest.approx(
        features.p_draw_market - features.p_draw_dc
    )
    assert features.market_vs_dc_away == pytest.approx(
        features.p_away_market - features.p_away_dc
    )
    assert features.market_balance == pytest.approx(0.08)
    assert features.league_draw_rate == pytest.approx(0.24)
    assert features.league_avg_npxg == pytest.approx(1.35)
    assert features.league_upset_rate == pytest.approx(0.42)
    assert features.rest_day_difference == 2
    assert features.home_advantage_log == pytest.approx(0.12)
    assert features.home_advantage_coefficient == pytest.approx(float(__import__("math").exp(0.12)))
    assert features.travel_distance_km is None
    assert features.has_availability == 0
    assert features.home_missing_player_value is None

    assembler.home_advantage_calculator.process.assert_called_once()
    assembler.availability_calculator.calculate.assert_called_once()
    strength_call = assembler.strength_calculator.get_fixture_features.call_args
    assert strength_call.kwargs["target_league_external_id"] == 39
    assert strength_call.kwargs["home_advantage_coefficient"] == pytest.approx(
        float(__import__("math").exp(0.12))
    )
    balance_call = assembler.balance_calculator.calculate.call_args
    assert (
        balance_call.kwargs["strength"]
        is assembler.strength_calculator.get_fixture_features.return_value
    )


def _stub_assemblers(assembler: ResidualMLFeatureAssembler) -> MatchStrengthFeatures:
    strength = MatchStrengthFeatures(
        match_id=99,
        home_team_id=1,
        away_team_id=2,
        home=None,
        away=None,
        home_npxg_for=1.4,
        home_npxg_against=1.0,
        away_npxg_for=1.2,
        away_npxg_against=1.1,
        expected_home_goals=1.55,
        expected_away_goals=1.10,
        dixon_coles_home_probability=0.46,
        dixon_coles_draw_probability=0.27,
        dixon_coles_away_probability=0.27,
    )
    assembler.league_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=10, external_id=39)
    )
    assembler.fixture_repo.resolve_internal_league_id_for_team = MagicMock(
        return_value=10
    )
    assembler.fixture_repo.find_before_date_by_team = MagicMock(return_value=[])
    assembler.strength_calculator.get_fixture_features = MagicMock(return_value=strength)
    assembler.strength_calculator.league_averages_by_league_id = MagicMock(
        return_value={"npxg": 1.35}
    )
    assembler.balance_calculator.calculate = MagicMock(
        return_value=BalanceAndEnvironmentFeatures(
            attack_strength_difference=0.1,
            expected_goal_difference=0.45,
            expected_goal_total=2.65,
            market_balance=0.08,
            defence_strength_difference=0.05,
            favourite_strength=0.08,
            home_recent_draw_rate=0.2,
            away_recent_draw_rate=0.25,
            home_one_goal_match_rate=0.3,
            away_one_goal_match_rate=0.28,
            home_close_match_rate=0.4,
            away_close_match_rate=0.35,
            home_low_scoring_rate=0.2,
            away_low_scoring_rate=0.22,
            combined_draw_rate=0.225,
            combined_one_goal_match_rate=0.29,
            combined_close_match_rate=0.375,
            combined_low_scoring_rate=0.21,
        )
    )
    assembler.league_behavior_calculator.calculate = MagicMock(
        return_value=LeagueBehaviorFeatures(
            league_draw_rate=0.24,
            league_home_win_rate=0.45,
            league_away_win_rate=0.31,
            league_avg_goals=2.7,
            league_goal_std=1.4,
            league_favourite_win_rate=0.58,
            league_competitive_balance=0.5,
            league_promoted_team_effect=0.0,
            league_sample_size=200,
            league_data_quality=0.9,
            league_prior_weight=0.8,
        )
    )
    assembler.rest_calculator.calculate = MagicMock(
        return_value=RestCongestionFeatures(
            home_rest_days=6,
            away_rest_days=4,
            rest_day_difference=2,
            home_matches_last_14_days=2,
            away_matches_last_14_days=3,
            home_short_rest=0,
            away_short_rest=1,
            home_congestion=0,
            away_congestion=1,
            home_extra_time_in_previous_match=False,
            away_extra_time_in_previous_match=True,
            home_extra_time_short_rest=0,
            away_extra_time_short_rest=1,
            extra_time_x_short_rest=1,
            home_lineup_changes=None,
            away_lineup_changes=None,
            congestion_x_squad_depth=None,
            short_rest_x_rotation=None,
        )
    )
    assembler.rest_calculator.previous_fixtures = MagicMock(return_value=(None, None))
    assembler.availability_calculator.calculate = MagicMock(
        return_value=_empty_availability()
    )
    assembler.home_advantage_calculator.process = MagicMock(
        return_value=SimpleNamespace(home_advantage=0.12)
    )
    return strength


def test_assembler_classic_engine_uses_classic_dc_probs():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="classic"),
    )
    _stub_assemblers(assembler)
    classic_model = MagicMock()
    classic_model.predict.return_value = SimpleNamespace(
        lambda_home=1.8,
        lambda_away=0.9,
        p_home=0.55,
        p_draw=0.25,
        p_away=0.20,
    )
    assembler.dixon_coles_service.fit_league = MagicMock(return_value=classic_model)

    features = assembler.assemble(_match())

    assert features.p_home_dc == pytest.approx(0.55)
    assert features.p_draw_dc == pytest.approx(0.25)
    assert features.p_away_dc == pytest.approx(0.20)
    assert features.expected_home_goals == pytest.approx(1.8)
    assert features.expected_away_goals == pytest.approx(0.9)
    classic_model.predict.assert_called_once_with(42, 43)
    assembler.dixon_coles_service.fit_league.assert_called_once_with(
        39, date(2025, 8, 15)
    )


def test_assembler_strength_engine_keeps_strength_dc():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="strength"),
    )
    _stub_assemblers(assembler)
    assembler.dixon_coles_service.fit_league = MagicMock()

    features = assembler.assemble(_match())

    assert features.p_home_dc == pytest.approx(0.46)
    assert features.p_draw_dc == pytest.approx(0.27)
    assert features.expected_home_goals == pytest.approx(1.55)
    assembler.dixon_coles_service.fit_league.assert_not_called()


def test_assembler_classic_fit_failure_omits_engine_probs():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="classic"),
    )
    _stub_assemblers(assembler)
    assembler.dixon_coles_service.fit_league = MagicMock(
        side_effect=ValueError("not enough matches")
    )

    features = assembler.assemble(_match())

    assert features.p_home_dc is None
    assert features.p_draw_dc is None
    assert features.p_away_dc is None
    assert features.expected_home_goals is None
    assert features.expected_away_goals is None
    assert features.market_vs_dc_home is None
    assert features.market_vs_dc_draw is None
    assert features.market_vs_dc_away is None
    assert features.home_npxg_for == pytest.approx(1.4)
    assert features.attack_strength_difference == pytest.approx(0.1)


def test_assembler_classic_predict_failure_omits_engine_probs():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="classic"),
    )
    _stub_assemblers(assembler)
    classic_model = MagicMock()
    classic_model.predict.side_effect = RuntimeError("unknown team")
    assembler.dixon_coles_service.fit_league = MagicMock(return_value=classic_model)

    features = assembler.assemble(_match())

    assert features.p_home_dc is None
    assert features.p_draw_dc is None
    assert features.p_away_dc is None
    assert features.expected_home_goals is None
    assert features.expected_away_goals is None
    assert features.market_vs_dc_home is None
    assert features.home_npxg_for == pytest.approx(1.4)
    assert features.attack_strength_difference == pytest.approx(0.1)


def test_assembler_classic_fit_cached_per_league_day():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(
        session,
        config=DataSourceConfig(residual_ml_dc_engine="classic"),
    )
    _stub_assemblers(assembler)
    classic_model = MagicMock()
    classic_model.predict.return_value = SimpleNamespace(
        lambda_home=1.5,
        lambda_away=1.0,
        p_home=0.5,
        p_draw=0.25,
        p_away=0.25,
    )
    assembler.dixon_coles_service.fit_league = MagicMock(return_value=classic_model)

    first = _match()
    second = _match()
    second.id = 100
    assembler.assemble(first)
    assembler.assemble(second)

    assert assembler.dixon_coles_service.fit_league.call_count == 1
    assert classic_model.predict.call_count == 2

