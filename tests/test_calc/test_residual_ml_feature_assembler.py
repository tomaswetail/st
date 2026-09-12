"""Tests for ResidualMLFeatureAssembler (market baseline only)."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.calc.market_probabilities import MarketProbabilities
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
        league_country_name="England",
        match_odds=SimpleNamespace(
            id=1,
            stryktipset_match_id=99,
            odds_1=2.0,
            odds_X=3.5,
            odds_2=3.8,
        ),
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
    )
    assembler.league_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=10, external_id=39)
    )
    assembler.league_repo.get_by_name_and_country = MagicMock(
        return_value=SimpleNamespace(id=10, external_id=39)
    )
    assembler.fixture_repo.resolve_internal_league_id_for_team = MagicMock(return_value=10)
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


def test_assembler_maps_feature_groups():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    _stub_assemblers(assembler)
    match = _match()

    features = assembler.assemble(match, event_number=3)

    assert features.match_id == 99
    assert features.feature_cutoff_date == date(2025, 8, 15)
    assert features.p_home_market == pytest.approx(0.475, rel=1e-2)
    assert features.home_npxg_for == pytest.approx(1.4)
    assert features.market_balance == pytest.approx(0.08)
    assert features.league_draw_rate == pytest.approx(0.24)
    assert features.league_avg_npxg == pytest.approx(1.35)
    assert features.league_upset_rate == pytest.approx(0.42)
    assert features.rest_day_difference == 2
    assert features.home_advantage_log == pytest.approx(0.12)
    assert features.home_advantage_coefficient == pytest.approx(math.exp(0.12))
    assert features.travel_distance_km is None
    assert features.has_availability == 0
    assert features.home_missing_player_value is None

    assembler.home_advantage_calculator.process.assert_called_once()
    assembler.availability_calculator.calculate.assert_called_once()
    strength_call = assembler.strength_calculator.get_fixture_features.call_args
    assert strength_call.args[0] == 1
    assert strength_call.args[1] == 2
    assert strength_call.args[2] == match.start_time
    assert strength_call.kwargs["target_league_external_id"] == 39
    assert strength_call.kwargs["home_advantage_coefficient"] == pytest.approx(
        math.exp(0.12)
    )
    balance_call = assembler.balance_calculator.calculate.call_args
    assert (
        balance_call.kwargs["strength"]
        is assembler.strength_calculator.get_fixture_features.return_value
    )
    expected_market = MarketProbabilities.from_decimal_odds(2.0, 3.5, 3.8)
    assert features.market_overround == pytest.approx(expected_market.overround)
    assert features.market_entropy == pytest.approx(expected_market.market_entropy)
    assert features.market_top_probability == pytest.approx(
        expected_market.market_top_probability
    )
    assert features.market_second_probability == pytest.approx(
        expected_market.market_second_probability
    )
    assert features.market_probability_gap == pytest.approx(
        expected_market.market_probability_gap
    )
    assert features.market_price_type is None
    assert "market_price_type" not in features.model_feature_names()
    assert "market_overround" in features.model_feature_names()


def test_before_date_override_is_threaded_to_history_queries():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    _stub_assemblers(assembler)
    override = date(2024, 1, 1)

    assembler.assemble(_match(), before_date=override)

    strength_before = assembler.strength_calculator.get_fixture_features.call_args.args[2]
    assert strength_before == override
    assert assembler.home_advantage_calculator.process.call_args.args[1] == override
    assert (
        assembler.league_behavior_calculator.calculate.call_args.kwargs["before_date"]
        == override
    )
    assert assembler.rest_calculator.calculate.call_args.kwargs["before_date"] == override
    assert (
        assembler.availability_calculator.calculate.call_args.kwargs["before_date"]
        == override
    )
    assert assembler.balance_calculator.calculate.call_args.kwargs["before_date"] == override


def _fixture():
    return SimpleNamespace(
        id=55,
        fixture_id=9055,
        fixture_date=datetime(2024, 3, 10, 15, 0, tzinfo=timezone.utc),
        home_team_id=42,
        away_team_id=43,
        home_team_name="Arsenal",
        away_team_name="Chelsea",
        home_team=None,
        away_team=None,
        goals_home=2,
        goals_away=1,
        league_id=39,
        league_name="Premier League",
        league_country="England",
        league_season=2023,
        status_short="FT",
    )


def test_assemble_fixture_resolves_internal_team_ids_and_fixture_odds(monkeypatch):
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    _stub_assemblers(assembler)
    home_team = SimpleNamespace(id=1, external_id=42, name="Arsenal", national=False)
    away_team = SimpleNamespace(id=2, external_id=43, name="Chelsea", national=False)
    assembler.team_repo.get_by_external_id = MagicMock(
        side_effect=lambda external_id: {42: home_team, 43: away_team}.get(int(external_id))
    )
    assembler.team_repo.get = MagicMock(side_effect=AssertionError("must not use teams.id lookup for API ids"))
    assembler.league_repo.get_by_external_id = MagicMock(
        return_value=SimpleNamespace(id=10, external_id=39)
    )
    breakdown = MarketProbabilities.from_decimal_odds(
        1.80, 3.60, 4.50, bookmaker="Avg", price_type="closing"
    )
    monkeypatch.setattr(
        "src.calc.residual_ml.feature_assembler.load_fixture_market_probabilities",
        lambda *_args, **_kwargs: breakdown,
    )

    features = assembler.assemble(_fixture(), before_date=date(2024, 3, 10))

    assert features.match_id == 55
    assert features.draw_number is None
    assert features.league_external_id == 39
    assert features.feature_cutoff_date == date(2024, 3, 10)
    assert features.p_home_market == pytest.approx(breakdown.p_home)
    assert features.market_overround == pytest.approx(breakdown.overround)
    assert features.market_price_type == "closing"
    assembler.team_repo.get_by_external_id.assert_any_call(42)
    assembler.team_repo.get_by_external_id.assert_any_call(43)
    strength_call = assembler.strength_calculator.get_fixture_features.call_args
    assert strength_call.args[0] == 1
    assert strength_call.args[1] == 2
    assert strength_call.args[2] == date(2024, 3, 10)
    assembler.team_repo.get.assert_not_called()


def test_assemble_fixture_skips_unresolved_home_team():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    _stub_assemblers(assembler)
    assembler.team_repo.get_by_external_id = MagicMock(return_value=None)
    with pytest.raises(ValueError, match="Unresolved home team"):
        assembler.assemble(_fixture())


def test_assemble_fixture_skips_when_odds_missing(monkeypatch):
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    _stub_assemblers(assembler)
    assembler.team_repo.get_by_external_id = MagicMock(
        side_effect=lambda external_id: SimpleNamespace(
            id=int(external_id),
            external_id=int(external_id),
            name="T",
            national=False,
        )
    )
    monkeypatch.setattr(
        "src.calc.residual_ml.feature_assembler.load_fixture_market_probabilities",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="No usable fixture odds"):
        assembler.assemble(_fixture())


def test_resolve_league_name_miss_falls_back_to_fixture_helper():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    assembler.league_repo.get_by_name = MagicMock(return_value=None)
    assembler.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    assembler.fixture_repo.resolve_league_external_id_for_match = MagicMock(
        return_value=39
    )
    match = _match()
    match.league_name = "Unmapped Svenska Spel Name"
    match.league_country_name = "England"

    resolved = assembler._resolve_league_external_id(match)

    assert resolved == 39
    assembler.fixture_repo.resolve_league_external_id_for_match.assert_called_once()
    assert (
        assembler.fixture_repo.resolve_league_external_id_for_match.call_args.kwargs[
            "skip_name"
        ]
        is True
    )


def test_resolve_league_name_cache_miss_does_not_block_later_fallback():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    assembler.league_repo.get_by_name = MagicMock(return_value=None)
    assembler.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    assembler.fixture_repo.resolve_league_external_id_for_match = MagicMock(
        side_effect=[None, 180]
    )

    first = _match()
    first.league_name = "Allsvenskan Weird"
    first.league_country_name = "Sweden"
    second = _match()
    second.id = 100
    second.league_name = "Allsvenskan Weird"
    second.league_country_name = "Sweden"
    second.home_team = SimpleNamespace(id=3, name="AIK", external_id=99)

    assert assembler._resolve_league_external_id(first) is None
    assert assembler._resolve_league_external_id(second) == 180
    assert "Allsvenskan Weird" not in [
        key[0] for key in assembler._league_external_id_cache
    ]
    assert assembler.fixture_repo.resolve_league_external_id_for_match.call_count == 2


def test_resolve_league_successful_name_is_cached():
    session = MagicMock()
    assembler = ResidualMLFeatureAssembler(session, config=DataSourceConfig())
    assembler.league_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(external_id=39)
    )
    assembler.league_repo.get_by_name_and_country = MagicMock(
        return_value=SimpleNamespace(external_id=39)
    )
    assembler.fixture_repo.resolve_league_external_id_for_match = MagicMock()

    match = _match()
    assert assembler._resolve_league_external_id(match) == 39
    assert assembler._resolve_league_external_id(match) == 39
    assembler.league_repo.get_by_name_and_country.assert_called_once()
    assembler.fixture_repo.resolve_league_external_id_for_match.assert_not_called()
