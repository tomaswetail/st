"""Unit tests for RestCongestionCalculator."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from calc.rest_congestion_calculator import RestCongestionCalculator
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.rest_congestion_features import RestCongestionFeatures


def _match(
    *,
    start_time: datetime = datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc),
    home_name: str = "Arsenal",
    away_name: str = "Chelsea",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        start_time=start_time,
        home_team=SimpleNamespace(name=home_name, league_id=1),
        away_team=SimpleNamespace(name=away_name, league_id=1),
    )


def _historical(
    *,
    match_date: date,
    home_team: str = "Arsenal",
    away_team: str = "Brighton",
    extra_time: bool | None = None,
    went_to_extra_time: bool | None = None,
    raw_data: dict | None = None,
) -> SimpleNamespace:
    status_short = "FT"
    if extra_time or went_to_extra_time:
        status_short = "AET"
    if raw_data and raw_data.get("went_to_extra_time"):
        status_short = "AET"
    return SimpleNamespace(
        fixture_date=match_date,
        home_team_name=home_team,
        away_team_name=away_team,
        status_short=status_short,
        score_extratime_home=1 if status_short == "AET" else None,
        score_extratime_away=None,
    )


def _calculator(
    *,
    home_history: list[SimpleNamespace] | None = None,
    away_history: list[SimpleNamespace] | None = None,
    history_by_team: dict[str, list[SimpleNamespace]] | None = None,
) -> RestCongestionCalculator:
    session = MagicMock()
    config = DataSourceConfig(
        rest_congestion_window_days=14,
        rest_short_rest_threshold_days=4,
        rest_congestion_match_threshold=3,
        rest_congestion_lookback_matches=20,
    )
    calculator = RestCongestionCalculator(session, config=config)
    calculator.fixture_repo = MagicMock()

    def _find(*, team_name, before_date, venue, limit):
        if history_by_team is not None:
            return history_by_team.get(team_name, [])
        if team_name == "Arsenal":
            return home_history or []
        return away_history or []

    calculator.fixture_repo.find_before_date_by_team = MagicMock(side_effect=_find)
    return calculator


def test_normal_rest_days():
    calculator = _calculator(
        home_history=[_historical(match_date=date(2024, 6, 8))],
        away_history=[_historical(match_date=date(2024, 6, 8), home_team="Chelsea")],
    )
    features = calculator.calculate(_match())

    assert features.home_rest_days == 7
    assert features.away_rest_days == 7
    assert features.rest_day_difference == 0
    assert features.home_short_rest == 0
    assert features.away_short_rest == 0
    assert isinstance(features, RestCongestionFeatures)


def test_very_short_rest():
    calculator = _calculator(
        home_history=[_historical(match_date=date(2024, 6, 13))],
        away_history=[_historical(match_date=date(2024, 6, 8), home_team="Chelsea")],
    )
    features = calculator.calculate(_match())

    assert features.home_rest_days == 2
    assert features.home_short_rest == 2
    assert features.away_rest_days == 7
    assert features.away_short_rest == 0
    assert features.rest_day_difference == -5


def test_no_previous_match_uses_neutral_defaults():
    calculator = _calculator(home_history=[], away_history=[])
    features = calculator.calculate(_match())

    assert features.home_rest_days is None
    assert features.away_rest_days is None
    assert features.rest_day_difference is None
    assert features.home_matches_last_14_days == 0
    assert features.away_matches_last_14_days == 0
    assert features.home_short_rest == 0
    assert features.away_short_rest == 0
    assert features.home_extra_time_in_previous_match is False
    assert features.away_extra_time_in_previous_match is False
    assert features.home_lineup_changes is None
    assert features.congestion_x_squad_depth is None
    assert features.short_rest_x_rotation is None


def test_availability_fills_lineup_and_squad_stubs():
    calculator = _calculator(
        home_history=[_historical(match_date=date(2024, 6, 13))],
        away_history=[_historical(match_date=date(2024, 6, 8), home_team="Chelsea")],
    )
    availability = SimpleNamespace(
        has_availability=1,
        home_lineup_changes=3,
        away_lineup_changes=1,
        home_unavailable_count=2,
        away_unavailable_count=1,
    )
    features = calculator.calculate(_match(), availability=availability)

    assert features.home_lineup_changes == 3
    assert features.away_lineup_changes == 1
    # home short_rest=2 (threshold 4 - rest 2), away short_rest=0 (rest 7)
    # matches in 14d: 1 each → congestion 0; unavailable total 3
    assert features.congestion_x_squad_depth == 0.0
    assert features.short_rest_x_rotation == 2.0 * 3 + 0.0 * 1



def test_multiple_matches_within_14_days():
    home_history = [
        _historical(match_date=date(2024, 6, 12)),
        _historical(match_date=date(2024, 6, 8)),
        _historical(match_date=date(2024, 6, 4)),
        _historical(match_date=date(2024, 6, 1)),
        _historical(match_date=date(2024, 5, 20)),
    ]
    calculator = _calculator(
        home_history=home_history,
        away_history=[_historical(match_date=date(2024, 6, 1), home_team="Chelsea")],
    )
    features = calculator.calculate(_match())

    assert features.home_matches_last_14_days == 4
    assert features.home_congestion == 1
    assert features.home_rest_days == 3
    assert features.away_matches_last_14_days == 1
    assert features.away_congestion == 0


def test_same_day_and_future_matches_are_excluded():
    cutoff = date(2024, 6, 15)
    leaked = [
        _historical(match_date=date(2024, 6, 20)),
        _historical(match_date=cutoff),
        _historical(match_date=date(2024, 6, 8)),
    ]
    calculator = _calculator(home_history=leaked, away_history=[])
    features = calculator.calculate(_match())

    assert features.home_rest_days == 7
    assert features.home_matches_last_14_days == 1
    kwargs = calculator.fixture_repo.find_before_date_by_team.call_args_list[0].kwargs
    assert kwargs["before_date"] == cutoff
    assert kwargs["venue"] is None


def test_previous_match_extra_time_sets_interaction():
    calculator = _calculator(
        home_history=[
            _historical(match_date=date(2024, 6, 13), extra_time=True),
        ],
        away_history=[
            _historical(
                match_date=date(2024, 6, 13),
                home_team="Chelsea",
                raw_data={"went_to_extra_time": True},
            ),
        ],
    )
    features = calculator.calculate(_match())

    assert features.home_extra_time_in_previous_match is True
    assert features.away_extra_time_in_previous_match is True
    assert features.home_short_rest == 2
    assert features.home_extra_time_short_rest == 2
    assert features.away_extra_time_short_rest == 2
    assert features.extra_time_x_short_rest == 4


def test_missing_extra_time_metadata_defaults_to_false():
    calculator = _calculator(
        home_history=[_historical(match_date=date(2024, 6, 13))],
        away_history=[
            _historical(match_date=date(2024, 6, 8), home_team="Chelsea", raw_data={})
        ],
    )
    features = calculator.calculate(_match())

    assert features.home_extra_time_in_previous_match is False
    assert features.away_extra_time_in_previous_match is False
    assert features.home_extra_time_short_rest == 0
    assert features.extra_time_x_short_rest == 0


def test_home_and_away_appearances_count_toward_schedule():
    arsenal_history = [
        _historical(
            match_date=date(2024, 6, 12),
            home_team="Arsenal",
            away_team="Everton",
        ),
        _historical(
            match_date=date(2024, 6, 8),
            home_team="Tottenham",
            away_team="Arsenal",
        ),
        _historical(
            match_date=date(2024, 6, 3),
            home_team="Arsenal",
            away_team="Fulham",
        ),
        _historical(
            match_date=date(2024, 6, 1),
            home_team="Newcastle",
            away_team="Arsenal",
        ),
    ]
    calculator = _calculator(history_by_team={"Arsenal": arsenal_history, "Chelsea": []})
    features = calculator.calculate(_match())

    assert features.home_matches_last_14_days == 4
    assert features.home_congestion == 1
    find_calls = calculator.fixture_repo.find_before_date_by_team.call_args_list
    assert find_calls[0].kwargs["team_name"] == "Arsenal"
    assert find_calls[0].kwargs["venue"] is None
