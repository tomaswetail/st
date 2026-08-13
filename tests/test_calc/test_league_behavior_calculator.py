"""Unit tests for LeagueBehaviorCalculator."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.league_behavior_calculator import LeagueBehaviorCalculator
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.league_behavior_features import LeagueBehaviorFeatures


def _match(
    *,
    start_time: datetime = datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc),
    league_id: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        start_time=start_time,
        home_team=SimpleNamespace(league_id=league_id, name="Arsenal"),
        away_team=SimpleNamespace(league_id=league_id, name="Chelsea"),
    )


def _historical(
    *,
    match_date: date,
    home_team: str = "A",
    away_team: str = "B",
    home_goals: int = 1,
    away_goals: int = 0,
    result: str | None = None,
    odds_home: float | None = None,
    odds_draw: float | None = None,
    odds_away: float | None = None,
) -> SimpleNamespace:
    if result is None:
        if home_goals > away_goals:
            result = "1"
        elif home_goals < away_goals:
            result = "2"
        else:
            result = "X"
    return SimpleNamespace(
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        result=result,
        odds_home=odds_home,
        odds_draw=odds_draw,
        odds_away=odds_away,
    )


def _calculator(
    *,
    league_matches: list[SimpleNamespace],
    global_matches: list[SimpleNamespace] | None = None,
    shrinkage: int = 0,
    lookback: int = 500,
    min_team_matches: int = 1,
    quality_reference: int = 100,
) -> LeagueBehaviorCalculator:
    session = MagicMock()
    config = DataSourceConfig(
        league_behavior_lookback_matches=lookback,
        league_behavior_shrinkage_matches=shrinkage,
        league_behavior_min_team_matches_for_balance=min_team_matches,
        league_behavior_quality_reference_matches=quality_reference,
    )
    calculator = LeagueBehaviorCalculator(session, config=config)
    calculator.historical_repo = MagicMock()
    calculator.historical_repo.find_before_date_by_league_id = MagicMock(
        return_value=league_matches
    )
    calculator.historical_repo.get_filtered = MagicMock(
        return_value=global_matches if global_matches is not None else league_matches
    )
    return calculator


def test_result_rates_and_goal_environment():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=2, away_goals=1),  # 1
        _historical(match_date=date(2024, 6, 2), home_goals=1, away_goals=1),  # X
        _historical(match_date=date(2024, 6, 3), home_goals=0, away_goals=2),  # 2
        _historical(match_date=date(2024, 6, 4), home_goals=3, away_goals=0),  # 1
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    features = calculator.calculate(_match())

    assert features.league_sample_size == 4
    assert features.league_home_win_rate == pytest.approx(0.5)
    assert features.league_draw_rate == pytest.approx(0.25)
    assert features.league_away_win_rate == pytest.approx(0.25)
    assert features.league_avg_goals == pytest.approx(2.5)
    assert features.league_goal_std == pytest.approx(
        __import__("statistics").pstdev([3, 2, 2, 3])
    )
    assert features.league_promoted_team_effect == 0.0
    assert features.league_prior_weight == pytest.approx(1.0)


def test_favourite_win_rate_uses_market_not_result():
    # Home won, but away was the market favourite (shorter away odds).
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_goals=2,
            away_goals=0,
            odds_home=4.0,
            odds_draw=3.5,
            odds_away=1.5,
        ),
        _historical(
            match_date=date(2024, 6, 2),
            home_goals=0,
            away_goals=1,
            odds_home=4.0,
            odds_draw=3.5,
            odds_away=1.5,
        ),
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    features = calculator.calculate(_match())

    # Favourite is away both times; favourite won once.
    assert features.league_favourite_win_rate == pytest.approx(0.5)


def test_shrinkage_pulls_sparse_league_toward_global():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),  # only 1s
        _historical(match_date=date(2024, 6, 2), home_goals=2, away_goals=0),
    ]
    global_matches = [
        _historical(match_date=date(2024, 5, day), home_goals=1, away_goals=1)
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=100,
    )
    features = calculator.calculate(_match())

    assert features.league_prior_weight == pytest.approx(2 / 102)
    # Raw league home win rate = 1.0; global draw-heavy home rate = 0.0
    assert features.league_home_win_rate == pytest.approx(
        (2 / 102) * 1.0 + (100 / 102) * 0.0
    )
    assert features.league_draw_rate == pytest.approx(
        (2 / 102) * 0.0 + (100 / 102) * 1.0
    )


def test_missing_market_odds_uses_global_favourite_prior():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),
        _historical(match_date=date(2024, 6, 2), home_goals=0, away_goals=0),
    ]
    global_matches = [
        _historical(
            match_date=date(2024, 5, day),
            home_goals=1,
            away_goals=0,
            odds_home=1.5,
            odds_draw=4.0,
            odds_away=5.0,
        )
        for day in range(1, 11)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=50,
    )
    features = calculator.calculate(_match())

    # n_market=0 → full global favourite win rate (home favourite always won)
    assert features.league_favourite_win_rate == pytest.approx(1.0)


def test_no_future_or_same_day_leakage_in_repo_call():
    calculator = _calculator(league_matches=[], global_matches=[], shrinkage=0)
    match = _match(start_time=datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc))
    calculator.calculate(match)

    kwargs = calculator.historical_repo.find_before_date_by_league_id.call_args.kwargs
    assert kwargs["before_date"] == date(2024, 6, 15)
    assert kwargs["league_id"] == 1
    global_kwargs = calculator.historical_repo.get_filtered.call_args.kwargs
    assert global_kwargs["before_date"] == date(2024, 6, 15)


def test_promoted_team_effect_is_neutral():
    calculator = _calculator(
        league_matches=[
            _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=1)
        ],
        shrinkage=0,
    )
    features = calculator.calculate(_match())
    assert features.league_promoted_team_effect == 0.0


def test_cache_avoids_second_repo_query():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0)
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    match = _match()
    first = calculator.calculate(match)
    second = calculator.calculate(match)

    assert first == second
    assert isinstance(first, LeagueBehaviorFeatures)
    assert (
        calculator.historical_repo.find_before_date_by_league_id.call_count == 1
    )


def test_competitive_balance_higher_when_teams_closer():
    balanced = [
        _historical(
            match_date=date(2024, 6, day),
            home_team="A",
            away_team="B",
            home_goals=1,
            away_goals=1,
        )
        for day in range(1, 5)
    ] + [
        _historical(
            match_date=date(2024, 5, day),
            home_team="C",
            away_team="D",
            home_goals=1,
            away_goals=1,
        )
        for day in range(1, 5)
    ]
    unbalanced = [
        _historical(
            match_date=date(2024, 6, day),
            home_team="Strong",
            away_team="Weak",
            home_goals=4,
            away_goals=0,
        )
        for day in range(1, 5)
    ] + [
        _historical(
            match_date=date(2024, 5, day),
            home_team="Strong",
            away_team="Weak",
            home_goals=3,
            away_goals=0,
        )
        for day in range(1, 5)
    ]

    balanced_features = _calculator(
        league_matches=balanced, shrinkage=0, min_team_matches=2
    ).calculate(_match())
    unbalanced_features = _calculator(
        league_matches=unbalanced, shrinkage=0, min_team_matches=2
    ).calculate(_match())

    assert balanced_features.league_competitive_balance is not None
    assert unbalanced_features.league_competitive_balance is not None
    assert (
        balanced_features.league_competitive_balance
        > unbalanced_features.league_competitive_balance
    )


def test_data_quality_increases_with_sample_and_market_coverage():
    sparse = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),
    ]
    rich = [
        _historical(
            match_date=date(2024, 6, 1) - timedelta(days=day),
            home_team=f"H{day}",
            away_team=f"A{day}",
            home_goals=1,
            away_goals=0,
            odds_home=1.8,
            odds_draw=3.5,
            odds_away=4.0,
        )
        for day in range(1, 51)
    ]
    sparse_q = _calculator(
        league_matches=sparse, shrinkage=0, quality_reference=100
    ).calculate(_match()).league_data_quality
    rich_q = _calculator(
        league_matches=rich, shrinkage=0, quality_reference=100
    ).calculate(_match()).league_data_quality
    assert 0.0 <= sparse_q <= 1.0
    assert 0.0 <= rich_q <= 1.0
    assert rich_q > sparse_q
