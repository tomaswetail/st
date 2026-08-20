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
        league_name="Premier League",
        home_team=SimpleNamespace(league_id=league_id, name="Arsenal", external_id=1),
        away_team=SimpleNamespace(league_id=league_id, name="Chelsea", external_id=2),
    )


def _historical(
    *,
    match_date: date,
    home_team: str = "A",
    away_team: str = "B",
    home_goals: int | None = 1,
    away_goals: int | None = 0,
    result: str | None = None,
    odds_home: float | None = None,
    odds_draw: float | None = None,
    odds_away: float | None = None,
    home_promoted: bool | None = None,
    away_promoted: bool | None = None,
    raw_data: dict | None = None,
) -> SimpleNamespace:
    if result is None and home_goals is not None and away_goals is not None:
        if home_goals > away_goals:
            result = "1"
        elif home_goals < away_goals:
            result = "2"
        else:
            result = "X"
    payload = {
        "fixture_date": match_date,
        "home_team_name": home_team,
        "away_team_name": away_team,
        "goals_home": home_goals,
        "goals_away": away_goals,
        "result": result,
        "odds_home": odds_home,
        "odds_draw": odds_draw,
        "odds_away": odds_away,
        "raw_data": raw_data,
    }
    if home_promoted is not None:
        payload["home_promoted"] = home_promoted
    if away_promoted is not None:
        payload["away_promoted"] = away_promoted
    return SimpleNamespace(**payload)


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
    calculator.fixture_repo = MagicMock()
    calculator.fixture_repo.find_before_date_by_league_id = MagicMock(
        return_value=league_matches
    )
    calculator.fixture_repo.get_filtered = MagicMock(
        return_value=global_matches if global_matches is not None else league_matches
    )
    calculator.league_repo = MagicMock()
    calculator.league_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=1)
    )
    return calculator


def test_result_rates_and_goal_environment():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=2, away_goals=1),
        _historical(match_date=date(2024, 6, 2), home_goals=1, away_goals=1),
        _historical(match_date=date(2024, 6, 3), home_goals=0, away_goals=2),
        _historical(match_date=date(2024, 6, 4), home_goals=3, away_goals=0),
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    features = calculator.calculate(_match())

    assert features.league_sample_size == 4
    assert features.league_home_win_rate == pytest.approx(0.5)
    assert features.league_draw_rate == pytest.approx(0.25)
    assert features.league_away_win_rate == pytest.approx(0.25)
    assert features.league_avg_goals == pytest.approx(2.5)
    assert features.league_promoted_team_effect == 0.0
    assert features.league_prior_weight == pytest.approx(1.0)


def test_missing_result_matches_do_not_dilute_result_rates():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),
        _historical(match_date=date(2024, 6, 2), home_goals=1, away_goals=0),
        _historical(
            match_date=date(2024, 6, 3),
            home_goals=None,
            away_goals=None,
            result=None,
        ),
        _historical(
            match_date=date(2024, 6, 4),
            home_goals=None,
            away_goals=None,
            result=None,
        ),
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    features = calculator.calculate(_match())

    assert features.league_sample_size == 4
    assert features.league_home_win_rate == pytest.approx(1.0)


def test_same_day_matches_do_not_consume_lookback_limit():
    cutoff = date(2024, 6, 15)
    # Simulate a buggy repo response that includes same-day rows inside the limit.
    leaked = [
        _historical(match_date=cutoff, home_goals=9, away_goals=0),
        _historical(match_date=date(2024, 6, 14), home_goals=1, away_goals=0),
        _historical(match_date=date(2024, 6, 13), home_goals=2, away_goals=0),
        _historical(match_date=date(2024, 6, 12), home_goals=0, away_goals=1),
    ]
    calculator = _calculator(league_matches=leaked, shrinkage=0, lookback=3)
    features = calculator.calculate(_match())

    # Same-day row dropped; newest 3 strictly-prior remaining → only 3 exist.
    assert features.league_sample_size == 3
    assert features.league_home_win_rate == pytest.approx(2 / 3)
    kwargs = calculator.fixture_repo.find_before_date_by_league_id.call_args.kwargs
    assert kwargs["before_date"] == cutoff
    assert kwargs["limit"] == 3


def test_newest_n_strictly_prior_matches_are_used():
    older = [
        _historical(match_date=date(2024, 5, day), home_goals=0, away_goals=1)
        for day in range(1, 6)
    ]
    newer = [
        _historical(match_date=date(2024, 6, day), home_goals=1, away_goals=0)
        for day in range(1, 4)
    ]
    calculator = _calculator(
        league_matches=older + newer,
        shrinkage=0,
        lookback=3,
    )
    features = calculator.calculate(_match())

    assert features.league_sample_size == 3
    # Newest three are June 1–3 home wins.
    assert features.league_home_win_rate == pytest.approx(1.0)
    assert features.league_away_win_rate == pytest.approx(0.0)


def test_valid_odds_missing_result_count_toward_market_completeness_only():
    """Fixture odds were dropped; market favourite metrics stay empty."""
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_goals=1,
            away_goals=0,
            odds_home=1.5,
            odds_draw=4.0,
            odds_away=5.0,
        ),
        _historical(
            match_date=date(2024, 6, 2),
            home_goals=None,
            away_goals=None,
            result=None,
            odds_home=1.6,
            odds_draw=3.8,
            odds_away=5.5,
        ),
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    stats = calculator._compute_raw_stats(league_matches)

    assert stats.market_data_sample_size == 0
    assert stats.favourite_result_sample_size == 0
    assert stats.market_completeness == pytest.approx(0.0)
    assert stats.favourite_win_rate is None


def test_favourite_reliability_shrinkage_uses_favourite_result_sample_size():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_goals=1,
            away_goals=0,
            odds_home=1.5,
            odds_draw=4.0,
            odds_away=5.0,
        ),
        _historical(
            match_date=date(2024, 6, 2),
            home_goals=None,
            away_goals=None,
            result=None,
            odds_home=1.6,
            odds_draw=3.8,
            odds_away=5.5,
        ),
    ]
    global_matches = [
        _historical(match_date=date(2024, 5, day), home_goals=1, away_goals=1)
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=10,
    )
    features = calculator.calculate(_match())

    # No fixture odds → favourite sample size 0 → full default prior.
    assert features.league_favourite_win_rate == pytest.approx(0.55)


def test_shrinkage_uses_feature_specific_sample_sizes():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_goals=1,
            away_goals=0,
            odds_home=1.5,
            odds_draw=4.0,
            odds_away=5.0,
        ),
        _historical(
            match_date=date(2024, 6, 2),
            home_goals=None,
            away_goals=None,
            result=None,
        ),
        _historical(
            match_date=date(2024, 6, 3),
            home_goals=2,
            away_goals=2,
            result="X",
        ),
    ]
    global_matches = [
        _historical(match_date=date(2024, 5, day), home_goals=1, away_goals=1)
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=10,
    )
    features = calculator.calculate(_match())

    assert features.league_home_win_rate == pytest.approx(
        (2 / 12) * 0.5 + (10 / 12) * 0.0
    )
    assert features.league_favourite_win_rate == pytest.approx(0.55)


def test_favourite_win_rate_uses_market_not_result():
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
    # Fixture odds dropped: favourite metric falls back to default prior.
    assert features.league_favourite_win_rate == pytest.approx(0.55)


def test_shrinkage_pulls_sparse_league_toward_global():
    league_matches = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),
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
    assert features.league_home_win_rate == pytest.approx(
        (2 / 102) * 1.0 + (100 / 102) * 0.0
    )


def test_complete_high_sample_league_relies_on_league_stats():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1) - timedelta(days=day),
            home_goals=1,
            away_goals=0,
        )
        for day in range(1, 101)
    ]
    global_matches = [
        _historical(match_date=date(2024, 1, day), home_goals=0, away_goals=1)
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=10,
    )
    features = calculator.calculate(_match())

    assert features.league_prior_weight == pytest.approx(100 / 110)
    assert features.league_home_win_rate == pytest.approx(
        (100 / 110) * 1.0 + (10 / 110) * 0.0
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
    # Global favourite also unavailable without odds → default prior 0.55.
    assert features.league_favourite_win_rate == pytest.approx(0.55)


def test_missing_goal_data_reduces_league_data_quality():
    with_goals = [
        _historical(match_date=date(2024, 6, day), home_goals=1, away_goals=0)
        for day in range(1, 11)
    ]
    missing_goals = [
        _historical(
            match_date=date(2024, 6, day),
            home_goals=None,
            away_goals=None,
            result="1",
        )
        for day in range(1, 11)
    ]
    quality_with = _calculator(
        league_matches=with_goals, shrinkage=0, quality_reference=10
    ).calculate(_match()).league_data_quality
    quality_missing = _calculator(
        league_matches=missing_goals, shrinkage=0, quality_reference=10
    ).calculate(_match()).league_data_quality
    assert quality_with > quality_missing


def test_strict_exclusion_of_target_day_and_future_matches():
    cutoff = date(2024, 6, 15)
    leaked = [
        _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=0),
        _historical(match_date=cutoff, home_goals=5, away_goals=0),
        _historical(match_date=date(2024, 6, 20), home_goals=4, away_goals=0),
    ]
    calculator = _calculator(league_matches=leaked, shrinkage=0)
    features = calculator.calculate(_match())

    assert features.league_sample_size == 1
    assert features.league_home_win_rate == pytest.approx(1.0)


def test_no_future_or_same_day_leakage_in_repo_call():
    calculator = _calculator(league_matches=[], global_matches=[], shrinkage=0)
    match = _match(start_time=datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc))
    calculator.calculate(match)

    kwargs = calculator.fixture_repo.find_before_date_by_league_id.call_args.kwargs
    assert kwargs["before_date"] == date(2024, 6, 15)
    assert kwargs["limit"] == 500
    global_kwargs = calculator.fixture_repo.get_filtered.call_args.kwargs
    assert global_kwargs["before_date"] == date(2024, 6, 15)
    assert global_kwargs["limit"] == 500


def test_promoted_team_effect_fallback_when_unavailable():
    calculator = _calculator(
        league_matches=[
            _historical(match_date=date(2024, 6, 1), home_goals=1, away_goals=1)
        ],
        shrinkage=0,
    )
    features = calculator.calculate(_match())
    assert features.league_promoted_team_effect == 0.0


def test_promotion_effect_one_observation_per_promoted_vs_established_match():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_team="PromotedA",
            away_team="Established",
            home_goals=2,
            away_goals=0,
            home_promoted=True,
            away_promoted=False,
        ),
        _historical(
            match_date=date(2024, 6, 2),
            home_team="Established",
            away_team="PromotedB",
            home_goals=2,
            away_goals=1,
            home_promoted=False,
            away_promoted=True,
        ),
        # Both promoted — excluded from non-overlapping sample
        _historical(
            match_date=date(2024, 6, 3),
            home_team="PromotedA",
            away_team="PromotedB",
            home_goals=3,
            away_goals=0,
            home_promoted=True,
            away_promoted=True,
        ),
    ]
    calculator = _calculator(league_matches=league_matches, shrinkage=0)
    stats = calculator._compute_raw_stats(league_matches)

    assert stats.promotion_sample_size == 2
    # +2 (home promoted) and -1 (away promoted) → mean 0.5
    assert stats.promoted_team_effect == pytest.approx(0.5)
    features = calculator.calculate(_match())
    assert features.league_promoted_team_effect == pytest.approx(0.5)


def test_promotion_effect_strongly_shrunk_for_tiny_samples():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1),
            home_team="Promoted",
            away_team="Established",
            home_goals=4,
            away_goals=0,
            home_promoted=True,
            away_promoted=False,
        )
    ]
    global_matches = [
        _historical(
            match_date=date(2024, 5, day),
            home_team="P",
            away_team="E",
            home_goals=1,
            away_goals=1,
            home_promoted=True,
            away_promoted=False,
        )
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=100,
    )
    features = calculator.calculate(_match())

    # League effect = 4, global effect = 0, n=1, k=100
    assert features.league_promoted_team_effect == pytest.approx(
        (1 / 101) * 4.0 + (100 / 101) * 0.0
    )


def test_promotion_effect_approaches_league_with_large_samples():
    league_matches = [
        _historical(
            match_date=date(2024, 6, 1) - timedelta(days=day),
            home_team="Promoted",
            away_team=f"Est{day}",
            home_goals=2,
            away_goals=0,
            home_promoted=True,
            away_promoted=False,
        )
        for day in range(1, 51)
    ]
    global_matches = [
        _historical(
            match_date=date(2024, 1, day),
            home_team="P",
            away_team="E",
            home_goals=0,
            away_goals=0,
            home_promoted=True,
            away_promoted=False,
        )
        for day in range(1, 21)
    ]
    calculator = _calculator(
        league_matches=league_matches,
        global_matches=global_matches,
        shrinkage=10,
    )
    features = calculator.calculate(_match())

    assert features.league_promoted_team_effect == pytest.approx(
        (50 / 60) * 2.0 + (10 / 60) * 0.0
    )


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
    assert calculator.fixture_repo.find_before_date_by_league_id.call_count == 1


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

    assert (
        balanced_features.league_competitive_balance
        > unbalanced_features.league_competitive_balance
    )
