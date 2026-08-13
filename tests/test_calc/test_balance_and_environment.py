"""Unit tests for BalanceAndEnvironment feature calculator."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.balance_and_environment import BalanceAndEnvironment
from objects.schema.data_classes.balance_and_environment_features import (
    BalanceAndEnvironmentFeatures,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.team_strength_features import MatchStrengthFeatures
from utils.common import ensure_unit_probabilities


def _strength_features(
    *,
    home_attack: float | None = 1.2,
    away_attack: float | None = 0.9,
    home_defence: float | None = 1.0,
    away_defence: float | None = 1.1,
    home_xg: float | None = 1.5,
    away_xg: float | None = 1.1,
) -> MatchStrengthFeatures:
    return MatchStrengthFeatures(
        home_team_id=1,
        away_team_id=2,
        home=None,
        away=None,
        match_id=10,
        home_attack_strength=home_attack,
        away_attack_strength=away_attack,
        home_defence_strength=home_defence,
        away_defence_strength=away_defence,
        expected_home_goals=home_xg,
        expected_away_goals=away_xg,
    )


def _match(
    *,
    start_time: datetime = datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        home_team_id=1,
        away_team_id=2,
        start_time=start_time,
        home_team=SimpleNamespace(name="Arsenal"),
        away_team=SimpleNamespace(name="Chelsea"),
    )


def _historical(
    *,
    match_date: date,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    result: str | None = None,
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
    )


def _calculator(
    strength: MatchStrengthFeatures,
    *,
    recent_matches: int = 10,
    low_scoring_threshold: int = 2,
) -> BalanceAndEnvironment:
    session = MagicMock()
    config = DataSourceConfig(
        balance_recent_matches=recent_matches,
        balance_low_scoring_goal_threshold=low_scoring_threshold,
    )
    strength_calculator = MagicMock()
    strength_calculator.get_fixture_features.return_value = strength
    return BalanceAndEnvironment(
        session,
        config=config,
        strength_calculator=strength_calculator,
    )


def test_core_strength_and_xg_features():
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": 0.45, "X": 0.25, "2": 0.30},
    )

    assert features.attack_strength_difference == pytest.approx(0.3)
    assert features.expected_goal_difference == pytest.approx(0.4)
    assert features.expected_goal_total == pytest.approx(2.6)
    assert features.defence_strength_difference == pytest.approx(0.1)


def test_attack_strength_difference_uses_fixture_attack_strengths():
    calculator = _calculator(
        _strength_features(home_attack=1.4, away_attack=0.8, home_defence=1.9, away_defence=0.2)
    )
    features = calculator.calculate(
        _match(),
        [],
        {"1": 0.45, "X": 0.25, "2": 0.30},
    )

    assert features.attack_strength_difference == pytest.approx(0.6)
    assert features.defence_strength_difference == pytest.approx(1.7)


def test_market_balance_and_favourite_strength():
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": 0.55, "X": 0.25, "2": 0.20},
    )

    assert features.market_balance == pytest.approx(0.35)
    assert features.favourite_strength == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("p_home", "p_away", "expected_favourite"),
    [
        (0.40, 0.30, 0.40),
        (0.55, 0.20, 0.55),
        (0.20, 0.75, 0.75),
    ],
)
def test_favourite_strength_is_strongest_team_probability(
    p_home: float, p_away: float, expected_favourite: float
):
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": p_home, "X": 1.0 - p_home - p_away, "2": p_away},
    )

    assert features.favourite_strength == pytest.approx(expected_favourite)


def test_unit_scale_market_inputs_are_used_as_is():
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": 0.45, "X": 0.25, "2": 0.30},
    )

    assert features.market_balance == pytest.approx(0.15)
    assert features.favourite_strength == pytest.approx(0.45)


def test_percentage_market_inputs_are_normalized_to_unit_scale():
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": 55, "X": 25, "2": 20},
    )

    assert features.market_balance == pytest.approx(0.35)
    assert features.favourite_strength == pytest.approx(0.55)


def test_mixed_market_probability_scales_are_rejected():
    calculator = _calculator(_strength_features())
    with pytest.raises(ValueError, match="Mixed"):
        calculator.calculate(
            _match(),
            [],
            {"1": 0.55, "X": 25, "2": 20},
        )


def test_ensure_unit_probabilities_keeps_0_45():
    assert ensure_unit_probabilities({"1": 0.45, "X": 0.25, "2": 0.30}) == {
        "1": 0.45,
        "X": 0.25,
        "2": 0.30,
    }


def test_ensure_unit_probabilities_converts_45_to_0_45():
    normalized = ensure_unit_probabilities({"1": 45, "X": 25, "2": 30})
    assert normalized["1"] == pytest.approx(0.45)
    assert normalized["X"] == pytest.approx(0.25)
    assert normalized["2"] == pytest.approx(0.30)


def test_ensure_unit_probabilities_rejects_mixed_and_invalid_scales():
    with pytest.raises(ValueError, match="Mixed"):
        ensure_unit_probabilities({"1": 0.45, "X": 25, "2": 30})
    with pytest.raises(ValueError, match="non-negative"):
        ensure_unit_probabilities({"1": -0.1, "X": 0.5, "2": 0.6})
    with pytest.raises(ValueError, match="<= 100"):
        ensure_unit_probabilities({"1": 145, "X": 25, "2": 20})


def test_missing_strength_and_xg_yield_none():
    calculator = _calculator(
        _strength_features(
            home_attack=None,
            away_attack=1.0,
            home_defence=None,
            away_defence=1.0,
            home_xg=None,
            away_xg=1.0,
        )
    )
    features = calculator.calculate(_match(), [], {"1": 0.4, "X": 0.3, "2": 0.3})

    assert features.attack_strength_difference is None
    assert features.defence_strength_difference is None
    assert features.expected_goal_difference is None
    assert features.expected_goal_total is None


def test_empty_history_rates_are_none():
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        [],
        {"1": 0.4, "X": 0.3, "2": 0.3},
    )

    assert features.home_recent_draw_rate is None
    assert features.away_recent_draw_rate is None
    assert features.combined_draw_rate is None
    assert features.home_recent_sample_size == 0
    assert features.away_recent_sample_size == 0


def test_recent_draw_one_goal_close_and_low_scoring_rates():
    history = [
        # Arsenal: draw 1-1 (close, low-scoring, not one-goal)
        _historical(
            match_date=date(2024, 6, 1),
            home_team="Arsenal",
            away_team="A",
            home_goals=1,
            away_goals=1,
        ),
        # Arsenal: win 2-1 (close, one-goal, not low-scoring)
        _historical(
            match_date=date(2024, 5, 20),
            home_team="B",
            away_team="Arsenal",
            home_goals=1,
            away_goals=2,
        ),
        # Chelsea: loss 0-3 (not close)
        _historical(
            match_date=date(2024, 6, 2),
            home_team="Chelsea",
            away_team="C",
            home_goals=0,
            away_goals=3,
        ),
        # Chelsea: draw 0-0 (close, low-scoring, not one-goal)
        _historical(
            match_date=date(2024, 5, 10),
            home_team="D",
            away_team="Chelsea",
            home_goals=0,
            away_goals=0,
        ),
    ]
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        history,
        {"1": 0.4, "X": 0.3, "2": 0.3},
    )

    assert features.home_recent_sample_size == 2
    assert features.away_recent_sample_size == 2
    assert features.home_recent_draw_rate == pytest.approx(0.5)
    assert features.away_recent_draw_rate == pytest.approx(0.5)
    assert features.home_one_goal_match_rate == pytest.approx(0.5)
    assert features.away_one_goal_match_rate == pytest.approx(0.0)
    assert features.home_close_match_rate == pytest.approx(1.0)
    assert features.away_close_match_rate == pytest.approx(0.5)
    assert features.home_low_scoring_rate == pytest.approx(0.5)
    assert features.away_low_scoring_rate == pytest.approx(0.5)
    assert features.combined_draw_rate == pytest.approx(0.5)
    assert features.combined_one_goal_match_rate == pytest.approx(0.25)
    assert features.combined_close_match_rate == pytest.approx(0.75)
    assert features.combined_low_scoring_rate == pytest.approx(0.5)


def test_future_and_target_date_matches_are_excluded():
    cutoff = date(2024, 6, 15)
    history = [
        _historical(
            match_date=date(2024, 6, 1),
            home_team="Arsenal",
            away_team="A",
            home_goals=1,
            away_goals=1,
        ),
        # Same day as target — must be excluded
        _historical(
            match_date=cutoff,
            home_team="Arsenal",
            away_team="B",
            home_goals=5,
            away_goals=0,
        ),
        # Future — must be excluded
        _historical(
            match_date=date(2024, 6, 20),
            home_team="Arsenal",
            away_team="C",
            home_goals=4,
            away_goals=0,
        ),
    ]
    calculator = _calculator(_strength_features())
    features = calculator.calculate(
        _match(),
        history,
        {"1": 0.4, "X": 0.3, "2": 0.3},
    )

    assert features.home_recent_sample_size == 1
    assert features.home_recent_draw_rate == pytest.approx(1.0)
    assert features.home_one_goal_match_rate == pytest.approx(0.0)


def test_recent_window_limits_to_configured_matches():
    history = [
        _historical(
            match_date=date(2024, 6, day),
            home_team="Arsenal",
            away_team=f"Opp{day}",
            home_goals=1,
            away_goals=0,
        )
        for day in range(1, 8)
    ]
    calculator = _calculator(_strength_features(), recent_matches=3)
    features = calculator.calculate(
        _match(),
        history,
        {"1": 0.4, "X": 0.3, "2": 0.3},
    )

    assert features.home_recent_sample_size == 3


def test_calculate_is_deterministic():
    history = [
        _historical(
            match_date=date(2024, 6, 1),
            home_team="Arsenal",
            away_team="A",
            home_goals=2,
            away_goals=1,
        ),
        _historical(
            match_date=date(2024, 5, 28),
            home_team="Chelsea",
            away_team="B",
            home_goals=0,
            away_goals=0,
        ),
    ]
    calculator = _calculator(_strength_features())
    market = {"1": 0.42, "X": 0.28, "2": 0.30}
    first = calculator.calculate(_match(), history, market)
    second = calculator.calculate(_match(), history, market)

    assert first == second
    assert isinstance(first, BalanceAndEnvironmentFeatures)


def test_uses_strength_calculator_fixture_features():
    calculator = _calculator(_strength_features())
    match = _match()
    calculator.calculate(match, [], {"1": 0.4, "X": 0.3, "2": 0.3})

    calculator.strength_calculator.get_fixture_features.assert_called_once_with(
        1,
        2,
        match.start_time,
        match_id=10,
    )
