"""Tests for calc.balance_and_environment.BalanceAndEnvironment.

These tests deliberately mock:
- SQLAlchemy Session
- StrengthCalculator
- STMatchModel-like objects
- HistoricalMatch / HistoricalMatchModel-like rows

The goal is to test BalanceAndEnvironment's calculations and data-leakage
behaviour without requiring a real database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from calc.balance_and_environment import BalanceAndEnvironment


@dataclass(frozen=True)
class FakeTeam:
    name: str


@dataclass(frozen=True)
class FakeFixture:
    fixture_date: date
    home_team_name: str
    away_team_name: str
    goals_home: int
    goals_away: int
    result: str


def fixture(
    match_date: date,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> FakeFixture:
    """Build a HistoricalMatch-like row and derive its 1/X/2 result."""
    if home_goals > away_goals:
        result = "1"
    elif home_goals < away_goals:
        result = "2"
    else:
        result = "X"

    return FakeFixture(
        fixture_date=match_date,
        home_team_name=home_team,
        away_team_name=away_team,
        goals_home=home_goals,
        goals_away=away_goals,
        result=result,
    )


@pytest.fixture
def target_match():
    """STMatchModel-like object used as the prediction target."""
    return SimpleNamespace(
        id=999,
        home_team_id=101,
        away_team_id=202,
        home_team=FakeTeam("Alpha"),
        away_team=FakeTeam("Beta"),
        start_time=datetime(2026, 8, 10, 15, 0),
    )


@pytest.fixture
def config():
    """Only the BalanceAndEnvironment config values are needed here."""
    return SimpleNamespace(
        balance_recent_matches=10,
        balance_low_scoring_goal_threshold=2,
    )


@pytest.fixture
def strength_features():
    """Fixture features as returned by StrengthCalculator."""
    return SimpleNamespace(
        home_attack_strength=1.20,
        away_attack_strength=0.90,
        home_defence_strength=0.80,
        away_defence_strength=1.10,
        expected_home_goals=1.40,
        expected_away_goals=1.00,
    )


@pytest.fixture
def strength_calculator(strength_features):
    calculator = Mock(name="StrengthCalculator")
    calculator.get_fixture_features.return_value = strength_features
    return calculator


@pytest.fixture
def calculator(config, strength_calculator):
    return BalanceAndEnvironment(
        session=Mock(name="Session"),
        config=config,
        strength_calculator=strength_calculator,
    )


def test_calculate_uses_strength_calculator_and_computes_core_features(
    calculator,
    strength_calculator,
    target_match,
):
    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    strength_calculator.get_fixture_features.assert_called_once_with(
        101,
        202,
        target_match.start_time,
        match_id=999,
    )

    assert features.attack_strength_difference == pytest.approx(0.30)
    assert features.expected_goal_difference == pytest.approx(0.40)
    assert features.expected_goal_total == pytest.approx(2.40)
    assert features.market_balance == pytest.approx(0.22)
    assert features.defence_strength_difference == pytest.approx(0.30)

    # This is the CURRENT implementation contract:
    # favourite_strength = max(home_probability, away_probability).
    assert features.favourite_strength == pytest.approx(0.46)


def test_calculate_computes_all_recent_team_rates_and_combined_rates(
    calculator,
    target_match,
):
    history = [
        # Alpha: 4 matches
        # draw=2/4, one-goal=1/4, close=3/4, low-scoring=2/4
        fixture(date(2026, 8, 9), "Alpha", "A1", 1, 1),
        fixture(date(2026, 8, 8), "A2", "Alpha", 0, 1),
        fixture(date(2026, 8, 7), "Alpha", "A3", 3, 0),
        fixture(date(2026, 8, 6), "A4", "Alpha", 2, 2),

        # Beta: 4 matches
        # draw=1/4, one-goal=1/4, close=2/4, low-scoring=2/4
        fixture(date(2026, 8, 9), "B1", "Beta", 0, 0),
        fixture(date(2026, 8, 8), "Beta", "B2", 2, 1),
        fixture(date(2026, 8, 7), "B3", "Beta", 1, 3),
        fixture(date(2026, 8, 6), "Beta", "B4", 0, 2),
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_sample_size == 4
    assert features.away_recent_sample_size == 4

    assert features.home_recent_draw_rate == pytest.approx(0.50)
    assert features.home_one_goal_match_rate == pytest.approx(0.25)
    assert features.home_close_match_rate == pytest.approx(0.75)
    assert features.home_low_scoring_rate == pytest.approx(0.50)

    assert features.away_recent_draw_rate == pytest.approx(0.25)
    assert features.away_one_goal_match_rate == pytest.approx(0.25)
    assert features.away_close_match_rate == pytest.approx(0.50)
    assert features.away_low_scoring_rate == pytest.approx(0.50)

    assert features.combined_draw_rate == pytest.approx(0.375)
    assert features.combined_one_goal_match_rate == pytest.approx(0.25)
    assert features.combined_close_match_rate == pytest.approx(0.625)
    assert features.combined_low_scoring_rate == pytest.approx(0.50)


def test_target_date_and_future_matches_are_excluded(
    calculator,
    target_match,
):
    history = [
        # The only valid pre-target match: a draw, close and low-scoring.
        fixture(date(2026, 8, 9), "Alpha", "OldOpponent", 1, 1),

        # Same date as target: MUST be excluded by the strict < cutoff.
        # These rows would materially change every rate if leaked.
        fixture(date(2026, 8, 10), "Alpha", "TargetOpponent", 5, 0),

        # Future: MUST also be excluded.
        fixture(date(2026, 8, 11), "FutureOpponent", "Alpha", 4, 0),
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_sample_size == 1
    assert features.home_recent_draw_rate == pytest.approx(1.0)
    assert features.home_one_goal_match_rate == pytest.approx(0.0)
    assert features.home_close_match_rate == pytest.approx(1.0)
    assert features.home_low_scoring_rate == pytest.approx(1.0)


def test_recent_window_uses_only_most_recent_matches_before_target(
    target_match,
    strength_calculator,
):
    config = SimpleNamespace(
        balance_recent_matches=2,
        balance_low_scoring_goal_threshold=2,
    )
    calculator = BalanceAndEnvironment(
        session=Mock(name="Session"),
        config=config,
        strength_calculator=strength_calculator,
    )

    # Deliberately unsorted input. Only Aug 9 + Aug 8 should count.
    history = [
        fixture(date(2026, 8, 6), "Alpha", "A4", 1, 1),  # old draw
        fixture(date(2026, 8, 9), "Alpha", "A1", 0, 0),  # recent draw
        fixture(date(2026, 8, 7), "Alpha", "A3", 4, 0),  # old blowout
        fixture(date(2026, 8, 8), "A2", "Alpha", 0, 1),  # recent 1-goal
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_sample_size == 2
    assert features.home_recent_draw_rate == pytest.approx(0.50)
    assert features.home_one_goal_match_rate == pytest.approx(0.50)
    assert features.home_close_match_rate == pytest.approx(1.0)
    assert features.home_low_scoring_rate == pytest.approx(1.0)


def test_low_scoring_threshold_is_configurable(
    target_match,
    strength_calculator,
):
    config = SimpleNamespace(
        balance_recent_matches=10,
        balance_low_scoring_goal_threshold=1,
    )
    calculator = BalanceAndEnvironment(
        session=Mock(name="Session"),
        config=config,
        strength_calculator=strength_calculator,
    )

    history = [
        fixture(date(2026, 8, 9), "Alpha", "A1", 1, 1),  # total=2: not low at threshold=1
        fixture(date(2026, 8, 8), "Alpha", "A2", 1, 0),  # total=1: low
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_low_scoring_rate == pytest.approx(0.50)


def test_no_history_is_handled_gracefully(
    calculator,
    target_match,
):
    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_sample_size == 0
    assert features.away_recent_sample_size == 0

    assert features.home_recent_draw_rate is None
    assert features.away_recent_draw_rate is None
    assert features.home_one_goal_match_rate is None
    assert features.away_one_goal_match_rate is None
    assert features.home_close_match_rate is None
    assert features.away_close_match_rate is None
    assert features.home_low_scoring_rate is None
    assert features.away_low_scoring_rate is None

    assert features.combined_draw_rate is None
    assert features.combined_one_goal_match_rate is None
    assert features.combined_close_match_rate is None
    assert features.combined_low_scoring_rate is None


def test_combined_rate_falls_back_to_team_with_available_history(
    calculator,
    target_match,
):
    history = [
        fixture(date(2026, 8, 9), "Alpha", "A1", 1, 1),
        fixture(date(2026, 8, 8), "Alpha", "A2", 1, 0),
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_draw_rate == pytest.approx(0.50)
    assert features.away_recent_draw_rate is None
    assert features.combined_draw_rate == pytest.approx(0.50)

    assert features.home_close_match_rate == pytest.approx(1.0)
    assert features.away_close_match_rate is None
    assert features.combined_close_match_rate == pytest.approx(1.0)


def test_market_percentages_are_normalized_to_unit_probabilities(
    calculator,
    target_match,
):
    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 50.0, "X": 30.0, "2": 20.0},
    )

    assert features.market_balance == pytest.approx(0.30)
    assert features.favourite_strength == pytest.approx(0.50)


def test_missing_strength_values_propagate_as_none(
    config,
    target_match,
):
    strength_calculator = Mock(name="StrengthCalculator")
    strength_calculator.get_fixture_features.return_value = SimpleNamespace(
        home_attack_strength=None,
        away_attack_strength=0.90,
        home_defence_strength=0.80,
        away_defence_strength=None,
        expected_home_goals=None,
        expected_away_goals=1.00,
    )

    calculator = BalanceAndEnvironment(
        session=Mock(name="Session"),
        config=config,
        strength_calculator=strength_calculator,
    )

    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.attack_strength_difference is None
    assert features.defence_strength_difference is None
    assert features.expected_goal_difference is None
    assert features.expected_goal_total is None


@pytest.mark.parametrize(
    ("home_team", "away_team"),
    [
        (None, FakeTeam("Beta")),
        (FakeTeam("Alpha"), None),
    ],
)
def test_missing_team_raises_value_error(
    calculator,
    target_match,
    home_team,
    away_team,
):
    broken_match = SimpleNamespace(
        **{
            **vars(target_match),
            "home_team": home_team,
            "away_team": away_team,
        }
    )

    with pytest.raises(ValueError, match="Missing team"):
        calculator.calculate(
            match=broken_match,
            fixtures=[],
            market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
        )


def test_missing_start_time_raises_value_error(
    calculator,
    target_match,
):
    broken_match = SimpleNamespace(
        **{
            **vars(target_match),
            "start_time": None,
        }
    )

    with pytest.raises(ValueError, match="Missing start_time"):
        calculator.calculate(
            match=broken_match,
            fixtures=[],
            market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
        )


def test_result_d_or_x_is_recognized_as_draw_even_if_result_encoding_varies(
    calculator,
    target_match,
):
    # Explicitly test the implementation's support for both "D" and "X".
    history = [
        FakeFixture(
            fixture_date=date(2026, 8, 9),
            home_team_name="Alpha",
            away_team_name="A1",
            goals_home=1,
            goals_away=1,
            result="D",
        ),
        FakeFixture(
            fixture_date=date(2026, 8, 8),
            home_team_name="Alpha",
            away_team_name="A2",
            goals_home=2,
            goals_away=2,
            result="x",
        ),
    ]

    features = calculator.calculate(
        match=target_match,
        fixtures=history,
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.home_recent_draw_rate == pytest.approx(1.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Original prompt describes favourite strength as distance of the strongest "
        "team from 50%; current implementation returns max(p_home, p_away)."
    ),
)
def test_prompt_interpretation_favourite_strength_is_distance_from_50_percent(
    calculator,
    target_match,
):
    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    # If the prompt's example is intended literally, this should be 0.04.
    assert features.favourite_strength == pytest.approx(0.04)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Original prompt names the core feature strength_difference; current "
        "implementation exposes attack_strength_difference instead."
    ),
)
def test_prompt_contract_exposes_strength_difference(
    calculator,
    target_match,
):
    features = calculator.calculate(
        match=target_match,
        fixtures=[],
        market_probabilities={"1": 0.46, "X": 0.30, "2": 0.24},
    )

    assert features.strength_difference == pytest.approx(0.30)
