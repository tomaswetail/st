"""Unit tests for HomeAdvantageCalculator competition-type home advantage."""

from __future__ import annotations

from datetime import date
from math import log
from unittest.mock import MagicMock

import pytest

from calc.home_advantage_calculator import HomeAdvantageCalculator
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.team import Team
from utils.competition_type import competition_type_flags, is_league_match


def _calculator_with_goal_sums(
    goal_sums: dict[str, tuple[int, int, int, int]],
    *,
    shrinkage_matches: int = 0,
    max_competition_home_advantage: float = 0.30,
) -> HomeAdvantageCalculator:
    session = MagicMock()
    config = DataSourceConfig(
        competition_ha_shrinkage_matches=shrinkage_matches,
        max_competition_home_advantage=max_competition_home_advantage,
    )
    calculator = HomeAdvantageCalculator(session, config=config)
    calculator.historical_match_repo.get_goal_sums_by_league_before_date = MagicMock(
        return_value=goal_sums
    )
    return calculator


def _league_reference_ha(
    goal_sums: dict[str, tuple[int, int, int, int]],
) -> float:
    sum_home = sum_away = home_count = away_count = 0
    for league_code, totals in goal_sums.items():
        if not is_league_match(league_code):
            continue
        sum_home += totals[0]
        home_count += totals[1]
        sum_away += totals[2]
        away_count += totals[3]
    return log((sum_home / home_count) / (sum_away / away_count))


def test_competition_type_flags():
    assert competition_type_flags("E0") == (False, False, False)
    assert competition_type_flags("ENG-FA Cup") == (True, False, False)
    assert competition_type_flags("WC2022") == (False, True, False)
    assert competition_type_flags("INT-FRIENDLY") == (False, False, True)
    assert is_league_match("E0") is True
    assert is_league_match("ENG-FA Cup") is False


def test_league_match_has_zero_competition_home_advantage():
    goal_sums = {
        "E0": (150, 100, 100, 100),
        "ENG-FA Cup": (40, 20, 20, 20),
    }
    calculator = _calculator_with_goal_sums(goal_sums)
    result = calculator._calc_competition_home_advantage("E0", date(2024, 1, 1))

    assert result["competition_home_advantage"] == pytest.approx(0.0)
    assert result["raw_competition_home_advantage"] == pytest.approx(0.0)
    assert result["competition_home_advantage_sample_size"] == 0


def test_domestic_cup_competition_home_advantage():
    goal_sums = {
        "E0": (150, 100, 100, 100),
        "ENG-FA Cup": (40, 20, 20, 20),
    }
    calculator = _calculator_with_goal_sums(goal_sums, shrinkage_matches=0)
    result = calculator._calc_competition_home_advantage(
        "ENG-FA Cup", date(2024, 1, 1)
    )

    league_reference = _league_reference_ha(goal_sums)
    expected_raw = log(2.0) - league_reference
    assert result["raw_competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage"] > 0.0
    assert result["competition_home_advantage_sample_size"] == 20


def test_international_cup_competition_home_advantage():
    goal_sums = {
        "E0": (120, 100, 100, 100),
        "WC2022": (30, 15, 10, 15),
    }
    calculator = _calculator_with_goal_sums(
        goal_sums,
        shrinkage_matches=0,
        max_competition_home_advantage=1.0,
    )
    result = calculator._calc_competition_home_advantage("WC2022", date(2024, 1, 1))

    league_reference = _league_reference_ha(goal_sums)
    expected_raw = log((30 / 15) / (10 / 15)) - league_reference
    assert result["raw_competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage_sample_size"] == 15


def test_friendly_competition_home_advantage():
    goal_sums = {
        "E0": (120, 100, 100, 100),
        "INT-FRIENDLY": (25, 10, 15, 10),
    }
    calculator = _calculator_with_goal_sums(
        goal_sums,
        shrinkage_matches=0,
        max_competition_home_advantage=1.0,
    )
    result = calculator._calc_competition_home_advantage(
        "INT-FRIENDLY", date(2024, 1, 1)
    )

    league_reference = _league_reference_ha(goal_sums)
    expected_raw = log(2.5 / 1.5) - league_reference
    assert result["raw_competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage_sample_size"] == 10


def test_low_sample_applies_strong_shrinkage():
    goal_sums = {
        "E0": (150, 100, 100, 100),
        "ENG-FA Cup": (40, 5, 5, 5),
    }
    calculator = _calculator_with_goal_sums(goal_sums, shrinkage_matches=100)
    result = calculator._calc_competition_home_advantage(
        "ENG-FA Cup", date(2024, 1, 1)
    )

    league_reference = _league_reference_ha(goal_sums)
    expected_raw = log(8.0) - league_reference
    expected_weight = 5 / (5 + 100)
    assert result["raw_competition_home_advantage"] == pytest.approx(expected_raw)
    assert result["competition_home_advantage_shrinkage_weight"] == pytest.approx(
        expected_weight
    )
    assert result["competition_home_advantage"] == pytest.approx(
        expected_raw * expected_weight
    )
    assert result["competition_home_advantage_shrinkage_weight"] < 0.1


def test_future_matches_do_not_affect_result():
    before_date = date(2024, 6, 1)
    historical_only = {
        "E0": (100, 50, 80, 50),
        "ENG-FA Cup": (20, 10, 10, 10),
    }
    session = MagicMock()
    config = DataSourceConfig(competition_ha_shrinkage_matches=0)
    calculator = HomeAdvantageCalculator(session, config=config)

    def goal_sums_side_effect(cutoff: date):
        assert cutoff == before_date
        return historical_only

    calculator.historical_match_repo.get_goal_sums_by_league_before_date = MagicMock(
        side_effect=goal_sums_side_effect
    )

    first = calculator._calc_competition_home_advantage("ENG-FA Cup", before_date)
    second = calculator._calc_competition_home_advantage("ENG-FA Cup", before_date)

    assert first == second
    calculator.historical_match_repo.get_goal_sums_by_league_before_date.assert_called_with(
        before_date
    )


def test_competition_home_advantage_added_exactly_once_in_process():
    goal_sums = {
        "E0": (150, 100, 100, 100),
        "ENG-FA Cup": (40, 20, 20, 20),
    }
    calculator = _calculator_with_goal_sums(goal_sums, shrinkage_matches=0)
    calculator.calc_league_season_home_advantage = MagicMock(
        return_value={
            "league_season_home_advantage": 0.12,
            "raw_league_season_home_advantage": 0.15,
            "league_home_advantage_shrinkage_weight": 0.8,
            "league_home_advantage_sample_size": 100,
        }
    )
    calculator.calculate_team_home_advantage = MagicMock(
        return_value=calculator._empty_result(
            0.12,
            competition_home_advantage=0.0,
        )
    )
    calculator.calculate_team_home_advantage.return_value.team_home_advantage = 0.05
    calculator.calculate_team_home_advantage.return_value.home_advantage = 0.17

    team = Team(id=1, name="Arsenal", league_id=10)
    calculator._resolve_season_from_team_history = MagicMock(return_value="2023")

    result = calculator.process(
        team,
        date(2024, 1, 1),
        target_league_code="ENG-FA Cup",
    )

    competition_only = calculator._calc_competition_home_advantage(
        "ENG-FA Cup", date(2024, 1, 1)
    )
    assert result.competition_home_advantage == pytest.approx(
        competition_only["competition_home_advantage"]
    )
    assert result.home_advantage == pytest.approx(
        result.league_season_home_advantage
        + result.team_home_advantage
        + result.competition_home_advantage
    )
    assert calculator.calculate_team_home_advantage.return_value.home_advantage == (
        0.17
    )
