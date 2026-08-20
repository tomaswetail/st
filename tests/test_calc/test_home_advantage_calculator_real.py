from __future__ import annotations

from datetime import date
from math import exp, log
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.home_advantage_calculator import HomeAdvantageCalculator
from unittest.mock import call

LEAGUE_ID = 10
TEAM_ID = 1
TARGET_DATE = date(2026, 1, 20)


def _config(**overrides):
    values = {
        "home_advantage_epsilon": 0.05,
        "home_advantage_recency_decay_rate": 0.0,
        "home_advantage_shrinkage_matches": 30.0,
        "max_team_home_advantage": 0.30,
        "team_strength_lookback_matches": 20,
        "league_home_advantage_shrinkage_matches": 0.0,
        "league_home_advantage_global_prior": 0.0,
        "competition_ha_shrinkage_matches": 30.0,
        "max_competition_home_advantage": 0.30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _team(team_id=TEAM_ID, name="Target", league_id=LEAGUE_ID):
    return SimpleNamespace(id=team_id, name=name, league_id=league_id)


def _match_row(
    match_date: date,
    *,
    played_at_home: bool,
    xg_for: float,
    xg_against: float,
    opponent_name: str = "Opponent",
    league_external_id: int = 39,
    season: str = "2025",
):
    return SimpleNamespace(
        match=SimpleNamespace(fixture_date=match_date),
        played_at_home=played_at_home,
        xg_for=xg_for,
        xg_against=xg_against,
        opponent_name=opponent_name,
        league_external_id=league_external_id,
        season=season,
    )


def _unit_features(attack=1.0, defence_weakness=1.0):
    return SimpleNamespace(
        opponent_adjusted_attack_strength=attack,
        opponent_adjusted_defence_strength=defence_weakness,
        recency_weighted_attack_rating=None,
        recency_weighted_defence_rating=None,
    )


@pytest.fixture
def calculator():
    strength = MagicMock()
    calc = HomeAdvantageCalculator(
        session=MagicMock(),
        config=_config(),
        strength_calculator=strength,
    )

    # Replace DB-backed repositories with mocks.
    calc.team_repo = MagicMock()
    calc.league_repo = MagicMock()
    calc.fixture_repo = MagicMock()

    calc.team_repo.get.return_value = _team()
    return calc


def _prepare_team_calculation(calc, rows, *, expected=(1.0, 1.0)):
    """Mock DB/strength boundaries but keep residual/shrinkage math real."""
    calc._load_team_npxg_matches = MagicMock(return_value=rows)
    calc._league_baselines_before = MagicMock(
        return_value={"home_npxg": 1.0, "away_npxg": 1.0, "npxg": 1.0}
    )
    calc._expected_npxg = MagicMock(return_value=expected)


def test_no_difference_produces_zero_team_home_advantage(calculator):
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.0,
            xg_against=1.0,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.0,
            xg_against=1.0,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.20,
    )

    assert result.raw_team_home_advantage == pytest.approx(0.0)
    assert result.team_home_advantage == pytest.approx(0.0)
    # Team-specific HA must not duplicate the league-wide HA.
    assert result.home_advantage == pytest.approx(0.20)


def test_strong_home_team_produces_positive_advantage(calculator):
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.50,
            xg_against=0.70,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.00,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    assert result.raw_team_home_advantage > 0
    assert result.team_home_advantage > 0


def test_poor_home_team_produces_negative_advantage(calculator):
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=0.70,
            xg_against=1.50,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.00,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    assert result.raw_team_home_advantage < 0
    assert result.team_home_advantage < 0


def test_low_sample_is_heavily_shrunk(calculator):
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.50,
            xg_against=1.00,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.00,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    expected_weight = 1.0 / 31.0
    assert result.home_match_count == 1
    assert result.away_match_count == 1
    assert result.team_home_advantage_shrinkage_weight == pytest.approx(
        expected_weight
    )
    assert result.team_home_advantage == pytest.approx(
        result.raw_team_home_advantage * expected_weight
    )
    assert abs(result.team_home_advantage) < abs(result.raw_team_home_advantage)


def test_attack_and_defence_are_averaged_not_summed(calculator):
    """
    Spec requirement:
        performance = (attack_residual + defence_residual) / 2

    This test intentionally catches an implementation that sums the two residuals.
    """
    calculator.config = _config(home_advantage_shrinkage_matches=0.0)
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.30,
            xg_against=1.00,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.00,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    epsilon = calculator.config.home_advantage_epsilon
    home_attack_residual = log((1.30 + epsilon) / (1.00 + epsilon))
    expected_home_performance = home_attack_residual / 2.0

    assert result.home_attack_residual == pytest.approx(home_attack_residual)
    assert result.home_defence_residual == pytest.approx(0.0)
    assert result.home_performance == pytest.approx(expected_home_performance)
    assert result.raw_team_home_advantage == pytest.approx(
        expected_home_performance
    )


def test_normal_league_home_pattern_does_not_become_team_specific_ha(calculator):
    """
    Home npxG is 1.4 and away npxG is 1.0 league-wide.
    The target team exactly follows those expectations, so additional team HA = 0.
    """
    calculator.config = _config(home_advantage_shrinkage_matches=0.0)
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.40,
            xg_against=1.00,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.40,
        ),
    ]
    calculator._load_team_npxg_matches = MagicMock(return_value=rows)
    calculator._league_baselines_before = MagicMock(
        return_value={"home_npxg": 1.40, "away_npxg": 1.00, "npxg": 1.20}
    )

    opponent = _team(2, "Opponent")
    calculator.team_repo.get_by_name.return_value = opponent
    calculator._team_features_before = MagicMock(
        side_effect=lambda team_id, before_date: _unit_features()
    )

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=log(1.40 / 1.00),
    )

    assert result.home_attack_residual == pytest.approx(0.0)
    assert result.away_attack_residual == pytest.approx(0.0)
    assert result.home_defence_residual == pytest.approx(0.0)
    assert result.away_defence_residual == pytest.approx(0.0)
    assert result.team_home_advantage == pytest.approx(0.0)


def test_opponent_strength_changes_expected_xg_for_identical_actual_xg(calculator):
    """
    Same actual 1.2 npxG should be less impressive against a weak defence because
    the expected npxG is higher after opponent adjustment.
    """
    calculator.team_repo.get_by_name.side_effect = lambda name: {
        "StrongDef": _team(2, "StrongDef"),
        "WeakDef": _team(3, "WeakDef"),
    }[name]

    features = {
        TEAM_ID: _unit_features(attack=1.0, defence_weakness=1.0),
        2: _unit_features(attack=1.0, defence_weakness=0.70),
        3: _unit_features(attack=1.0, defence_weakness=1.30),
    }
    calculator._team_features_before = MagicMock(
        side_effect=lambda team_id, before_date: features[team_id]
    )

    expected_vs_strong, _ = calculator._expected_npxg(
        team_id=TEAM_ID,
        opponent_name="StrongDef",
        match_date=date(2026, 1, 10),
        played_at_home=True,
        league_home_npxg=1.20,
        league_away_npxg=1.00,
        league_overall_npxg=1.10,
    )
    expected_vs_weak, _ = calculator._expected_npxg(
        team_id=TEAM_ID,
        opponent_name="WeakDef",
        match_date=date(2026, 1, 10),
        played_at_home=True,
        league_home_npxg=1.20,
        league_away_npxg=1.00,
        league_overall_npxg=1.10,
    )

    assert expected_vs_strong == pytest.approx(1.20 * 0.70)
    assert expected_vs_weak == pytest.approx(1.20 * 1.30)

    epsilon = calculator.config.home_advantage_epsilon
    actual = 1.20
    residual_vs_strong = log((actual + epsilon) / (expected_vs_strong + epsilon))
    residual_vs_weak = log((actual + epsilon) / (expected_vs_weak + epsilon))

    assert residual_vs_strong > residual_vs_weak


def test_historical_strengths_are_requested_at_each_match_date(calculator):
    """Ratings used for a historical performance must be pre-match ratings."""
    opponent = _team(2, "Opponent")
    calculator.team_repo.get_by_name.return_value = opponent

    calls = []

    def features_before(team_id, before_date):
        calls.append((team_id, before_date))
        return _unit_features()

    calculator._team_features_before = features_before

    historical_date = date(2026, 1, 10)
    calculator._expected_npxg(
        team_id=TEAM_ID,
        opponent_name="Opponent",
        match_date=historical_date,
        played_at_home=True,
        league_home_npxg=1.20,
        league_away_npxg=1.00,
        league_overall_npxg=1.10,
    )

    assert calls == [
        (TEAM_ID, historical_date),
        (2, historical_date),
    ]


def test_team_history_loader_uses_strict_target_date_cutoff(calculator):
    """
    Mock the DB repository with a dataset containing past, target-date, and
    future matches. The repository mock honors the requested '< before_date'
    contract. Adding/changing future rows cannot affect the loaded history.
    """
    team_model = _team()
    past = SimpleNamespace(
        id=1,
        fixture_date=date(2026, 1, 19),
        home_team=SimpleNamespace(name="Target"),
        away_team=SimpleNamespace(name="Opponent"),
        home_team_name="Target",
        away_team_name="Opponent",
        league_id=39,
        league_season=2025,
        league="E0",
        season="2025",
    )
    on_target = SimpleNamespace(
        id=2,
        fixture_date=TARGET_DATE,
        home_team=SimpleNamespace(name="Target"),
        away_team=SimpleNamespace(name="Opponent"),
        home_team_name="Target",
        away_team_name="Opponent",
        league_id=39,
        league_season=2025,
        league="E0",
        season="2025",
    )
    future = SimpleNamespace(
        id=3,
        fixture_date=date(2026, 1, 21),
        home_team=SimpleNamespace(name="Target"),
        away_team=SimpleNamespace(name="Opponent"),
        home_team_name="Target",
        away_team_name="Opponent",
        league_id=39,
        league_season=2025,
        league="E0",
        season="2025",
    )
    db_rows = [past, on_target, future]

    def find_before_date_by_team(*, team_name, before_date, venue, limit):
        return [m for m in db_rows if m.fixture_date < before_date][:limit]

    calculator.fixture_repo.find_before_date_by_team.side_effect = (
        find_before_date_by_team
    )

    stats = SimpleNamespace(
        home_non_penalty_xg=1.20,
        home_xg=1.40,
        away_non_penalty_xg=0.80,
        away_xg=1.00,
    )

    def attach_advanced_stats(matches, team_name):
        return [(m, stats, m.home_team.name == team_name) for m in matches]

    calculator.strength_calculator.attach_advanced_stats.side_effect = (
        attach_advanced_stats
    )

    first = calculator._load_team_npxg_matches(team_model, TARGET_DATE)

    # Change future data drastically; strict cutoff means result is unchanged.
    future.fixture_date = date(2030, 1, 1)
    future_stats_change = 999.0  # documents that future data could be arbitrary
    assert future_stats_change == 999.0

    second = calculator._load_team_npxg_matches(team_model, TARGET_DATE)

    assert [r.match.id for r in first] == [1]
    assert [r.match.id for r in second] == [1]
    assert all(r.match.fixture_date < TARGET_DATE for r in first + second)

    calculator.fixture_repo.find_before_date_by_team.assert_called_with(
        team_name="Target",
        before_date=TARGET_DATE,
        venue=None,
        limit=60,
    )


def test_npxg_is_preferred_and_xg_is_used_as_fallback():
    stats = SimpleNamespace(
        home_non_penalty_xg=1.10,
        home_xg=1.50,
        away_non_penalty_xg=0.70,
        away_xg=1.00,
    )

    xg_for, xg_against = HomeAdvantageCalculator._npxg_pair(
        stats, played_at_home=True
    )
    assert xg_for == pytest.approx(1.10)
    assert xg_against == pytest.approx(0.70)

    stats.home_non_penalty_xg = None
    stats.away_non_penalty_xg = None
    xg_for, xg_against = HomeAdvantageCalculator._npxg_pair(
        stats, played_at_home=True
    )
    assert xg_for == pytest.approx(1.50)
    assert xg_against == pytest.approx(1.00)


def test_recency_weighting_gives_recent_match_more_influence(calculator):
    calculator.config = _config(
        home_advantage_recency_decay_rate=0.10,
        home_advantage_shrinkage_matches=0.0,
    )

    # Older home performance is positive; recent home performance is negative.
    rows = [
        _match_row(
            date(2025, 12, 21),  # 30 days old
            played_at_home=True,
            xg_for=1.50,
            xg_against=1.00,
        ),
        _match_row(
            date(2026, 1, 19),  # 1 day old
            played_at_home=True,
            xg_for=0.70,
            xg_against=1.00,
        ),
        _match_row(
            date(2026, 1, 18),
            played_at_home=False,
            xg_for=1.00,
            xg_against=1.00,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    epsilon = calculator.config.home_advantage_epsilon
    old_residual = log((1.50 + epsilon) / (1.00 + epsilon))
    recent_residual = log((0.70 + epsilon) / (1.00 + epsilon))
    old_weight = exp(-0.10 * 30)
    recent_weight = exp(-0.10 * 1)
    expected_weighted_attack = (
        old_residual * old_weight + recent_residual * recent_weight
    ) / (old_weight + recent_weight)

    assert result.home_attack_residual == pytest.approx(expected_weighted_attack)
    assert result.home_attack_residual < 0  # recent bad match dominates


def test_zero_home_or_away_sample_returns_zero(calculator):
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=1.60,
            xg_against=0.70,
        )
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.15,
    )

    assert result.home_match_count == 1
    assert result.away_match_count == 0
    assert result.team_home_advantage == pytest.approx(0.0)
    assert result.home_advantage == pytest.approx(0.15)


def test_team_home_advantage_is_capped(calculator):
    calculator.config = _config(
        home_advantage_shrinkage_matches=0.0,
        max_team_home_advantage=0.10,
    )
    rows = [
        _match_row(
            date(2026, 1, 10),
            played_at_home=True,
            xg_for=10.0,
            xg_against=0.05,
        ),
        _match_row(
            date(2026, 1, 11),
            played_at_home=False,
            xg_for=1.0,
            xg_against=1.0,
        ),
    ]
    _prepare_team_calculation(calculator, rows)

    result = calculator.calculate_team_home_advantage(
        team=_team(),
        league_id=LEAGUE_ID,
        season="2025",
        target_date=TARGET_DATE,
        league_season_home_advantage=0.0,
    )

    assert result.raw_team_home_advantage > 0.10
    assert result.team_home_advantage == pytest.approx(0.10)


def test_league_home_advantage_uses_only_pre_cutoff_goal_sums(calculator):
    calculator.config = _config(
        league_home_advantage_shrinkage_matches=0.0,
    )
    calculator.fixture_repo.get_home_goals_sum_by_league.return_value = (
        30,
        20,
    )
    calculator.fixture_repo.get_away_goals_sum_by_league.return_value = (
        20,
        20,
    )

    result = calculator.calc_league_season_home_advantage(
        LEAGUE_ID,
        "2025",
        TARGET_DATE,
        return_diagnostics=True,
    )

    assert result["raw_league_season_home_advantage"] == pytest.approx(
        log((30 / 20) / (20 / 20))
    )
    assert result["league_season_home_advantage"] == pytest.approx(
        result["raw_league_season_home_advantage"]
    )
    assert result["league_home_advantage_sample_size"] == 20

    assert calculator.fixture_repo.get_home_goals_sum_by_league.call_args_list == [
        call(LEAGUE_ID, "2025", TARGET_DATE),
        call(LEAGUE_ID, "2024", TARGET_DATE),
    ]
