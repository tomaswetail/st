"""Unit tests for HomeAdvantageCalculator team-specific HA."""

from __future__ import annotations

from datetime import date, timedelta
from math import exp, log
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.home_advantage_calculator import (
    HomeAdvantageCalculator,
    HomeAdvantageResult,
    _MatchNpxg,
)
from calc.strength_helpers import baselines_from_stats, npxg_or_xg
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.team_strength_features import TeamStrengthFeatures
from objects.schema.db.team import Team


TARGET = date(2025, 3, 1)
TEAM = Team(id=1, name="Alpha", league_id=10)


def _match(
    match_date: date,
    home: str = "Alpha",
    away: str = "Beta",
    match_id: int = 1,
    league: str = "E0",
    season: str = "2425",
):
    return SimpleNamespace(
        id=match_id,
        match_date=match_date,
        home_team=home,
        away_team=away,
        league=league,
        season=season,
    )


def _row(
    *,
    match_date: date,
    played_at_home: bool,
    xg_for: float,
    xg_against: float,
    opponent: str = "Beta",
    match_id: int = 1,
    league: str = "E0",
    season: str = "2425",
) -> _MatchNpxg:
    home = "Alpha" if played_at_home else opponent
    away = opponent if played_at_home else "Alpha"
    return _MatchNpxg(
        match=_match(
            match_date,
            home=home,
            away=away,
            match_id=match_id,
            league=league,
            season=season,
        ),
        played_at_home=played_at_home,
        xg_for=xg_for,
        xg_against=xg_against,
        opponent_name=opponent,
        league_code=league,
        season=season,
    )


def _features(
    *,
    attack: float | None = 1.0,
    defence_weakness: float | None = 1.0,
) -> TeamStrengthFeatures:
    return TeamStrengthFeatures(
        team_id=1,
        before=TARGET,
        venue=None,
        lookback_matches=20,
        sample_size=10,
        non_penalty_xg_for=None,
        non_penalty_xg_against=None,
        average_shot_xg_for=None,
        average_shot_xg_against=None,
        home_attack_strength=None,
        home_defence_strength=None,
        away_attack_strength=None,
        away_defence_strength=None,
        recency_weighted_attack_rating=None,
        recency_weighted_defence_rating=None,
        opponent_adjusted_attack_strength=attack,
        opponent_adjusted_defence_strength=defence_weakness,
    )


def _calculator(
    *,
    shrinkage: int = 30,
    decay: float = 0.0,
    epsilon: float = 0.05,
    max_ha: float = 0.30,
    league_shrinkage: int = 30,
    league_prior: float = 0.20,
) -> HomeAdvantageCalculator:
    session = MagicMock()
    config = DataSourceConfig(
        home_advantage_shrinkage_matches=shrinkage,
        home_advantage_recency_decay_rate=decay,
        home_advantage_epsilon=epsilon,
        max_team_home_advantage=max_ha,
        league_home_advantage_shrinkage_matches=league_shrinkage,
        league_home_advantage_global_prior=league_prior,
    )
    strength = MagicMock()
    strength.league_averages_by_league_id.return_value = {
        "home_npxg": 1.4,
        "away_npxg": 1.0,
        "npxg": 1.2,
    }
    calc = HomeAdvantageCalculator(
        session=session,
        config=config,
        strength_calculator=strength,
    )
    calc.team_repo = MagicMock()
    calc.team_repo.get.return_value = SimpleNamespace(
        id=1, name="Alpha", league_id=10
    )
    calc.team_repo.get_by_name.return_value = SimpleNamespace(
        id=2, name="Beta", league_id=10
    )
    calc.league_repo = MagicMock()
    calc.league_repo.get_by_code.return_value = SimpleNamespace(id=10, name="EPL")
    calc.league_repo.get.return_value = SimpleNamespace(
        id=10, name="England Premier League"
    )
    calc.historical_match_repo = MagicMock()
    calc.historical_match_repo.get_home_goals_sum_by_league.return_value = (100, 50)
    calc.historical_match_repo.get_away_goals_sum_by_league.return_value = (80, 50)

    def _stub_league_ha(league_id, season, before_date, return_diagnostics=False):
        if return_diagnostics:
            return {
                "league_season_home_advantage": 0.1,
                "raw_league_season_home_advantage": 0.1,
                "league_home_advantage_shrinkage_weight": 1.0,
                "league_home_advantage_sample_size": 50,
            }
        return 0.1

    calc.calc_league_season_home_advantage = MagicMock(side_effect=_stub_league_ha)
    return calc


def _patch_unit_strength_expectations(calc: HomeAdvantageCalculator):
    """Strengths are 1.0 so expected npxG equals venue league baselines."""

    def expected_npxg(**kwargs):
        if kwargs["played_at_home"]:
            return kwargs["league_home_npxg"], kwargs["league_away_npxg"]
        return kwargs["league_away_npxg"], kwargs["league_home_npxg"]

    calc._expected_npxg = MagicMock(side_effect=expected_npxg)


def test_scale_raw_team_home_advantage_not_halved():
    calc = _calculator(shrinkage=0)
    _patch_unit_strength_expectations(calc)
    L_home, L_away = 1.4, 1.0
    epsilon = 0.05
    home_for = exp(0.10) * (L_home + epsilon) - epsilon
    home_against = (L_away + epsilon) / exp(0.10) - epsilon
    rows = [
        _row(
            match_date=TARGET - timedelta(days=1),
            played_at_home=True,
            xg_for=home_for,
            xg_against=home_against,
            match_id=1,
        ),
        _row(
            match_date=TARGET - timedelta(days=2),
            played_at_home=False,
            xg_for=L_away,
            xg_against=L_home,
            match_id=2,
        ),
    ]
    calc._load_team_npxg_matches = MagicMock(return_value=rows)
    result = calc.calculate_team_home_advantage(
        TEAM, league_id=10, season="2024", target_date=TARGET
    )
    assert result.raw_team_home_advantage == pytest.approx(0.20, abs=1e-6)


def test_season_specific_baseline_ignores_other_season():
    calc = _calculator(shrinkage=1)
    calls: list[tuple] = []

    def baselines(league_id, before_date, season=None):
        calls.append((league_id, season, before_date))
        if season == "2324":
            return {"home_npxg": 1.0, "away_npxg": 0.8, "npxg": 0.9}
        return {"home_npxg": 2.0, "away_npxg": 1.5, "npxg": 1.75}

    calc.strength_calculator.league_averages_by_league_id = MagicMock(
        side_effect=baselines
    )
    captured = []

    def expected_npxg(**kwargs):
        captured.append(kwargs)
        return kwargs["league_home_npxg"], kwargs["league_away_npxg"]

    calc._expected_npxg = MagicMock(side_effect=expected_npxg)
    hist = TARGET - timedelta(days=5)
    calc._load_team_npxg_matches = MagicMock(
        return_value=[
            _row(
                match_date=hist,
                played_at_home=True,
                xg_for=1.0,
                xg_against=0.8,
                season="2324",
            ),
            _row(
                match_date=hist - timedelta(days=1),
                played_at_home=False,
                xg_for=0.8,
                xg_against=1.0,
                match_id=2,
                season="2324",
            ),
        ]
    )
    calc.calculate_team_home_advantage(
        TEAM, league_id=10, season="2024", target_date=TARGET
    )
    assert all(call[1] == "2324" for call in calls)
    assert captured[0]["league_home_npxg"] == 1.0
    # Changing other-season return value must not affect season 2324 expectations.
    assert all(c["league_home_npxg"] == 1.0 for c in captured if c["played_at_home"])


def test_historical_match_uses_own_league_and_season():
    calc = _calculator(shrinkage=1)
    championship = SimpleNamespace(id=20, name="Championship")
    premier = SimpleNamespace(id=10, name="EPL")
    calc.league_repo.get_by_code.side_effect = (
        lambda code: championship if code == "E1" else premier
    )

    def baselines(league_id, before_date, season=None):
        if league_id == 20 and season == "2324":
            return {"home_npxg": 1.1, "away_npxg": 0.9, "npxg": 1.0}
        return {"home_npxg": 1.6, "away_npxg": 1.2, "npxg": 1.4}

    calc.strength_calculator.league_averages_by_league_id = MagicMock(
        side_effect=baselines
    )
    captured = []
    calc._expected_npxg = MagicMock(
        side_effect=lambda **kwargs: (
            captured.append(kwargs) or (1.0, 1.0)
        )
    )
    calc._load_team_npxg_matches = MagicMock(
        return_value=[
            _row(
                match_date=TARGET - timedelta(days=5),
                played_at_home=True,
                xg_for=1.1,
                xg_against=0.9,
                league="E1",
                season="2324",
            ),
            _row(
                match_date=TARGET - timedelta(days=2),
                played_at_home=False,
                xg_for=1.2,
                xg_against=1.6,
                match_id=2,
                league="E0",
                season="2425",
            ),
        ]
    )
    calc.calculate_team_home_advantage(
        TEAM, league_id=10, season="2024", target_date=TARGET
    )
    assert captured[0]["league_home_npxg"] == 1.1
    assert captured[1]["league_home_npxg"] == 1.6


def test_defence_weakness_semantics_in_expected_npxg():
    calc = _calculator()
    calc._team_features_before = MagicMock(
        side_effect=lambda team_id, before_date: (
            _features(attack=1.0, defence_weakness=1.0)
            if team_id == 1
            else _features(attack=1.0, defence_weakness=1.0)
        )
    )
    # Strong opponent defence weakness 0.8 → lower expected_for
    calc.team_repo.get_by_name.return_value = SimpleNamespace(id=2, name="Beta")
    calc._team_features_before = MagicMock(
        side_effect=lambda team_id, d: (
            _features(attack=1.0, defence_weakness=1.0)
            if team_id == 1
            else _features(attack=1.0, defence_weakness=0.8)
        )
    )
    strong = calc._expected_npxg(
        team_id=1,
        opponent_name="Beta",
        match_date=TARGET - timedelta(days=1),
        played_at_home=True,
        league_home_npxg=1.4,
        league_away_npxg=1.0,
        league_overall_npxg=1.2,
    )
    calc._team_features_before = MagicMock(
        side_effect=lambda team_id, d: (
            _features(attack=1.0, defence_weakness=1.0)
            if team_id == 1
            else _features(attack=1.0, defence_weakness=1.2)
        )
    )
    weak = calc._expected_npxg(
        team_id=1,
        opponent_name="Beta",
        match_date=TARGET - timedelta(days=1),
        played_at_home=True,
        league_home_npxg=1.4,
        league_away_npxg=1.0,
        league_overall_npxg=1.2,
    )
    assert strong[0] < weak[0]
    assert strong[0] == pytest.approx(1.4 * 1.0 * 0.8)
    assert weak[0] == pytest.approx(1.4 * 1.0 * 1.2)

    # Strong team defence → lower expected_against
    calc._team_features_before = MagicMock(
        side_effect=lambda team_id, d: (
            _features(attack=1.0, defence_weakness=0.8)
            if team_id == 1
            else _features(attack=1.0, defence_weakness=1.0)
        )
    )
    strong_def = calc._expected_npxg(
        team_id=1,
        opponent_name="Beta",
        match_date=TARGET - timedelta(days=1),
        played_at_home=True,
        league_home_npxg=1.4,
        league_away_npxg=1.0,
        league_overall_npxg=1.2,
    )
    calc._team_features_before = MagicMock(
        side_effect=lambda team_id, d: (
            _features(attack=1.0, defence_weakness=1.2)
            if team_id == 1
            else _features(attack=1.0, defence_weakness=1.0)
        )
    )
    weak_def = calc._expected_npxg(
        team_id=1,
        opponent_name="Beta",
        match_date=TARGET - timedelta(days=1),
        played_at_home=True,
        league_home_npxg=1.4,
        league_away_npxg=1.0,
        league_overall_npxg=1.2,
    )
    assert strong_def[1] < weak_def[1]


def test_league_ha_shrinkage_toward_prior():
    calc = _calculator(league_shrinkage=30, league_prior=0.20)
    # Avoid stubbing calc_league_season_home_advantage — use real method.
    calc.calc_league_season_home_advantage = (
        HomeAdvantageCalculator.calc_league_season_home_advantage.__get__(calc)
    )
    calc._league_home_advantage_prior = MagicMock(return_value=0.20)
    calc.historical_match_repo.get_home_goals_sum_by_league.return_value = (
        10,
        2,
    )  # rate 5.0
    calc.historical_match_repo.get_away_goals_sum_by_league.return_value = (
        2,
        2,
    )  # rate 1.0
    # raw = log(5) ≈ 1.609
    result = calc.calc_league_season_home_advantage(
        10, "2024", TARGET, return_diagnostics=True
    )
    raw = result["raw_league_season_home_advantage"]
    shrunk = result["league_season_home_advantage"]
    assert raw == pytest.approx(log(5.0))
    assert abs(shrunk - 0.20) < abs(shrunk - raw)
    assert shrunk == pytest.approx((2 / 32) * raw + (30 / 32) * 0.20)

    # Large sample approaches raw.
    calc.historical_match_repo.get_home_goals_sum_by_league.return_value = (5000, 2000)
    calc.historical_match_repo.get_away_goals_sum_by_league.return_value = (1000, 2000)
    large = calc.calc_league_season_home_advantage(
        10, "2024", TARGET, return_diagnostics=True
    )
    assert abs(
        large["league_season_home_advantage"] - large["raw_league_season_home_advantage"]
    ) < 0.05
    assert large["league_home_advantage_shrinkage_weight"] > 0.95


def test_no_cross_competition_season_fallback():
    calc = _calculator()
    calc.historical_match_repo.find_before_date_by_team.return_value = [
        _match(TARGET - timedelta(days=1), league="FAC", season="2425"),
    ]
    calc.league_repo.get.return_value = SimpleNamespace(
        id=10, name="England Premier League"
    )
    season = calc._resolve_season_from_team_history(TEAM, TARGET)
    assert season is None


def test_calendar_year_season_from_match_history_not_july_rule():
    calc = _calculator()
    may_date = date(2025, 5, 15)
    calc.historical_match_repo.find_before_date_by_team.return_value = [
        _match(may_date - timedelta(days=1), league="SWE", season="2025")
    ]
    calc.league_repo.get.return_value = SimpleNamespace(id=30, name="Allsvenskan")
    season = calc._resolve_season_from_team_history(
        Team(id=1, name="Alpha", league_id=30), may_date
    )
    assert season == "2025"


def test_strict_cutoff_uses_match_date_not_target():
    calc = _calculator(shrinkage=1)
    hist_date = TARGET - timedelta(days=10)
    baseline_calls: list[date] = []
    feature_calls: list[date] = []

    def baselines(league_id, before_date, season=None):
        baseline_calls.append(before_date)
        return {"home_npxg": 1.4, "away_npxg": 1.0, "npxg": 1.2}

    calc.strength_calculator.league_averages_by_league_id = MagicMock(
        side_effect=baselines
    )
    calc._expected_npxg = MagicMock(return_value=(1.4, 1.0))
    # Also verify features path when not stubbing expected
    original_features = calc._team_features_before

    def tracking_features(team_id, before_date):
        feature_calls.append(before_date)
        return _features()

    calc._load_team_npxg_matches = MagicMock(
        return_value=[
            _row(
                match_date=hist_date,
                played_at_home=True,
                xg_for=1.4,
                xg_against=1.0,
            ),
            _row(
                match_date=hist_date - timedelta(days=1),
                played_at_home=False,
                xg_for=1.0,
                xg_against=1.4,
                match_id=2,
            ),
        ]
    )
    calc.calculate_team_home_advantage(
        TEAM, league_id=10, season="2024", target_date=TARGET
    )
    assert hist_date in baseline_calls
    assert TARGET not in baseline_calls
    assert all(d < TARGET for d in baseline_calls)


def test_npxg_fallback_consistent_in_baselines_and_matches():
    assert npxg_or_xg(None, 1.7) == 1.7
    row = SimpleNamespace(
        home_xg=1.7,
        away_xg=0.9,
        home_non_penalty_xg=None,
        away_non_penalty_xg=None,
        home_set_piece_xg=None,
        away_set_piece_xg=None,
    )
    baselines = baselines_from_stats([row], decay=1.0)
    assert baselines["home_npxg"] == pytest.approx(1.7)
    assert baselines["away_npxg"] == pytest.approx(0.9)


def test_process_combines_league_and_team_once():
    calc = _calculator(shrinkage=1)
    calc._resolve_season_from_team_history = MagicMock(return_value="2024")
    calc.calc_league_season_home_advantage = MagicMock(
        return_value={
            "league_season_home_advantage": 0.12,
            "raw_league_season_home_advantage": 0.15,
            "league_home_advantage_shrinkage_weight": 0.5,
            "league_home_advantage_sample_size": 30,
        }
    )
    # process expects float unless return_diagnostics - fix stub
    def calc_ha(league_id, season, before_date, return_diagnostics=False):
        if return_diagnostics:
            return {
                "league_season_home_advantage": 0.12,
                "raw_league_season_home_advantage": 0.15,
                "league_home_advantage_shrinkage_weight": 0.5,
                "league_home_advantage_sample_size": 30,
            }
        return 0.12

    calc.calc_league_season_home_advantage = MagicMock(side_effect=calc_ha)
    team_only = HomeAdvantageResult(
        home_advantage=0.17,
        league_season_home_advantage=0.12,
        team_home_advantage=0.05,
        raw_team_home_advantage=0.08,
        team_home_advantage_shrinkage_weight=0.6,
        home_attack_residual=0.1,
        away_attack_residual=0.0,
        home_defence_residual=0.1,
        away_defence_residual=0.0,
        home_performance=0.2,
        away_performance=0.0,
        home_match_count=10,
        away_match_count=10,
    )
    calc.calculate_team_home_advantage = MagicMock(return_value=team_only)
    result = calc.process(TEAM, TARGET)
    assert result.home_advantage == pytest.approx(
        result.league_season_home_advantage + result.team_home_advantage
    )


def test_usable_sample_and_unknown_competition():
    calc = _calculator(shrinkage=30)
    calc.league_repo.get_by_code.return_value = None
    calc._load_team_npxg_matches = MagicMock(
        return_value=[
            _row(
                match_date=TARGET - timedelta(days=1),
                played_at_home=True,
                xg_for=1.4,
                xg_against=1.0,
                league="CUP",
            ),
            _row(
                match_date=TARGET - timedelta(days=2),
                played_at_home=False,
                xg_for=1.0,
                xg_against=1.4,
                match_id=2,
                league="CUP",
            ),
        ]
    )
    result = calc.calculate_team_home_advantage(
        TEAM, league_id=10, season="2024", target_date=TARGET
    )
    assert result.team_home_advantage == 0.0
    assert result.home_match_count == 0
