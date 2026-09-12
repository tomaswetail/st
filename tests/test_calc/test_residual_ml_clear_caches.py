"""Tests for residual ML calculator cache clearing."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from src.calc.home_advantage_calculator import HomeAdvantageCalculator
from src.calc.league_behavior_calculator import LeagueBehaviorCalculator
from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.calc.rest_congestion_calculator import RestCongestionCalculator
from src.calc.strength_calculator import StrengthCalculator


def test_clear_caches_empties_calculator_dicts():
    session = MagicMock()
    strength = StrengthCalculator(session)
    home_advantage = HomeAdvantageCalculator(
        session, strength_calculator=strength
    )
    strength._home_advantage_calculator = home_advantage
    rest = RestCongestionCalculator(session)
    league = LeagueBehaviorCalculator(session)
    assembler = ResidualMLFeatureAssembler(session)
    assembler.strength_calculator = strength
    assembler.home_advantage_calculator = home_advantage
    assembler.rest_calculator = rest
    assembler.league_behavior_calculator = league

    cutoff = date(2025, 1, 1)
    strength._team_features_cache[(1, cutoff)] = MagicMock()
    strength._league_averages_cache[(1, cutoff)] = MagicMock()
    strength._team_match_stats_cache[(1, cutoff, 10)] = MagicMock()
    strength._opponent_strength_cache[(1, cutoff)] = MagicMock()
    strength._league_goal_rates_cache[(1, cutoff)] = (1.0, 1.0)
    strength._team_by_id_cache[1] = MagicMock()
    home_advantage._features_cache[(1, cutoff)] = MagicMock()
    home_advantage._league_baselines_cache[(1, "2024", cutoff)] = {}
    home_advantage._competition_beta_cache[cutoff] = {}
    home_advantage._process_cache[(1, cutoff, None)] = MagicMock()
    rest._team_history_cache[("Arsenal", cutoff, 20)] = []
    league._features_cache[(1, cutoff)] = MagicMock()
    league._global_stats_cache[cutoff] = MagicMock()
    assembler._team_fixtures_cache[("Arsenal", cutoff, 20)] = []

    assembler.clear_caches()

    assert strength._team_features_cache == {}
    assert strength._league_averages_cache == {}
    assert strength._team_match_stats_cache == {}
    assert strength._opponent_strength_cache == {}
    assert strength._league_goal_rates_cache == {}
    assert 1 in strength._team_by_id_cache
    assert home_advantage._features_cache == {}
    assert home_advantage._league_baselines_cache == {}
    assert home_advantage._competition_beta_cache == {}
    assert home_advantage._process_cache == {}
    assert rest._team_history_cache == {}
    assert league._features_cache == {}
    assert league._global_stats_cache == {}
    assert assembler._team_fixtures_cache == {}
