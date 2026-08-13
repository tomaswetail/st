"""Unit tests for StrengthCalculator helpers and Dixon-Coles."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.strength_calculator import (
    StrengthCalculator,
    dixon_coles_matrix,
    expected_goals_from_strengths,
)
from calc.strength_helpers import (
    normalize_strength,
    recency_weights,
    shrink,
    weighted_mean,
)
from objects.schema.data_classes.team_strength_features import TeamStrengthFeatures


def test_weighted_mean_basic():
    assert weighted_mean([1.0, 3.0], [1.0, 1.0]) == pytest.approx(2.0)
    assert weighted_mean([1.0, 3.0], [1.0, 3.0]) == pytest.approx(2.5)
    assert weighted_mean([], []) is None
    assert weighted_mean([1.0], [0.0]) is None


def test_recency_weights():
    assert recency_weights(4, decay=0.9) == pytest.approx(
        [1.0, 0.9, 0.81, 0.729]
    )
    assert recency_weights(0) == []


def test_shrink_toward_prior():
    assert shrink(2.0, sample_size=0, prior_rating=1.0, prior_strength=8) == pytest.approx(
        1.0
    )
    assert shrink(2.0, sample_size=8, prior_rating=1.0, prior_strength=8) == pytest.approx(
        1.5
    )
    assert shrink(None, sample_size=5) is None


def test_normalize_strength():
    assert normalize_strength(1.5, 1.0) == pytest.approx(1.5)
    assert normalize_strength(1.5, None) is None
    assert normalize_strength(1.5, 0.0) is None


def test_expected_goals_from_strengths_uses_neutral_league_rate():
    home, away = expected_goals_from_strengths(1.2, 0.9, 0.8, 1.1, 1.35)
    assert home == pytest.approx(1.35 * 1.2 * 0.9)
    assert away == pytest.approx(1.35 * 0.8 * 1.1)
    assert expected_goals_from_strengths(None, 1.0, 1.0, 1.0, 1.0) == (None, None)


def test_neutral_teams_have_no_embedded_home_advantage():
    home, away = expected_goals_from_strengths(1.0, 1.0, 1.0, 1.0, 1.35)
    assert home == pytest.approx(1.35)
    assert away == pytest.approx(1.35)


def test_external_home_advantage_coefficient_applies_cleanly():
    neutral_home_lambda, neutral_away_lambda = expected_goals_from_strengths(
        1.0, 1.0, 1.0, 1.0, 1.35
    )
    assert neutral_home_lambda == pytest.approx(1.35)
    assert neutral_away_lambda == pytest.approx(1.35)

    home_advantage_coefficient = 1.10
    adjusted_home_lambda = neutral_home_lambda * home_advantage_coefficient
    assert adjusted_home_lambda == pytest.approx(1.35 * 1.10)


def test_home_advantage_coefficient_applied_before_dixon_coles():
    calculator = StrengthCalculator(session=MagicMock())
    calculator._match_expected_goals = MagicMock(return_value=(1.50, 1.20))
    calculator._home_advantage_coefficient = MagicMock(return_value=1.10)
    calculator._dixon_coles_probs = MagicMock(return_value=(0.4, 0.3, 0.3))
    calculator._league_goal_rates = MagicMock(return_value=(1.35, 1.35))
    calculator.league_averages = MagicMock(return_value={"npxg": 1.0})

    home_team = SimpleNamespace(id=1, league_id=10, name="Arsenal")
    features = calculator._assemble_match_features(
        match_id=1,
        home_team=home_team,
        away_team=SimpleNamespace(id=2, league_id=10, name="Chelsea"),
        home_features=_team_features(),
        away_features=_team_features(),
        match_date=date(2024, 6, 1),
        target_league_code="E0",
    )

    assert features.expected_home_goals == pytest.approx(1.65)
    assert features.expected_away_goals == pytest.approx(1.20)
    called_home, called_away = calculator._dixon_coles_probs.call_args[0]
    assert called_home == pytest.approx(1.65)
    assert called_away == pytest.approx(1.20)
    calculator._home_advantage_coefficient.assert_called_once_with(
        home_team,
        date(2024, 6, 1),
        target_league_code="E0",
    )


def test_dixon_coles_probabilities_sum_to_one():
    _, p_home, p_draw, p_away = dixon_coles_matrix(1.5, 1.1, rho=-0.13, max_goals=10)
    assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-9)
    assert p_home > 0 and p_draw > 0 and p_away > 0


def test_dixon_coles_low_score_adjustment_changes_00():
    matrix_rho0, *_ = dixon_coles_matrix(1.2, 1.0, rho=0.0, max_goals=5)
    matrix_rho, *_ = dixon_coles_matrix(1.2, 1.0, rho=-0.13, max_goals=5)
    # With negative rho, P(0-0) increases vs independent Poisson after τ.
    assert matrix_rho[0][0] != pytest.approx(matrix_rho0[0][0])


def test_missing_inputs_stay_none():
    assert weighted_mean([1.0, 2.0], [1.0]) is None  # length mismatch
    assert math.isclose(shrink(1.0, 10, prior_strength=0) or 0, 1.0)


def _team_features(
    *,
    home_attack: float | None = 1.4,
    away_attack: float | None = 0.75,
    home_defence: float | None = 1.3,
    away_defence: float | None = 0.8,
    opponent_attack: float | None = 1.0,
    opponent_defence: float | None = 1.0,
    recency_attack: float | None = 2.0,
    recency_defence: float | None = 2.0,
) -> TeamStrengthFeatures:
    return TeamStrengthFeatures(
        team_id=1,
        before=datetime(2024, 1, 1, tzinfo=timezone.utc),
        venue=None,
        lookback_matches=20,
        sample_size=10,
        non_penalty_xg_for=1.0,
        non_penalty_xg_against=1.0,
        average_shot_xg_for=0.1,
        average_shot_xg_against=0.1,
        home_attack_strength=home_attack,
        home_defence_strength=home_defence,
        away_attack_strength=away_attack,
        away_defence_strength=away_defence,
        recency_weighted_attack_rating=recency_attack,
        recency_weighted_defence_rating=recency_defence,
        opponent_adjusted_attack_strength=opponent_attack,
        opponent_adjusted_defence_strength=opponent_defence,
    )


def test_match_expected_goals_ignores_venue_specific_strengths():
    calculator = StrengthCalculator(session=MagicMock())
    home_features = _team_features(
        home_attack=1.40,
        away_attack=0.75,
        opponent_attack=1.0,
        opponent_defence=1.0,
    )
    away_features = _team_features(
        home_attack=1.40,
        away_attack=0.75,
        opponent_attack=1.0,
        opponent_defence=1.0,
    )

    home_lambda, away_lambda = calculator._match_expected_goals(
        home_features,
        away_features,
        league_goal_rate=1.35,
        league_npxg=1.0,
    )

    assert home_lambda == pytest.approx(1.35)
    assert away_lambda == pytest.approx(1.35)


def test_neutral_strength_falls_back_to_normalized_recency():
    calculator = StrengthCalculator(session=MagicMock())
    features = _team_features(
        opponent_attack=None,
        opponent_defence=None,
        recency_attack=1.5,
        recency_defence=0.75,
    )

    assert calculator._neutral_attack_strength(features, 1.0) == pytest.approx(1.5)
    assert calculator._neutral_defence_strength(features, 1.0) == pytest.approx(0.75)


# tests/calc/test_strength_calculator_mocked.py

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calc.strength_calculator import StrengthCalculator


def test_strength_calculator_with_mocked_database_data():
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = MagicMock()
    config.team_strength_lookback_matches = 20
    config.team_strength_recency_decay = 0.90
    config.team_strength_prior_matches = 8
    config.football_data_feature_shrinkage_prior_matches = 8
    config.team_strength_min_venue_matches = 3
    config.goalkeeper_prior_shots = 100

    calculator = StrengthCalculator(
        session=MagicMock(),
        config=config,
        provider="test",
    )

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------
    arsenal = SimpleNamespace(
        id=1,
        name="Arsenal",
        league_id=100,
    )

    calculator.team_repo = MagicMock()
    calculator.team_repo.get.return_value = arsenal

    # ------------------------------------------------------------------
    # HistoricalMatch
    #
    # IMPORTANT: newest match first, because StrengthCalculator expects
    # repository history in newest-first order.
    # ------------------------------------------------------------------
    match_1 = SimpleNamespace(
        id=101,
        match_date=date(2026, 8, 1),
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=2,
        away_goals=1,
        league="PL",
    )

    match_2 = SimpleNamespace(
        id=102,
        match_date=date(2026, 7, 25),
        home_team="Liverpool",
        away_team="Arsenal",
        home_goals=1,
        away_goals=1,
        league="PL",
    )

    calculator.historical_repo = MagicMock()
    calculator.historical_repo.find_before_date_by_team.return_value = [
        match_1,
        match_2,
    ]

    # ------------------------------------------------------------------
    # MatchAdvancedStats
    # ------------------------------------------------------------------
    stats_1 = SimpleNamespace(
        match_id=101,

        home_non_penalty_xg=2.0,
        away_non_penalty_xg=0.8,

        home_xg=2.2,
        away_xg=0.9,
        home_xg_from_shots=2.2,
        away_xg_from_shots=0.9,

        home_shots=10,
        away_shots=7,

        home_shots_on_target=5,
        away_shots_on_target=3,

        home_set_piece_xg=0.40,
        away_set_piece_xg=0.10,

        home_xgot=2.1,
        away_xgot=0.7,
    )

    stats_2 = SimpleNamespace(
        match_id=102,

        home_non_penalty_xg=1.2,
        away_non_penalty_xg=1.0,

        home_xg=1.3,
        away_xg=1.1,
        home_xg_from_shots=1.3,
        away_xg_from_shots=1.1,

        home_shots=12,
        away_shots=8,

        home_shots_on_target=4,
        away_shots_on_target=4,

        home_set_piece_xg=0.30,
        away_set_piece_xg=0.20,

        home_xgot=1.4,
        away_xgot=1.0,
    )

    calculator.stats_repo = MagicMock()
    calculator.stats_repo.list_for_matches.return_value = [
        stats_1,
        stats_2,
    ]

    # ------------------------------------------------------------------
    # Mock league averages.
    # Keeps this test focused on the team calculations rather than
    # requiring another set of historical league matches.
    # ------------------------------------------------------------------
    calculator.league_averages = MagicMock(
        return_value={
            "npxg": 1.40,
            "home_npxg": 1.50,
            "away_npxg": 1.20,
            "set_piece_xg": 0.25,
        }
    )

    # Keep opponent adjustment separate from this basic smoke test.
    calculator._get_opponent_strength_before = MagicMock(
        return_value=(1.0, 1.0, 1.40)
    )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    before = datetime(2026, 8, 10)

    features = calculator.get_team_features(
        team_id=1,
        before=before,
        lookback_matches=20,
    )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert features.team_id == 1
    assert features.sample_size == 2

    # Recency:
    # match_1 weight = 1.0
    # match_2 weight = 0.9
    expected_npxg_for = (2.0 * 1.0 + 1.0 * 0.9) / 1.9
    expected_npxg_against = (0.8 * 1.0 + 1.2 * 0.9) / 1.9

    assert features.non_penalty_xg_for == pytest.approx(
        expected_npxg_for
    )
    assert features.non_penalty_xg_against == pytest.approx(
        expected_npxg_against
    )

    # Shot quality:
    # weighted xG / weighted shots
    expected_shot_quality_for = (
        2.2 * 1.0 + 1.1 * 0.9
    ) / (
        10 * 1.0 + 8 * 0.9
    )

    expected_shot_quality_against = (
        0.9 * 1.0 + 1.3 * 0.9
    ) / (
        7 * 1.0 + 12 * 0.9
    )

    assert features.average_shot_xg_for == pytest.approx(
        expected_shot_quality_for
    )
    assert features.average_shot_xg_against == pytest.approx(
        expected_shot_quality_against
    )

    # Home + away history was present.
    assert features.home_attack_strength is not None
    assert features.home_defence_strength is not None
    assert features.away_attack_strength is not None
    assert features.away_defence_strength is not None

    # Advanced features.
    assert features.opponent_adjusted_attack_strength is not None
    assert features.opponent_adjusted_defence_strength is not None

    assert features.set_piece_attack_strength is not None
    assert features.set_piece_defence_strength is not None

    assert features.goalkeeper_prevention_rating is not None

    assert features.has_xg_data is True
    assert features.has_xgot_data is True
    assert features.has_set_piece_data is True

    # ------------------------------------------------------------------
    # Verify leakage-safe query cutoff
    # ------------------------------------------------------------------
    calculator.historical_repo.find_before_date_by_team.assert_called_once_with(
        team_name="Arsenal",
        before_date=date(2026, 8, 10),
        venue=None,
        limit=20,
    )
