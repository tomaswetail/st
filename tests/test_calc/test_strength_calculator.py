from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from calc.strength_calculator import (
    StrengthCalculator,
    _scoreline_probability,
    dixon_coles_matrix,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_config():
    return SimpleNamespace(
        football_data_provider="test",
        team_strength_lookback_matches=20,
        team_strength_recency_decay=0.90,
        team_strength_prior_matches=8,
        football_data_feature_shrinkage_prior_matches=8,
        team_strength_min_venue_matches=3,
        goalkeeper_prior_shots=100,
        dixon_coles_rho=-0.13,
        dixon_coles_max_goals=10,
    )


def make_calculator():
    calculator = StrengthCalculator(
        session=MagicMock(),
        config=make_config(),
        provider="test",
    )

    calculator.team_repo = MagicMock()
    calculator.fixture_repo = MagicMock()
    calculator.stats_repo = MagicMock()

    return calculator


def make_team(
    team_id=1,
    name="Arsenal",
    league_id=100,
):
    return SimpleNamespace(
        id=team_id,
        name=name,
        league_id=league_id,
    )


def make_match(
    match_id,
    match_date,
    home_team,
    away_team,
    home_goals,
    away_goals,
):
    home_side = (
        home_team
        if hasattr(home_team, "name")
        else SimpleNamespace(name=home_team)
    )
    away_side = (
        away_team
        if hasattr(away_team, "name")
        else SimpleNamespace(name=away_team)
    )
    return SimpleNamespace(
        id=match_id,
        fixture_date=match_date,
        home_team=home_side,
        away_team=away_side,
        home_team_name=getattr(home_side, "name", None),
        away_team_name=getattr(away_side, "name", None),
        home_team_id=getattr(home_side, "id", None),
        away_team_id=getattr(away_side, "id", None),
        goals_home=home_goals,
        goals_away=away_goals,
        league_id=39,
        league_season=2024,
        league="PL",
    )


def make_stats(
    match_id,
    *,
    home_npxg,
    away_npxg,
    home_xg,
    away_xg,
    home_shots,
    away_shots,
    home_sot,
    away_sot,
    home_set_piece,
    away_set_piece,
    home_xgot,
    away_xgot,
):
    return SimpleNamespace(
        match_id=match_id,

        home_non_penalty_xg=home_npxg,
        away_non_penalty_xg=away_npxg,

        home_xg=home_xg,
        away_xg=away_xg,

        home_xg_from_shots=home_xg,
        away_xg_from_shots=away_xg,

        home_shots=home_shots,
        away_shots=away_shots,

        home_shots_on_target=home_sot,
        away_shots_on_target=away_sot,

        home_set_piece_xg=home_set_piece,
        away_set_piece_xg=away_set_piece,

        home_xgot=home_xgot,
        away_xgot=away_xgot,
    )


def make_match_team_features(
    *,
    attack,
    defence,
):
    """
    Minimal TeamStrengthFeatures-like object needed by
    StrengthCalculator._assemble_match_features().
    """
    return SimpleNamespace(
        non_penalty_xg_for=1.5,
        non_penalty_xg_against=1.2,

        average_shot_xg_for=0.12,
        average_shot_xg_against=0.10,

        home_attack_strength=attack,
        home_defence_strength=defence,

        away_attack_strength=attack,
        away_defence_strength=defence,

        opponent_adjusted_attack_strength=attack,
        opponent_adjusted_defence_strength=defence,

        recency_weighted_attack_rating=1.5,
        recency_weighted_defence_rating=1.2,

        set_piece_attack_strength=1.1,
        set_piece_defence_strength=0.9,

        goalkeeper_prevention_rating=0.03,
    )


# ----------------------------------------------------------------------
# 1. Main team feature aggregation
# ----------------------------------------------------------------------


def test_get_team_features_full_calculation():
    calculator = make_calculator()

    arsenal = make_team()
    calculator.team_repo.get.return_value = arsenal

    # Newest first
    match_1 = make_match(
        101,
        date(2026, 8, 1),
        "Arsenal",
        "Chelsea",
        2,
        1,
    )

    match_2 = make_match(
        102,
        date(2026, 7, 25),
        "Liverpool",
        "Arsenal",
        1,
        1,
    )

    calculator.fixture_repo.find_before_date_by_team.return_value = [
        match_1,
        match_2,
    ]

    stats_1 = make_stats(
        101,
        home_npxg=2.0,
        away_npxg=0.8,
        home_xg=2.2,
        away_xg=0.9,
        home_shots=10,
        away_shots=7,
        home_sot=5,
        away_sot=3,
        home_set_piece=0.40,
        away_set_piece=0.10,
        home_xgot=2.1,
        away_xgot=0.7,
    )

    stats_2 = make_stats(
        102,
        home_npxg=1.2,
        away_npxg=1.0,
        home_xg=1.3,
        away_xg=1.1,
        home_shots=12,
        away_shots=8,
        home_sot=4,
        away_sot=4,
        home_set_piece=0.30,
        away_set_piece=0.20,
        home_xgot=1.4,
        away_xgot=1.0,
    )

    calculator.stats_repo.list_for_matches.return_value = [
        stats_1,
        stats_2,
    ]

    calculator.league_averages = MagicMock(
        return_value={
            "npxg": 1.40,
            "home_npxg": 1.50,
            "away_npxg": 1.20,
            "set_piece_xg": 0.25,
        }
    )

    # Keep this test focused on aggregation.
    calculator._get_opponent_strength_before = MagicMock(
        return_value=(1.0, 1.0, 1.40)
    )

    before = datetime(2026, 8, 10)

    features = calculator.get_team_features(
        team_id=1,
        before=before,
        lookback_matches=20,
    )

    assert features.team_id == 1
    assert features.sample_size == 2

    # --------------------------------------------------------------
    # Recency weighting
    #
    # newest = 1.0
    # previous = 0.9
    # --------------------------------------------------------------

    expected_npxg_for = (
        2.0 * 1.0
        + 1.0 * 0.9
    ) / 1.9

    expected_npxg_against = (
        0.8 * 1.0
        + 1.2 * 0.9
    ) / 1.9

    assert features.non_penalty_xg_for == pytest.approx(
        expected_npxg_for
    )

    assert features.non_penalty_xg_against == pytest.approx(
        expected_npxg_against
    )

    # --------------------------------------------------------------
    # Shot quality = weighted xG / weighted shots
    # --------------------------------------------------------------

    expected_shot_quality_for = (
        2.2 * 1.0
        + 1.1 * 0.9
    ) / (
        10 * 1.0
        + 8 * 0.9
    )

    expected_shot_quality_against = (
        0.9 * 1.0
        + 1.3 * 0.9
    ) / (
        7 * 1.0
        + 12 * 0.9
    )

    assert features.average_shot_xg_for == pytest.approx(
        expected_shot_quality_for
    )

    assert features.average_shot_xg_against == pytest.approx(
        expected_shot_quality_against
    )

    # Venue calculations
    assert features.home_attack_strength is not None
    assert features.home_defence_strength is not None

    assert features.away_attack_strength is not None
    assert features.away_defence_strength is not None

    # Opponent adjusted
    assert features.opponent_adjusted_attack_strength is not None
    assert features.opponent_adjusted_defence_strength is not None

    # Set pieces
    assert features.set_piece_attack_strength is not None
    assert features.set_piece_defence_strength is not None

    # Goalkeeper
    assert features.goalkeeper_prevention_rating is not None
    assert features.goalkeeper_goals_prevented is not None

    # Availability flags
    assert features.has_xg_data is True
    assert features.has_xgot_data is True
    assert features.has_set_piece_data is True


# ----------------------------------------------------------------------
# 2. Opponent strength BEFORE historical match
# ----------------------------------------------------------------------


def test_get_opponent_strength_before():
    calculator = make_calculator()

    chelsea = make_team(
        team_id=2,
        name="Chelsea",
    )

    calculator.team_repo.get_by_name.return_value = chelsea

    match_1 = make_match(
        201,
        date(2026, 7, 20),
        "Chelsea",
        "Man City",
        1,
        1,
    )

    match_2 = make_match(
        202,
        date(2026, 7, 10),
        "Arsenal",
        "Chelsea",
        2,
        1,
    )

    calculator.fixture_repo.find_before_date_by_team.return_value = [
        match_1,
        match_2,
    ]

    stats_1 = make_stats(
        201,
        home_npxg=1.5,
        away_npxg=1.0,
        home_xg=1.5,
        away_xg=1.0,
        home_shots=10,
        away_shots=8,
        home_sot=4,
        away_sot=3,
        home_set_piece=0.2,
        away_set_piece=0.1,
        home_xgot=1.2,
        away_xgot=1.0,
    )

    stats_2 = make_stats(
        202,
        home_npxg=1.6,
        away_npxg=1.1,
        home_xg=1.6,
        away_xg=1.1,
        home_shots=11,
        away_shots=9,
        home_sot=5,
        away_sot=4,
        home_set_piece=0.2,
        away_set_piece=0.2,
        home_xgot=1.3,
        away_xgot=1.1,
    )

    calculator.stats_repo.list_for_matches.return_value = [
        stats_1,
        stats_2,
    ]

    calculator.league_averages = MagicMock(
        return_value={
            "npxg": 1.40,
        }
    )

    attack, defence, league_npxg = (
        calculator._get_opponent_strength_before(
            "Chelsea",
            date(2026, 8, 1),
        )
    )

    assert league_npxg == pytest.approx(1.40)

    assert attack is not None
    assert defence is not None

    # Shrinkage should pull both towards league-average 1.0.
    assert 0.5 < attack < 1.5
    assert 0.5 < defence < 1.5

    # Most important leakage assertion.
    calculator.fixture_repo.find_before_date_by_team.assert_called_once_with(
        team_name="Chelsea",
        before_date=date(2026, 8, 1),
        venue=None,
        limit=20,
    )


# ----------------------------------------------------------------------
# 3. Opponent-adjusted performance calculation
# ----------------------------------------------------------------------


def test_opponent_adjustment():
    calculator = make_calculator()

    calculator._get_opponent_strength_before = MagicMock(
        return_value=(
            1.20,  # opponent attack
            0.80,  # opponent defence
            1.50,  # league npxG
        )
    )

    buckets = SimpleNamespace(
        opponent_adjusted_attack=[],
        opponent_adjusted_defence=[],
    )

    metrics = SimpleNamespace(
        opponent_team_name="Chelsea",
        match_date=date(2026, 7, 1),
        non_penalty_xg_for=1.80,
        non_penalty_xg_against=1.20,
    )

    calculator._accumulate_opponent_adjustment(
        buckets,
        metrics,
        match_weight=1.0,
    )

    # Attack:
    # expected conceded = 1.5 * 0.8 = 1.2
    # 1.8 / 1.2 = 1.5
    assert buckets.opponent_adjusted_attack[0][0] == pytest.approx(
        1.5
    )

    # Defence:
    # expected created = 1.5 * 1.2 = 1.8
    # 1.2 / 1.8 = 0.6667
    assert buckets.opponent_adjusted_defence[0][0] == pytest.approx(
        1.2 / 1.8
    )


# ----------------------------------------------------------------------
# 4. League averages
# ----------------------------------------------------------------------


def test_league_averages_uses_only_matches_before_cutoff():
    calculator = make_calculator()

    match_1 = make_match(
        301,
        date(2026, 7, 20),
        "Arsenal",
        "Chelsea",
        2,
        1,
    )

    match_2 = make_match(
        302,
        date(2026, 7, 10),
        "Liverpool",
        "Everton",
        1,
        0,
    )

    calculator.fixture_repo.find_before_date_by_league_id.return_value = [
        match_1,
        match_2,
    ]

    stat_1 = SimpleNamespace(match_id=301)
    stat_2 = SimpleNamespace(match_id=302)

    calculator.stats_repo.list_for_matches.return_value = [
        stat_1,
        stat_2,
    ]

    expected = {
        "npxg": 1.4,
        "home_npxg": 1.5,
        "away_npxg": 1.3,
    }

    with patch(
        "calc.strength_calculator.baselines_from_stats",
        return_value=expected,
    ) as baseline_mock:

        result = calculator.league_averages_by_league_id(
            league_id=100,
            before_date=date(2026, 8, 1),
        )

    assert result == expected

    calculator.fixture_repo.find_before_date_by_league_id.assert_called_once_with(
        league_id=100,
        before_date=date(2026, 8, 1),
        season=None,
        limit=500,
    )

    baseline_mock.assert_called_once_with(
        [stat_1, stat_2],
        decay=0.90,
    )


# ----------------------------------------------------------------------
# 5. Match expected goals + HA + Dixon-Coles
# ----------------------------------------------------------------------


def test_get_match_features_expected_goals_and_probabilities():
    calculator = make_calculator()

    arsenal = make_team(
        1,
        "Arsenal",
    )

    chelsea = make_team(
        2,
        "Chelsea",
    )

    target_match = make_match(
        500,
        date(2026, 8, 10),
        arsenal,
        chelsea,
        0,
        0,
    )

    calculator.fixture_repo.get.return_value = target_match

    home_features = make_match_team_features(
        attack=1.20,
        defence=1.10,
    )

    away_features = make_match_team_features(
        attack=0.80,
        defence=0.90,
    )

    def fake_team_features(
        team_id,
        before,
        venue=None,
        lookback_matches=None,
    ):
        if team_id == 1:
            return home_features

        return away_features

    calculator.get_team_features = MagicMock(
        side_effect=fake_team_features
    )

    # Average = (1.5 + 1.1) / 2 = 1.3
    calculator._league_goal_rates = MagicMock(
        return_value=(1.50, 1.10)
    )

    calculator.league_averages = MagicMock(
        return_value={
            "npxg": 1.40,
        }
    )

    # 10% multiplicative home advantage
    calculator._home_advantage_coefficient = MagicMock(
        return_value=1.10
    )

    features = calculator.get_match_features(500)

    # Before HA:
    #
    # home:
    # 1.3 * 1.2 * 0.9 = 1.404
    #
    # HA:
    # 1.404 * 1.1 = 1.5444
    expected_home = 1.3 * 1.2 * 0.9 * 1.10

    # away:
    # 1.3 * 0.8 * 1.1
    expected_away = 1.3 * 0.8 * 1.1

    assert features.expected_home_goals == pytest.approx(
        expected_home
    )

    assert features.expected_away_goals == pytest.approx(
        expected_away
    )

    assert features.dixon_coles_home_probability is not None
    assert features.dixon_coles_draw_probability is not None
    assert features.dixon_coles_away_probability is not None

    probability_sum = (
        features.dixon_coles_home_probability
        + features.dixon_coles_draw_probability
        + features.dixon_coles_away_probability
    )

    assert probability_sum == pytest.approx(
        1.0,
        abs=1e-10,
    )


# ----------------------------------------------------------------------
# 6. Dixon-Coles matrix
# ----------------------------------------------------------------------


def test_dixon_coles_probability_matrix_sums_to_one():
    matrix, home, draw, away = dixon_coles_matrix(
        lambda_home=1.50,
        lambda_away=1.10,
        rho=-0.13,
        max_goals=10,
    )

    matrix_total = sum(
        sum(row)
        for row in matrix
    )

    assert matrix_total == pytest.approx(
        1.0,
        abs=1e-10,
    )

    assert home + draw + away == pytest.approx(
        1.0,
        abs=1e-10,
    )

    assert 0 <= home <= 1
    assert 0 <= draw <= 1
    assert 0 <= away <= 1


# ----------------------------------------------------------------------
# 7. Dixon-Coles low-score correction
# ----------------------------------------------------------------------


def test_dixon_coles_low_score_adjustment():
    lambda_home = 1.5
    lambda_away = 1.1

    independent_00 = _scoreline_probability(
        0,
        0,
        lambda_home,
        lambda_away,
        rho=0.0,
    )

    corrected_00 = _scoreline_probability(
        0,
        0,
        lambda_home,
        lambda_away,
        rho=-0.13,
    )

    assert corrected_00 != pytest.approx(
        independent_00
    )

    # 2-2 should NOT receive Dixon-Coles correction.
    independent_22 = _scoreline_probability(
        2,
        2,
        lambda_home,
        lambda_away,
        rho=0.0,
    )

    corrected_22 = _scoreline_probability(
        2,
        2,
        lambda_home,
        lambda_away,
        rho=-0.13,
    )

    assert corrected_22 == pytest.approx(
        independent_22
    )


# ----------------------------------------------------------------------
# 8. Missing history
# ----------------------------------------------------------------------


def test_no_history_returns_empty_features():
    calculator = make_calculator()

    arsenal = make_team()

    calculator.team_repo.get.return_value = arsenal

    calculator.fixture_repo.find_before_date_by_team.return_value = []

    features = calculator.get_team_features(
        team_id=1,
        before=datetime(2026, 8, 10),
    )

    assert features.sample_size == 0

    assert features.non_penalty_xg_for is None
    assert features.non_penalty_xg_against is None

    assert features.home_attack_strength is None
    assert features.home_defence_strength is None

    assert features.away_attack_strength is None
    assert features.away_defence_strength is None


# ----------------------------------------------------------------------
# 9. Missing advanced stats are skipped
# ----------------------------------------------------------------------


def test_matches_without_advanced_stats_are_skipped():
    calculator = make_calculator()

    arsenal = make_team()

    match_1 = make_match(
        601,
        date(2026, 7, 1),
        "Arsenal",
        "Chelsea",
        2,
        0,
    )

    calculator.fixture_repo.find_before_date_by_team.return_value = [
        match_1
    ]

    # No advanced stats available for match.
    calculator.stats_repo.list_for_matches.return_value = []

    rows = calculator._load_team_match_stats(
        team=arsenal,
        before_date=date(2026, 8, 1),
        venue=None,
        lookback_matches=20,
    )

    assert rows == []


# ----------------------------------------------------------------------
# 10. Leakage-safe cutoff
# ----------------------------------------------------------------------


def test_history_query_uses_strict_pre_match_cutoff():
    calculator = make_calculator()

    arsenal = make_team()

    calculator.team_repo.get.return_value = arsenal

    calculator.fixture_repo.find_before_date_by_team.return_value = []

    target_date = datetime(
        2026,
        8,
        10,
        15,
        0,
    )

    calculator.get_team_features(
        team_id=1,
        before=target_date,
        lookback_matches=20,
    )

    calculator.fixture_repo.find_before_date_by_team.assert_called_once_with(
        team_name="Arsenal",
        before_date=date(2026, 8, 10),
        venue=None,
        limit=20,
    )


# ----------------------------------------------------------------------
# 11. Venue/recency fallback
# ----------------------------------------------------------------------


def test_side_strength_prefers_venue_strength():
    calculator = make_calculator()

    features = make_match_team_features(
        attack=1.25,
        defence=0.85,
    )

    result = calculator._side_attack_strength(
        features,
        venue="home",
        league_npxg=1.4,
    )

    assert result == pytest.approx(1.25)


def test_side_strength_falls_back_to_recency():
    calculator = make_calculator()

    features = make_match_team_features(
        attack=1.25,
        defence=0.85,
    )

    features.home_attack_strength = None
    features.recency_weighted_attack_rating = 1.40

    result = calculator._side_attack_strength(
        features,
        venue="home",
        league_npxg=1.40,
    )

    # 1.4 / 1.4
    assert result == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 12. Neutral strength prefers opponent-adjusted rating
# ----------------------------------------------------------------------


def test_neutral_strength_prefers_opponent_adjusted():
    calculator = make_calculator()

    features = make_match_team_features(
        attack=1.25,
        defence=0.85,
    )

    features.opponent_adjusted_attack_strength = 1.30
    features.opponent_adjusted_defence_strength = 0.75

    assert calculator._neutral_attack_strength(
        features,
        league_npxg=1.4,
    ) == pytest.approx(1.30)

    assert calculator._neutral_defence_strength(
        features,
        league_npxg=1.4,
    ) == pytest.approx(0.75)
