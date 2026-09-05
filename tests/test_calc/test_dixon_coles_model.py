"""Unit tests for classic Dixon–Coles goals MLE (synthetic matches, no DB)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
import numpy as np

from src.calc.dixon_coles.metrics import (
    clip_and_normalize_probs,
    log_loss_one,
    ranked_probability_score,
)
from src.calc.dixon_coles.model import (
    DixonColesModel,
    filter_matches_by_lookback,
    match_weight,
)
from src.calc.dixon_coles.types import DixonColesMatch


def _match(
    day_offset: int,
    home_team_id: int,
    away_team_id: int,
    goals_home: int,
    goals_away: int,
    *,
    base: date = date(2024, 6, 1),
) -> DixonColesMatch:
    return DixonColesMatch(
        match_date=base + timedelta(days=day_offset),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        goals_home=goals_home,
        goals_away=goals_away,
    )


def _round_robin_home_heavy(as_of: date = date(2024, 12, 1)) -> list[DixonColesMatch]:
    """Synthetic league where home sides score more; enough matches per team."""
    teams = [1, 2, 3, 4]
    matches: list[DixonColesMatch] = []
    day = 0
    # Several rounds so each team has many appearances.
    for _round in range(8):
        for index, home in enumerate(teams):
            away = teams[(index + 1 + _round) % len(teams)]
            if home == away:
                away = teams[(index + 2) % len(teams)]
            matches.append(
                _match(
                    day,
                    home,
                    away,
                    goals_home=2,
                    goals_away=0,
                    base=as_of - timedelta(days=200),
                )
            )
            day += 3
    return matches


def test_match_weight_larger_xi_downweights_old_matches():
    days = 365
    light = match_weight(days, xi=0.001)
    heavy = match_weight(days, xi=0.005)
    assert heavy < light
    assert match_weight(0, xi=0.005) == pytest.approx(1.0)


def test_filter_matches_by_lookback_excludes_old_and_future():
    as_of = date(2024, 6, 1)
    matches = [
        _match(-400, 1, 2, 1, 0, base=as_of),
        _match(-10, 1, 2, 1, 0, base=as_of),
        _match(0, 1, 2, 1, 0, base=as_of),  # on as_of — excluded (strict <)
        _match(5, 1, 2, 1, 0, base=as_of),  # future
    ]
    filtered = filter_matches_by_lookback(matches, as_of=as_of, lookback_days=30)
    assert len(filtered) == 1
    assert filtered[0].match_date == as_of - timedelta(days=10)


def test_filter_matches_by_lookback_matches_fixture_date_index():
    as_of = date(2024, 6, 1)
    matches = [
        _match(-30, 2, 3, 1, 0, base=as_of),
        _match(-400, 1, 2, 1, 0, base=as_of),
        _match(-10, 1, 2, 1, 0, base=as_of),
        _match(0, 1, 2, 1, 0, base=as_of),
        _match(5, 1, 2, 1, 0, base=as_of),
    ]
    filtered = filter_matches_by_lookback(matches, as_of=as_of, lookback_days=30)
    from src.calc.dixon_coles.model import FixtureDateIndex

    indexed = FixtureDateIndex.from_matches(matches).window(
        as_of=as_of,
        lookback_days=30,
    )
    assert filtered == indexed


def test_vectorized_scoreline_matches_scalar_for_low_scores():
    from src.calc.dixon_coles.model import _vectorized_scoreline_log_probability
    from src.calc.strength_calculator import _scoreline_probability

    for home_goals in range(6):
        for away_goals in range(6):
            for lambda_home, lambda_away in ((1.2, 0.9), (0.5, 2.1)):
                scalar = _scoreline_probability(
                    home_goals,
                    away_goals,
                    lambda_home,
                    lambda_away,
                    rho=-0.13,
                )
                vector = float(
                    np.exp(
                        _vectorized_scoreline_log_probability(
                            np.array([home_goals]),
                            np.array([away_goals]),
                            np.array([lambda_home]),
                            np.array([lambda_away]),
                            rho=-0.13,
                        )[0]
                    )
                )
                assert vector == pytest.approx(scalar, rel=1e-10, abs=1e-12)


def test_vectorized_neg_log_likelihood_matches_scalar_loop():
    from src.calc.dixon_coles.model import (
        _scalar_neg_log_likelihood,
        _vectorized_neg_log_likelihood,
    )

    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    model = DixonColesModel(
        xi=0.001,
        rho=-0.1,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    windowed = filter_matches_by_lookback(matches, as_of=as_of, lookback_days=400)
    usable = model.matches_with_enough_team_history(windowed)
    team_ids = sorted(
        {
            team_id
            for match in usable
            for team_id in (match.home_team_id, match.away_team_id)
        }
    )
    team_index = {team_id: index for index, team_id in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_params = (n_teams - 1) + n_teams + 1

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        log_attack_free = theta[: n_teams - 1]
        log_defence = theta[n_teams - 1 : n_teams - 1 + n_teams]
        log_home_advantage = float(theta[-1])
        log_attack_last = -float(np.sum(log_attack_free))
        log_attack = np.concatenate([log_attack_free, [log_attack_last]])
        return np.exp(log_attack), np.exp(log_defence), float(np.exp(log_home_advantage))

    home_idx = np.fromiter(
        (team_index[match.home_team_id] for match in usable),
        dtype=np.intp,
    )
    away_idx = np.fromiter(
        (team_index[match.away_team_id] for match in usable),
        dtype=np.intp,
    )
    goals_home = np.fromiter((match.goals_home for match in usable), dtype=np.intp)
    goals_away = np.fromiter((match.goals_away for match in usable), dtype=np.intp)
    weights = np.fromiter(
        (
            match_weight((as_of - match.match_date).days, model.xi)
            for match in usable
        ),
        dtype=float,
    )
    positive = weights > 0
    home_idx = home_idx[positive]
    away_idx = away_idx[positive]
    goals_home = goals_home[positive]
    goals_away = goals_away[positive]
    weights = weights[positive]

    rng = np.random.default_rng(42)
    for _ in range(5):
        theta = rng.normal(size=n_params)
        scalar = _scalar_neg_log_likelihood(
            theta,
            unpack=unpack,
            usable=usable,
            team_index=team_index,
            cutoff=as_of,
            xi=model.xi,
            rho=model.rho,
        )
        vector = _vectorized_neg_log_likelihood(
            theta,
            unpack=unpack,
            home_idx=home_idx,
            away_idx=away_idx,
            goals_home=goals_home,
            goals_away=goals_away,
            weights=weights,
            rho=model.rho,
        )
        assert vector == pytest.approx(scalar, rel=1e-10, abs=1e-9)


def test_warm_start_fit_matches_cold_start_on_consecutive_days():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    kwargs = dict(xi=0.001, rho=-0.1, lookback_days=400, min_team_matches=3)
    day_one = as_of - timedelta(days=1)
    day_two = as_of
    first = DixonColesModel(**kwargs, as_of=day_one).fit(matches, as_of=day_one)
    cold_second = DixonColesModel(**kwargs, as_of=day_two).fit(matches, as_of=day_two)
    warm_second = DixonColesModel(**kwargs, as_of=day_two).fit(
        matches,
        as_of=day_two,
        initial_theta=first.fitted_theta,
    )
    cold_prediction = cold_second.predict(1, 2)
    warm_prediction = warm_second.predict(1, 2)
    assert warm_prediction.p_home == pytest.approx(cold_prediction.p_home, abs=1e-4)
    assert warm_prediction.p_draw == pytest.approx(cold_prediction.p_draw, abs=1e-4)
    assert warm_prediction.p_away == pytest.approx(cold_prediction.p_away, abs=1e-4)


def test_warm_start_reduces_optimizer_iterations_on_consecutive_days():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    kwargs = dict(
        xi=0.001,
        rho=-0.1,
        lookback_days=400,
        min_team_matches=3,
    )
    day_one = as_of - timedelta(days=1)
    day_two = as_of
    cold_second = DixonColesModel(**kwargs, as_of=day_two).fit(matches, as_of=day_two)
    first = DixonColesModel(**kwargs, as_of=day_one).fit(matches, as_of=day_one)
    warm_second = DixonColesModel(**kwargs, as_of=day_two).fit(
        matches,
        as_of=day_two,
        initial_theta=first.fitted_theta,
    )
    assert warm_second.last_fit_iterations is not None
    assert cold_second.last_fit_iterations is not None
    assert warm_second.last_fit_iterations <= cold_second.last_fit_iterations


def test_fit_probabilities_sum_to_one():
    as_of = date(2024, 12, 1)
    model = DixonColesModel(
        xi=0.001,
        rho=-0.1,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(1, 2)
    total = prediction.p_home + prediction.p_draw + prediction.p_away
    assert total == pytest.approx(1.0, abs=1e-9)


def test_fit_home_advantage_greater_than_one_when_home_scores_more():
    as_of = date(2024, 12, 1)
    model = DixonColesModel(
        xi=0.0005,
        rho=0.0,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    assert model.home_advantage > 1.0


def test_insufficient_history_raises():
    as_of = date(2024, 6, 1)
    matches = [_match(-1, 1, 2, 1, 0, base=as_of)]
    model = DixonColesModel(min_team_matches=5, lookback_days=30, as_of=as_of)
    with pytest.raises(ValueError, match="No matches left"):
        model.fit(matches, as_of=as_of)


def test_unknown_team_predict_uses_average_strength():
    as_of = date(2024, 12, 1)
    model = DixonColesModel(
        xi=0.001,
        rho=-0.1,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(home_team_id=999, away_team_id=1)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(
        1.0, abs=1e-9
    )


def test_different_rho_changes_low_score_probs():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    model_a = DixonColesModel(
        xi=0.001, rho=0.0, lookback_days=400, min_team_matches=3, as_of=as_of
    )
    model_b = DixonColesModel(
        xi=0.001, rho=-0.2, lookback_days=400, min_team_matches=3, as_of=as_of
    )
    model_a.fit(matches, as_of=as_of)
    model_b.attack = dict(model_a.attack)
    model_b.defence = dict(model_a.defence)
    model_b.home_advantage = model_a.home_advantage
    model_b.team_ids = list(model_a.team_ids)
    model_b._fitted = True
    pred_a = model_a.predict(1, 2)
    pred_b = model_b.predict(1, 2)
    assert (pred_a.p_home, pred_a.p_draw, pred_a.p_away) != (
        pred_b.p_home,
        pred_b.p_draw,
        pred_b.p_away,
    )


def test_fit_is_deterministic():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    kwargs = dict(xi=0.001, rho=-0.1, lookback_days=400, min_team_matches=3, as_of=as_of)
    first = DixonColesModel(**kwargs).fit(matches, as_of=as_of)
    second = DixonColesModel(**kwargs).fit(matches, as_of=as_of)
    assert first.home_advantage == pytest.approx(second.home_advantage)
    for team_id in first.team_ids:
        assert first.attack[team_id] == pytest.approx(second.attack[team_id])
        assert first.defence[team_id] == pytest.approx(second.defence[team_id])


def test_clip_and_normalize_probs():
    p_home, p_draw, p_away = clip_and_normalize_probs(0.5, 0.3, 0.2)
    assert p_home + p_draw + p_away == pytest.approx(1.0)


def test_log_loss_and_rps_known_values():
    # Certain correct home win → log loss 0 (after clip still near 0), RPS 0.
    assert log_loss_one("1", 1.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)
    assert ranked_probability_score("1", 1.0, 0.0, 0.0) == pytest.approx(0.0)
    # Uniform forecast, home win: RPS = 0.5 * ((1/3-1)^2 + (2/3-1)^2)
    expected_rps = 0.5 * ((1 / 3 - 1) ** 2 + (2 / 3 - 1) ** 2)
    assert ranked_probability_score("1", 1 / 3, 1 / 3, 1 / 3) == pytest.approx(
        expected_rps
    )
