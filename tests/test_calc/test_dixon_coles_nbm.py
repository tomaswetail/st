"""Unit tests for Dixon–Coles NBM (NB2 marginals, classic τ)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from src.calc.dixon_coles.model import filter_matches_by_lookback, match_weight
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles_nbm import (
    DixonColesNBMModel,
    dixon_coles_nbm_matrix,
    dixon_coles_nbm_tau,
    negative_binomial_pmf,
)
from src.calc.dixon_coles_nbm import model as nbm_model_module
from src.calc.strength_calculator import _dixon_coles_tau, _scoreline_probability, dixon_coles_matrix


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
    """Classic 4-team home-heavy 2–0 round-robin (8 rounds)."""
    teams = [1, 2, 3, 4]
    matches: list[DixonColesMatch] = []
    day = 0
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


def _untruncated_nbm_cell(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    phi: float,
) -> float:
    independent = negative_binomial_pmf(
        home_goals, lambda_home, phi
    ) * negative_binomial_pmf(away_goals, lambda_away, phi)
    tau = dixon_coles_nbm_tau(
        home_goals, away_goals, lambda_home, lambda_away, rho
    )
    return max(0.0, float(independent) * tau)


# --- A. NB PMF sums to 1 ---


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
@pytest.mark.parametrize("phi", [1e-4, 0.05, 0.2])
def test_a_nb_pmf_sums_to_one(mu: float, phi: float):
    total = sum(float(negative_binomial_pmf(k, mu, phi)) for k in range(201))
    assert total == pytest.approx(1.0, abs=1e-9)


# --- B. Mean matches λ ---


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
@pytest.mark.parametrize("phi", [1e-4, 0.05, 0.2])
def test_b_nb_mean_matches_lambda(mu: float, phi: float):
    expected = sum(k * float(negative_binomial_pmf(k, mu, phi)) for k in range(201))
    assert expected == pytest.approx(mu, abs=1e-6)


# --- C. Poisson limit ---


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
def test_c_poisson_limit_pmf(mu: float):
    for k in range(16):
        nb_mass = float(negative_binomial_pmf(k, mu, 1e-4))
        poisson_mass = float(poisson.pmf(k, mu))
        assert abs(nb_mass - poisson_mass) < 1e-4


def test_c_poisson_limit_matrix_matches_classic():
    nbm_matrix, nbm_p1, nbm_px, nbm_p2 = dixon_coles_nbm_matrix(
        1.2, 1.0, rho=-0.13, phi=1e-4, max_goals=10
    )
    classic_matrix, classic_p1, classic_px, classic_p2 = dixon_coles_matrix(
        1.2, 1.0, rho=-0.13, max_goals=10
    )
    for home_goals in range(11):
        for away_goals in range(11):
            assert abs(
                nbm_matrix[home_goals][away_goals]
                - classic_matrix[home_goals][away_goals]
            ) < 1e-4
    assert abs(nbm_p1 - classic_p1) < 1e-4
    assert abs(nbm_px - classic_px) < 1e-4
    assert abs(nbm_p2 - classic_p2) < 1e-4


# --- D. 1X2 sums to 1 ---


def test_d_matrix_1x2_sums_to_one_and_nonnegative():
    matrix, p_home, p_draw, p_away = dixon_coles_nbm_matrix(
        1.2, 1.0, rho=-0.13, phi=0.05, max_goals=10
    )
    assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-12)
    for row in matrix:
        for cell in row:
            assert cell >= 0.0


def test_d_predict_after_synthetic_fit_sums_to_one():
    as_of = date(2024, 12, 1)
    model = DixonColesNBMModel(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_phi=True,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(1, 2)
    total = prediction.p_home + prediction.p_draw + prediction.p_away
    assert total == pytest.approx(1.0, abs=1e-9)


# --- E. τ bounds / positivity ---


@pytest.mark.parametrize(
    "lambda_home, lambda_away, rho",
    [(1.2, 1.0, -0.13), (2.5, 2.0, 0.2), (5.0, 1.0, -0.2)],
)
def test_e_tau_matches_classic_four_cells(
    lambda_home: float,
    lambda_away: float,
    rho: float,
):
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        nbm_tau = dixon_coles_nbm_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho
        )
        classic_tau = _dixon_coles_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho
        )
        assert nbm_tau == pytest.approx(classic_tau, abs=1e-15)


def test_e_tau_zero_low_score_cells_clip_probability():
    assert dixon_coles_nbm_tau(0, 1, 5.0, 5.0, -0.2) == pytest.approx(0.0)
    assert dixon_coles_nbm_tau(1, 0, 5.0, 5.0, -0.2) == pytest.approx(0.0)
    assert _untruncated_nbm_cell(0, 1, 5.0, 5.0, -0.2, 0.05) == 0.0
    assert _untruncated_nbm_cell(1, 0, 5.0, 5.0, -0.2, 0.05) == 0.0


def test_e_negative_tau_00_clips_cell_probability():
    tau_00 = dixon_coles_nbm_tau(0, 0, 3.0, 3.0, 0.2)
    assert tau_00 < 0.0
    assert tau_00 == pytest.approx(1.0 - 1.8)
    assert _untruncated_nbm_cell(0, 0, 3.0, 3.0, 0.2, 0.05) == 0.0


def test_e_default_rho_positive_tau_on_lambda_box():
    for lambda_home in np.linspace(0.5, 3.0, 6):
        for lambda_away in np.linspace(0.5, 3.0, 6):
            for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
                tau = dixon_coles_nbm_tau(
                    home_goals,
                    away_goals,
                    float(lambda_home),
                    float(lambda_away),
                    -0.13,
                )
                assert tau > 0.0


# --- F. Tiny synthetic fit does not explode ---


def test_f_synthetic_fit_does_not_explode():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    kwargs = dict(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_rho=False,
        fit_phi=True,
    )
    model = DixonColesNBMModel(**kwargs).fit(matches, as_of=as_of)

    assert math.isfinite(model.home_advantage) and model.home_advantage > 1.0
    assert math.isfinite(model.phi) and model.phi > 0.0
    assert 1e-4 <= model.phi <= 2.0
    for team_id in model.team_ids:
        assert math.isfinite(model.attack[team_id]) and model.attack[team_id] > 0.0
        assert math.isfinite(model.defence[team_id]) and model.defence[team_id] > 0.0
    log_attack_sum = sum(math.log(model.attack[team_id]) for team_id in model.team_ids)
    assert abs(log_attack_sum) < 1e-10

    prediction = model.predict(1, 2)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(
        1.0, abs=1e-9
    )

    first = DixonColesNBMModel(**kwargs).fit(matches, as_of=as_of)
    second = DixonColesNBMModel(**kwargs).fit(matches, as_of=as_of)
    assert first.home_advantage == pytest.approx(second.home_advantage)
    assert first.phi == pytest.approx(second.phi)
    for team_id in first.team_ids:
        assert first.attack[team_id] == pytest.approx(second.attack[team_id])
        assert first.defence[team_id] == pytest.approx(second.defence[team_id])

    assert isinstance(model.last_fit_iterations, int)
    assert model.last_fit_iterations >= 0


# --- H. Additional ---


def test_h1_fit_reuses_classic_filter_and_excludes_as_of_match():
    assert nbm_model_module.filter_matches_by_lookback is filter_matches_by_lookback
    assert nbm_model_module.match_weight is match_weight

    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    on_cutoff = DixonColesMatch(
        match_date=as_of,
        home_team_id=1,
        away_team_id=2,
        goals_home=10,
        goals_away=10,
    )
    filtered = filter_matches_by_lookback(
        matches + [on_cutoff],
        as_of=as_of,
        lookback_days=400,
    )
    assert all(match.match_date < as_of for match in filtered)
    assert on_cutoff not in filtered

    kwargs = dict(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_phi=True,
    )
    baseline = DixonColesNBMModel(**kwargs).fit(matches, as_of=as_of)
    injected = DixonColesNBMModel(**kwargs).fit(matches + [on_cutoff], as_of=as_of)
    assert baseline.home_advantage == pytest.approx(injected.home_advantage)
    assert baseline.phi == pytest.approx(injected.phi)
    for team_id in baseline.team_ids:
        assert baseline.attack[team_id] == pytest.approx(injected.attack[team_id])
        assert baseline.defence[team_id] == pytest.approx(injected.defence[team_id])


def test_h2_unknown_team_predict_is_finite_and_sums_to_one():
    as_of = date(2024, 12, 1)
    model = DixonColesNBMModel(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_phi=True,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(999, 1)
    assert math.isfinite(prediction.p_home)
    assert math.isfinite(prediction.p_draw)
    assert math.isfinite(prediction.p_away)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(
        1.0, abs=1e-9
    )


def test_h3_insufficient_history_raises():
    as_of = date(2024, 6, 1)
    matches = [_match(-1, 1, 2, 1, 0, base=as_of)]
    model = DixonColesNBMModel(min_team_matches=5, lookback_days=30, as_of=as_of)
    with pytest.raises(ValueError, match="No matches left"):
        model.fit(matches, as_of=as_of)


def test_h4_nbm_worsens_classic_0_0_leftover():
    """NBM is expected to worsen the classic 0-0 leftover.

    NB2 overdispersion at the same mean raises P(Y=0) vs Poisson, so untruncated
    P(0,0) is strictly larger than classic Poisson×τ at the same λ, ρ.
    """
    lambda_home, lambda_away, rho, phi = 1.2, 1.0, -0.13, 0.05
    nbm_p00 = _untruncated_nbm_cell(
        0, 0, lambda_home, lambda_away, rho, phi
    )
    classic_p00 = _scoreline_probability(0, 0, lambda_home, lambda_away, rho)
    assert nbm_p00 > classic_p00


def test_h5_predict_before_fit_raises():
    model = DixonColesNBMModel()
    with pytest.raises(RuntimeError):
        model.predict(1, 2)


def test_h6_fit_phi_false_keeps_configured_phi():
    as_of = date(2024, 12, 1)
    configured_phi = 0.05
    model = DixonColesNBMModel(
        xi=0.001,
        rho=-0.1,
        phi=configured_phi,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_phi=False,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    assert model.phi == configured_phi
    assert model.fitted_phi is None


def test_h7_nb_pmf_matches_scipy_nbinom():
    mu = 1.4
    phi = 0.05
    r = 1.0 / phi
    p = r / (r + mu)
    for k in range(8):
        assert float(negative_binomial_pmf(k, mu, phi)) == pytest.approx(
            float(nbinom.pmf(k, r, p)),
            abs=1e-14,
        )
