"""Unit tests for Sarmanov–NB (NB2 marginals, Michels/Karlis four-cell τ)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from src.calc.dixon_coles.model import filter_matches_by_lookback, match_weight
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles_nbm import dixon_coles_nbm_tau, negative_binomial_pmf
from src.calc.dixon_coles_nbm.nb import negative_binomial_pmf as nb_pmf_direct
from src.calc.sarmanov_nb import (
    SarmanovNBModel,
    a_factor,
    admissible_rho_bounds,
    sarmanov_nb_matrix,
    sarmanov_nb_tau,
)
from src.calc.sarmanov_nb import model as snb_model_module
from src.calc.sarmanov_nb.matrix import sarmanov_nb_tau as matrix_tau
from src.calc.strength_calculator import _dixon_coles_tau, dixon_coles_matrix


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


def _untruncated_snb_cell(
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
    tau = sarmanov_nb_tau(
        home_goals, away_goals, lambda_home, lambda_away, rho, phi
    )
    return max(0.0, float(independent) * tau)


# --- A. Marginals / NB import reuse ---


def test_a_nb_pmf_imported_from_dixon_coles_nbm():
    assert snb_model_module.negative_binomial_log_pmf is not None
    from src.calc.sarmanov_nb import matrix as snb_matrix_module

    assert snb_matrix_module.negative_binomial_pmf is nb_pmf_direct
    mu, phi = 1.4, 0.05
    for k in range(8):
        assert float(negative_binomial_pmf(k, mu, phi)) == pytest.approx(
            float(nb_pmf_direct(k, mu, phi))
        )


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
@pytest.mark.parametrize("phi", [1e-4, 0.05, 0.2])
def test_a_nb_marginal_pmf_sums_to_one(mu: float, phi: float):
    total = sum(float(negative_binomial_pmf(k, mu, phi)) for k in range(201))
    assert total == pytest.approx(1.0, abs=1e-9)


# --- B. a-factor definition and Poisson limit ---


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
@pytest.mark.parametrize("phi", [0.05, 0.2, 1.0])
def test_b_a_factor_formula(mu: float, phi: float):
    assert a_factor(mu, phi) == pytest.approx(mu / (1.0 + phi * mu))


@pytest.mark.parametrize("mu", [0.5, 1.4, 3.0])
def test_b_a_factor_poisson_limit(mu: float):
    assert a_factor(mu, 1e-8) == pytest.approx(mu)
    assert a_factor(mu, 1e-12) == pytest.approx(mu)


# --- C. Poisson limit ≈ classic DC ---


def test_c_poisson_limit_tau_matches_classic():
    lambda_home, lambda_away, rho = 1.2, 1.0, -0.13
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)):
        snb = sarmanov_nb_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho, phi=1e-12
        )
        classic = _dixon_coles_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho
        )
        assert snb == pytest.approx(classic, abs=1e-12)


def test_c_poisson_limit_matrix_matches_classic():
    snb_matrix, snb_p1, snb_px, snb_p2 = sarmanov_nb_matrix(
        1.2, 1.0, rho=-0.13, phi=1e-8, max_goals=10
    )
    classic_matrix, classic_p1, classic_px, classic_p2 = dixon_coles_matrix(
        1.2, 1.0, rho=-0.13, max_goals=10
    )
    for home_goals in range(11):
        for away_goals in range(11):
            assert abs(
                snb_matrix[home_goals][away_goals]
                - classic_matrix[home_goals][away_goals]
            ) < 1e-4
    assert abs(snb_p1 - classic_p1) < 1e-4
    assert abs(snb_px - classic_px) < 1e-4
    assert abs(snb_p2 - classic_p2) < 1e-4


# --- D. a-factor τ differs from NBM/raw-λ τ when φ > 0 ---


def test_d_a_factor_tau_differs_from_nbm_for_phi_positive():
    lambda_home, lambda_away, rho, phi = 1.2, 1.0, -0.13, 0.05
    a_home = a_factor(lambda_home, phi)
    a_away = a_factor(lambda_away, phi)
    assert a_home < lambda_home
    assert a_away < lambda_away

    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0)):
        snb = sarmanov_nb_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho, phi
        )
        nbm = dixon_coles_nbm_tau(
            home_goals, away_goals, lambda_home, lambda_away, rho
        )
        assert snb != pytest.approx(nbm, abs=1e-12)

    # (1,1) cell is identical (1 - ρ) in both mixers
    assert sarmanov_nb_tau(1, 1, lambda_home, lambda_away, rho, phi) == pytest.approx(
        dixon_coles_nbm_tau(1, 1, lambda_home, lambda_away, rho)
    )


# --- E. Joint non-negativity inside admissible / practical bound ---


def test_e_joint_nonnegative_inside_admissible_bounds():
    lambda_home, lambda_away, phi = 1.2, 1.0, 0.05
    lower, upper = admissible_rho_bounds(lambda_home, lambda_away, phi)
    for rho in (lower + 1e-9, -0.13, 0.0, upper - 1e-9):
        if not (lower <= rho <= upper):
            continue
        for home_goals in range(0, 6):
            for away_goals in range(0, 6):
                cell = _untruncated_snb_cell(
                    home_goals, away_goals, lambda_home, lambda_away, rho, phi
                )
                assert cell >= 0.0


def test_e_practical_rho_box_keeps_default_cells_positive():
    for lambda_home in np.linspace(0.5, 3.0, 6):
        for lambda_away in np.linspace(0.5, 3.0, 6):
            for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
                tau = sarmanov_nb_tau(
                    home_goals,
                    away_goals,
                    float(lambda_home),
                    float(lambda_away),
                    -0.13,
                    0.05,
                )
                assert tau > 0.0


# --- F. 1X2 renormalization ---


def test_f_matrix_1x2_sums_to_one_and_nonnegative():
    matrix, p_home, p_draw, p_away = sarmanov_nb_matrix(
        1.2, 1.0, rho=-0.13, phi=0.05, max_goals=10
    )
    assert p_home + p_draw + p_away == pytest.approx(1.0, abs=1e-12)
    for row in matrix:
        for cell in row:
            assert cell >= 0.0


def test_f_predict_after_synthetic_fit_sums_to_one():
    as_of = date(2024, 12, 1)
    model = SarmanovNBModel(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(1, 2)
    total = prediction.p_home + prediction.p_draw + prediction.p_away
    assert total == pytest.approx(1.0, abs=1e-9)


# --- G. Fit / predict smoke ---


def test_g_synthetic_fit_does_not_explode():
    as_of = date(2024, 12, 1)
    matches = _round_robin_home_heavy(as_of)
    kwargs = dict(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_rho=True,
        fit_phi=True,
    )
    model = SarmanovNBModel(**kwargs).fit(matches, as_of=as_of)

    assert math.isfinite(model.home_advantage) and model.home_advantage > 1.0
    assert math.isfinite(model.phi) and model.phi > 0.0
    assert 1e-4 <= model.phi <= 2.0
    assert model.rho_min <= model.rho <= model.rho_max
    for team_id in model.team_ids:
        assert math.isfinite(model.attack[team_id]) and model.attack[team_id] > 0.0
        assert math.isfinite(model.defence[team_id]) and model.defence[team_id] > 0.0
    log_attack_sum = sum(math.log(model.attack[team_id]) for team_id in model.team_ids)
    assert abs(log_attack_sum) < 1e-10

    prediction = model.predict(1, 2)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(
        1.0, abs=1e-9
    )
    goals = model.expected_goals(1, 2)
    assert math.isfinite(goals[0]) and math.isfinite(goals[1])
    as_dict = model.predict_dict(1, 2)
    assert set(as_dict) == {
        "lambda_home",
        "lambda_away",
        "phi",
        "p_home",
        "p_draw",
        "p_away",
    }

    first = SarmanovNBModel(**kwargs).fit(matches, as_of=as_of)
    second = SarmanovNBModel(**kwargs).fit(matches, as_of=as_of)
    assert first.home_advantage == pytest.approx(second.home_advantage)
    assert first.phi == pytest.approx(second.phi)
    assert first.rho == pytest.approx(second.rho)
    for team_id in first.team_ids:
        assert first.attack[team_id] == pytest.approx(second.attack[team_id])
        assert first.defence[team_id] == pytest.approx(second.defence[team_id])

    assert isinstance(model.last_fit_iterations, int)
    assert model.last_fit_iterations >= 0


# --- H. Fit flags ---


def test_h_defaults_fit_rho_and_phi_true():
    model = SarmanovNBModel()
    assert model.fit_rho is True
    assert model.fit_phi is True


def test_h_fit_flags_keep_configured_values():
    as_of = date(2024, 12, 1)
    configured_rho = -0.1
    configured_phi = 0.05
    model = SarmanovNBModel(
        xi=0.001,
        rho=configured_rho,
        phi=configured_phi,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
        fit_rho=False,
        fit_phi=False,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    assert model.rho == configured_rho
    assert model.phi == configured_phi
    assert model.fitted_rho is None
    assert model.fitted_phi is None


# --- I. Lookback / leakage ---


def test_i_fit_reuses_classic_filter_and_excludes_as_of_match():
    assert snb_model_module.filter_matches_by_lookback is filter_matches_by_lookback
    assert snb_model_module.match_weight is match_weight

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
        fit_rho=False,
        fit_phi=True,
    )
    baseline = SarmanovNBModel(**kwargs).fit(matches, as_of=as_of)
    injected = SarmanovNBModel(**kwargs).fit(matches + [on_cutoff], as_of=as_of)
    assert baseline.home_advantage == pytest.approx(injected.home_advantage)
    assert baseline.phi == pytest.approx(injected.phi)
    for team_id in baseline.team_ids:
        assert baseline.attack[team_id] == pytest.approx(injected.attack[team_id])
        assert baseline.defence[team_id] == pytest.approx(injected.defence[team_id])


# --- J. Misc API / edge ---


def test_j_predict_before_fit_raises():
    model = SarmanovNBModel()
    with pytest.raises(RuntimeError):
        model.predict(1, 2)


def test_j_insufficient_history_raises():
    as_of = date(2024, 6, 1)
    matches = [_match(-1, 1, 2, 1, 0, base=as_of)]
    model = SarmanovNBModel(min_team_matches=5, lookback_days=30, as_of=as_of)
    with pytest.raises(ValueError, match="No matches left"):
        model.fit(matches, as_of=as_of)


def test_j_unknown_team_predict_is_finite_and_sums_to_one():
    as_of = date(2024, 12, 1)
    model = SarmanovNBModel(
        xi=0.001,
        rho=-0.1,
        phi=0.05,
        lookback_days=400,
        min_team_matches=3,
        as_of=as_of,
    )
    model.fit(_round_robin_home_heavy(as_of), as_of=as_of)
    prediction = model.predict(999, 1)
    assert math.isfinite(prediction.p_home)
    assert math.isfinite(prediction.p_draw)
    assert math.isfinite(prediction.p_away)
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(
        1.0, abs=1e-9
    )


def test_j_public_exports_include_a_factor_and_tau():
    assert callable(a_factor)
    assert callable(matrix_tau)
    assert callable(admissible_rho_bounds)
    assert callable(sarmanov_nb_matrix)
