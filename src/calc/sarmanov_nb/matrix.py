"""Sarmanov–NB four-cell τ and bivariate scoreline matrix."""

from __future__ import annotations

import numpy as np

from src.calc.dixon_coles_nbm.nb import PHI_POISSON_THRESHOLD, negative_binomial_pmf


def a_factor(
    mu: float | np.ndarray,
    phi: float,
) -> float | np.ndarray:
    """Michels/Karlis a(μ, φ) = μ / (1 + φμ); Poisson limit a → μ."""
    mean = np.asarray(mu, dtype=float)
    if phi <= PHI_POISSON_THRESHOLD:
        result = mean
    else:
        result = mean / (1.0 + phi * mean)
    if np.ndim(mu) == 0:
        return float(result)
    return result


def admissible_rho_bounds(
    lambda_home: float,
    lambda_away: float,
    phi: float,
) -> tuple[float, float]:
    """Theoretical admissible ρ interval for the four-cell Sarmanov mixer."""
    a_home = a_factor(lambda_home, phi)
    a_away = a_factor(lambda_away, phi)
    lower = max(-1.0 / a_home, -1.0 / a_away) if a_home > 0 and a_away > 0 else float("-inf")
    upper = min(1.0 / (a_home * a_away), 1.0) if a_home > 0 and a_away > 0 else float("inf")
    return lower, upper


def sarmanov_nb_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    phi: float,
) -> float:
    """Four-cell Sarmanov τ with a(λ, φ) masses (not raw λ)."""
    a_home = a_factor(lambda_home, phi)
    a_away = a_factor(lambda_away, phi)
    if home_goals == 0 and away_goals == 0:
        return 1.0 - a_home * a_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + a_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + a_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _renormalize_sarmanov_nb(
    matrix: np.ndarray,
    home_win: float,
    draw: float,
    away_win: float,
    total: float,
) -> tuple[list[list[float]], float, float, float]:
    if total <= 0:
        return matrix.tolist(), 0.0, 0.0, 0.0
    scale = 1.0 / total
    return (matrix * scale).tolist(), home_win * scale, draw * scale, away_win * scale


def sarmanov_nb_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    phi: float = 0.05,
    max_goals: int = 10,
) -> tuple[list[list[float]], float, float, float]:
    """Untruncated NB2×τ_SNB cells on 0..G, then truncate/renormalize for 1X2."""
    size = max_goals + 1
    goals = np.arange(size)
    home_mass = np.asarray(
        negative_binomial_pmf(goals, lambda_home, phi),
        dtype=float,
    )
    away_mass = np.asarray(
        negative_binomial_pmf(goals, lambda_away, phi),
        dtype=float,
    )
    independent = np.outer(home_mass, away_mass)

    a_home = a_factor(lambda_home, phi)
    a_away = a_factor(lambda_away, phi)
    tau = np.ones((size, size), dtype=float)
    tau[0, 0] = 1.0 - a_home * a_away * rho
    if size >= 2:
        tau[0, 1] = 1.0 + a_home * rho
        tau[1, 0] = 1.0 + a_away * rho
        tau[1, 1] = 1.0 - rho

    raw = np.maximum(0.0, independent * tau)
    total = float(raw.sum())
    home_win = float(np.sum(np.tril(raw, k=-1)))
    draw = float(np.trace(raw))
    away_win = float(np.sum(np.triu(raw, k=1)))
    return _renormalize_sarmanov_nb(raw, home_win, draw, away_win, total)
