"""Classic four-cell τ and NB2 Dixon–Coles scoreline matrix."""

from __future__ import annotations

import numpy as np

from src.calc.dixon_coles_nbm.nb import negative_binomial_pmf


def dixon_coles_nbm_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Classic Dixon–Coles τ(λ, ρ); not rewritten in NB masses."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _renormalize_dixon_coles_nbm(
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


def dixon_coles_nbm_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    phi: float = 0.05,
    max_goals: int = 10,
) -> tuple[list[list[float]], float, float, float]:
    """Untruncated NB2×τ cells on 0..G, then truncate/renormalize for 1X2."""
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

    tau = np.ones((size, size), dtype=float)
    tau[0, 0] = 1.0 - lambda_home * lambda_away * rho
    if size >= 2:
        tau[0, 1] = 1.0 + lambda_home * rho
        tau[1, 0] = 1.0 + lambda_away * rho
        tau[1, 1] = 1.0 - rho

    raw = np.maximum(0.0, independent * tau)
    total = float(raw.sum())
    home_win = float(np.sum(np.tril(raw, k=-1)))
    draw = float(np.trace(raw))
    away_win = float(np.sum(np.triu(raw, k=1)))
    return _renormalize_dixon_coles_nbm(raw, home_win, draw, away_win, total)
