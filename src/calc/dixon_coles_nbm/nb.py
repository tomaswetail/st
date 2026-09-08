"""Negative-binomial (NB2) PMF helpers for Dixon–Coles NBM."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

PHI_POISSON_THRESHOLD = 1e-8
LAMBDA_FLOOR = 1e-12


def _maybe_scalar(values: np.ndarray, *like: object) -> float | np.ndarray:
    if all(np.ndim(item) == 0 for item in like):
        return float(values)
    return values


def _poisson_log_pmf(goals: np.ndarray, mu: np.ndarray) -> np.ndarray:
    first_term = np.where(goals == 0, 0.0, goals * np.log(mu))
    return first_term - mu - gammaln(goals + 1.0)


def negative_binomial_log_pmf(
    k: float | np.ndarray,
    mu: float | np.ndarray,
    phi: float | np.ndarray,
) -> float | np.ndarray:
    """Log NB2 PMF; Poisson when ``phi <= 1e-8``. Clips ``mu`` to ``1e-12``."""
    goals = np.asarray(k, dtype=float)
    mean = np.maximum(np.asarray(mu, dtype=float), LAMBDA_FLOOR)
    dispersion = np.asarray(phi, dtype=float)

    poisson_log = _poisson_log_pmf(goals, mean)

    phi_safe = np.maximum(dispersion, PHI_POISSON_THRESHOLD)
    r = 1.0 / phi_safe
    last_term = np.where(goals == 0, 0.0, goals * np.log(mean / (r + mean)))
    nb_log = (
        gammaln(goals + r)
        - gammaln(r)
        - gammaln(goals + 1.0)
        + r * np.log(r / (r + mean))
        + last_term
    )
    result = np.where(dispersion <= PHI_POISSON_THRESHOLD, poisson_log, nb_log)
    return _maybe_scalar(result, k, mu, phi)


def negative_binomial_pmf(
    k: float | np.ndarray,
    mu: float | np.ndarray,
    phi: float | np.ndarray,
) -> float | np.ndarray:
    """NB2 PMF; Poisson when ``phi <= 1e-8``."""
    return _maybe_scalar(
        np.exp(np.asarray(negative_binomial_log_pmf(k, mu, phi), dtype=float)),
        k,
        mu,
        phi,
    )
