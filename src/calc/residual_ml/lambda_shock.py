"""T2: multiplicative Dixon–Coles λ shock from unavailable counts."""

from __future__ import annotations

from typing import Mapping

from src.calc.residual_ml.baseline import (
    _parse_has_availability_flag,
    _parse_optional_count,
    normalize_probabilities,
)
from src.calc.strength_calculator import dixon_coles_matrix
from src.data_sources.classic_dc_config import load_league_params
from src.objects.schema.data_classes.data_sources import DataSourceConfig

LAMBDA_SHOCK_K = 0.02
LAMBDA_MULTIPLIER_MIN = 0.70
LAMBDA_MULTIPLIER_MAX = 1.30


def clip_lambda_multiplier(multiplier: float) -> float:
    return min(LAMBDA_MULTIPLIER_MAX, max(LAMBDA_MULTIPLIER_MIN, multiplier))


def shock_expected_goals(
    lambda_home: float,
    lambda_away: float,
    *,
    home_unavailable_count: object,
    away_unavailable_count: object,
    has_availability: object,
    k: float = LAMBDA_SHOCK_K,
) -> tuple[float, float]:
    """Return shocked (λ_H, λ_A). Unchanged when uncovered or counts are null."""
    if _parse_has_availability_flag(has_availability) != 1:
        return float(lambda_home), float(lambda_away)
    home_count = _parse_optional_count(home_unavailable_count)
    away_count = _parse_optional_count(away_unavailable_count)
    if home_count is None or away_count is None:
        return float(lambda_home), float(lambda_away)
    home_from_home_missing = clip_lambda_multiplier(1.0 - k * home_count)
    home_from_away_missing = clip_lambda_multiplier(1.0 + k * away_count)
    away_from_away_missing = clip_lambda_multiplier(1.0 - k * away_count)
    away_from_home_missing = clip_lambda_multiplier(1.0 + k * home_count)
    return (
        float(lambda_home) * home_from_home_missing * home_from_away_missing,
        float(lambda_away) * away_from_away_missing * away_from_home_missing,
    )


def rho_for_league(
    league_external_id: object,
    *,
    default_rho: float | None = None,
    league_params: Mapping[int, object] | None = None,
) -> float:
    """JSON median ρ for leagues 39/41/42/45/180; else DataSourceConfig / −0.13."""
    fallback = (
        DataSourceConfig().dixon_coles_rho if default_rho is None else default_rho
    )
    params = league_params if league_params is not None else load_league_params()
    if league_external_id in (None, ""):
        return float(fallback)
    try:
        league_id = int(float(league_external_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(fallback)
    entry = params.get(league_id)
    if entry is None:
        return float(fallback)
    return float(entry.rho)  # type: ignore[attr-defined]


def dixon_coles_1x2(
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> dict[str, float]:
    _matrix, p_home, p_draw, p_away = dixon_coles_matrix(
        lambda_home, lambda_away, rho
    )
    normalized = normalize_probabilities({"1": p_home, "X": p_draw, "2": p_away})
    if normalized is None:
        return {"1": float(p_home), "X": float(p_draw), "2": float(p_away)}
    return normalized
