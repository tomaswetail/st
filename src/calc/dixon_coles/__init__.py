"""Classic Dixon–Coles goals MLE package."""

from src.calc.dixon_coles.metrics import (
    clip_and_normalize_probs,
    mean_log_loss,
    mean_rps,
    ranked_probability_score,
)
from src.calc.dixon_coles.model import (
    DixonColesModel,
    DixonColesPrediction,
    FixtureDateIndex,
    filter_matches_by_lookback,
    match_weight,
)
from src.calc.dixon_coles.optimizer import (
    DixonColesLeagueOptimizationResult,
    DixonColesOptimizationResult,
    DixonColesOptimizer,
    DixonColesParameterResult,
)
from src.calc.dixon_coles.service import DixonColesService
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles.walk_forward import (
    EvalMatch,
    WalkForwardResult,
    run_walk_forward,
    run_walk_forward_per_league,
)

__all__ = [
    "DixonColesLeagueOptimizationResult",
    "DixonColesMatch",
    "DixonColesModel",
    "DixonColesOptimizationResult",
    "DixonColesOptimizer",
    "DixonColesParameterResult",
    "DixonColesPrediction",
    "DixonColesService",
    "EvalMatch",
    "FixtureDateIndex",
    "WalkForwardResult",
    "clip_and_normalize_probs",
    "filter_matches_by_lookback",
    "match_weight",
    "mean_log_loss",
    "mean_rps",
    "ranked_probability_score",
    "run_walk_forward",
    "run_walk_forward_per_league",
]
