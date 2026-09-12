"""Residual ML training, evaluation, and inference against the market baseline."""

from src.calc.residual_ml.baseline import (
    apply_large_move_or_identity,
    apply_residual_deltas,
    inv_logit,
    logit,
    market_baseline,
    normalize_probabilities,
    shrink_toward_market,
    target_logit_deltas,
)
from src.calc.residual_ml.dataset import ResidualMLDatasetBuilder
from src.calc.residual_ml.evaluation import (
    BacktestScoringResult,
    run_backtest_scoring,
    score_baseline_log_losses,
    score_outcome_metrics,
)
from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.calc.residual_ml.filters import filter_rows_by_draw, select_backtest_rows
from src.calc.residual_ml.io import load_dataset_rows
from src.calc.residual_ml.model import ResidualMLModel
from src.calc.residual_ml.sweep import run_hyperparameter_sweep, train_and_save
from src.calc.residual_ml.trainer import ResidualMLTrainer
from src.calc.residual_ml.validation import resolve_validation_fraction

__all__ = [
    "BacktestScoringResult",
    "ResidualMLDatasetBuilder",
    "ResidualMLFeatureAssembler",
    "ResidualMLModel",
    "ResidualMLTrainer",
    "apply_large_move_or_identity",
    "apply_residual_deltas",
    "filter_rows_by_draw",
    "inv_logit",
    "load_dataset_rows",
    "logit",
    "market_baseline",
    "normalize_probabilities",
    "resolve_validation_fraction",
    "run_backtest_scoring",
    "run_hyperparameter_sweep",
    "score_baseline_log_losses",
    "score_outcome_metrics",
    "select_backtest_rows",
    "shrink_toward_market",
    "target_logit_deltas",
    "train_and_save",
]
