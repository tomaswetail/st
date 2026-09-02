"""Residual ML training, evaluation, and inference."""

from calc.residual_ml.baseline import (
    apply_market_only_baseline,
    apply_residual_deltas,
    blend_baselines,
    is_market_only_weights,
    market_baseline,
    shrink_toward_market,
)
from calc.residual_ml.dataset import ResidualMLDatasetBuilder
from calc.residual_ml.evaluation import (
    BacktestScoringResult,
    run_backtest_scoring,
    score_baseline_log_losses,
)
from calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from calc.residual_ml.filters import filter_rows_by_draw, select_backtest_rows
from calc.residual_ml.io import load_dataset_rows
from calc.residual_ml.model import ResidualMLModel
from calc.residual_ml.sweep import run_hyperparameter_sweep, train_and_save
from calc.residual_ml.trainer import ResidualMLTrainer
from calc.residual_ml.validation import resolve_validation_fraction

__all__ = [
    "BacktestScoringResult",
    "ResidualMLDatasetBuilder",
    "ResidualMLFeatureAssembler",
    "ResidualMLModel",
    "ResidualMLTrainer",
    "apply_market_only_baseline",
    "apply_residual_deltas",
    "blend_baselines",
    "filter_rows_by_draw",
    "is_market_only_weights",
    "load_dataset_rows",
    "market_baseline",
    "resolve_validation_fraction",
    "run_backtest_scoring",
    "run_hyperparameter_sweep",
    "score_baseline_log_losses",
    "select_backtest_rows",
    "shrink_toward_market",
    "train_and_save",
]
