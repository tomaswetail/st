"""Residual ML training, evaluation, and inference."""

from src.calc.residual_ml.baseline import (
    apply_count_differential_logit_shift,
    apply_draw_adjustment,
    apply_market_only_baseline,
    apply_residual_deltas,
    blend_baselines,
    coverage_aware_shrink_alpha,
    is_market_only_weights,
    load_draw_adjustment_config,
    market_baseline,
    shrink_toward_market,
    shrink_toward_market_by_coverage,
)
from src.calc.residual_ml.blend_weights import (
    load_blend_weights_config,
    select_blend_weights,
)
from src.calc.residual_ml.dataset import ResidualMLDatasetBuilder
from src.calc.residual_ml.evaluation import (
    BacktestScoringResult,
    run_backtest_scoring,
    run_phase3_ablation,
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
    "apply_count_differential_logit_shift",
    "apply_draw_adjustment",
    "apply_market_only_baseline",
    "apply_residual_deltas",
    "blend_baselines",
    "filter_rows_by_draw",
    "is_market_only_weights",
    "load_blend_weights_config",
    "load_dataset_rows",
    "load_draw_adjustment_config",
    "market_baseline",
    "resolve_validation_fraction",
    "run_backtest_scoring",
    "run_hyperparameter_sweep",
    "run_phase3_ablation",
    "score_baseline_log_losses",
    "score_outcome_metrics",
    "select_backtest_rows",
    "select_blend_weights",
    "coverage_aware_shrink_alpha",
    "shrink_toward_market",
    "shrink_toward_market_by_coverage",
    "train_and_save",
]
