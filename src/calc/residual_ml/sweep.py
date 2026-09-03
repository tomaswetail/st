"""Hyperparameter sweep and single-model training for residual ML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calc.residual_ml.trainer import ResidualMLTrainer
from utils.time_split import DEFAULT_VALIDATION_FRACTION, time_split_dataset_rows

MAX_DEPTHS = [3, 4, 6]
LEARNING_RATES = [0.03, 0.05, 0.1]
MAX_ITERS = [100, 300, 500]
LABEL_SMOOTHINGS = [0.0, 0.05, 0.1]


def _is_better_trial(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    if candidate["validation_log_loss"] < best["validation_log_loss"]:
        return True
    if candidate["validation_log_loss"] > best["validation_log_loss"]:
        return False
    if candidate["train_log_loss"] < best["train_log_loss"]:
        return True
    if candidate["train_log_loss"] > best["train_log_loss"]:
        return False
    return candidate["max_depth"] < best["max_depth"]


def run_hyperparameter_sweep(
    rows: list[dict[str, Any]],
    *,
    market_weight: float,
    dc_weight: float,
    output_dir: Path,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    exclude_injury_features: bool = False,
) -> dict[str, Any]:
    """Train all grid combos; save the lowest validation-log-loss model."""
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_trainer: ResidualMLTrainer | None = None
    total = (
        len(MAX_DEPTHS)
        * len(LEARNING_RATES)
        * len(MAX_ITERS)
        * len(LABEL_SMOOTHINGS)
    )
    trial_index = 0

    for max_depth in MAX_DEPTHS:
        for learning_rate in LEARNING_RATES:
            for max_iter in MAX_ITERS:
                for label_smoothing in LABEL_SMOOTHINGS:
                    trial_index += 1
                    train_rows, valid_rows = time_split_dataset_rows(
                        rows,
                        validation_fraction=validation_fraction,
                    )
                    trainer = ResidualMLTrainer(
                        market_weight=market_weight,
                        dc_weight=dc_weight,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        label_smoothing=label_smoothing,
                        exclude_injury_features=exclude_injury_features,
                    )
                    result = trainer.fit(train_rows, valid_rows)
                    trial = {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "max_iter": max_iter,
                        "label_smoothing": label_smoothing,
                        "validation_fraction": validation_fraction,
                        "train_rows": result.train_rows,
                        "validation_rows": result.validation_rows,
                        "train_log_loss": result.train_log_loss,
                        "validation_log_loss": result.validation_log_loss,
                        "market_validation_log_loss": result.market_validation_log_loss,
                        "blend_validation_log_loss": result.blend_validation_log_loss,
                    }
                    trials.append(trial)
                    print(
                        f"[{trial_index}/{total}] "
                        f"depth={max_depth} lr={learning_rate} iter={max_iter} "
                        f"smooth={label_smoothing} val_frac={validation_fraction} "
                        f"val_ll={result.validation_log_loss:.4f} "
                        f"train_ll={result.train_log_loss:.4f}",
                        flush=True,
                    )
                    if _is_better_trial(trial, best):
                        best = dict(trial)
                        best_trainer = trainer

    if best is None or best_trainer is None:
        raise ValueError("Hyperparameter sweep produced no trials")

    saved = best_trainer.save(output_dir, version="sweep_best")
    best["model_path"] = str(saved.model_path)

    sweep_path = output_dir / "sweep_results.json"
    sweep_path.write_text(
        json.dumps({"best": best, "trials": trials}, indent=2),
        encoding="utf-8",
    )
    print(f"Best params: {best}", flush=True)
    print(f"Wrote sweep results to {sweep_path}", flush=True)
    print(f"Model saved to {saved.model_path}", flush=True)
    return best


def train_and_save(
    rows: list[dict[str, Any]],
    *,
    market_weight: float,
    dc_weight: float,
    output_dir: Path,
    version: str,
    validation_fraction: float,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
    label_smoothing: float,
    exclude_injury_features: bool = False,
) -> None:
    train_rows, valid_rows = time_split_dataset_rows(
        rows,
        validation_fraction=validation_fraction,
    )
    print(
        f"Fitting model on {len(train_rows)} train / {len(valid_rows)} validation rows",
        flush=True,
    )
    trainer = ResidualMLTrainer(
        market_weight=market_weight,
        dc_weight=dc_weight,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        label_smoothing=label_smoothing,
        exclude_injury_features=exclude_injury_features,
    )
    result = trainer.fit(train_rows, valid_rows)
    saved = trainer.save(output_dir, version=version)
    if exclude_injury_features:
        print("Excluded injury feature columns from training", flush=True)
    print(f"Train rows: {result.train_rows}, validation rows: {result.validation_rows}")
    print(f"Train log loss: {result.train_log_loss:.4f}")
    print(f"Validation log loss: {result.validation_log_loss:.4f}")
    if result.market_validation_log_loss is not None:
        print(f"Market validation log loss: {result.market_validation_log_loss:.4f}")
    if result.blend_validation_log_loss is not None:
        print(f"Blend validation log loss: {result.blend_validation_log_loss:.4f}")
    print(f"Model saved to {saved.model_path}")
