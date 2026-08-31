#!/usr/bin/env python3
"""Train the residual 1X2 ML model from a prebuilt dataset CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from calc.residual_ml_baseline import apply_market_only_baseline
from calc.residual_ml_dataset import ResidualMLDatasetBuilder
from calc.residual_ml_trainer import ResidualMLTrainer
from objects.schema.data_classes.data_sources import DataSourceConfig

MAX_DEPTHS = [3, 4, 6]
LEARNING_RATES = [0.03, 0.05, 0.1]
MAX_ITERS = [100, 300, 500]
LABEL_SMOOTHINGS = [0.0, 0.05, 0.1]
VALIDATION_FRACTIONS = [0.15, 0.2]

_STRING_FIELDS = frozenset(
    {
        "label",
        "match_date",
        "feature_cutoff_date",
    }
)


def repo_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "scripts":
        return cwd.parent
    return cwd


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def load_dataset_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                if value is None or value == "":
                    row[key] = None
                elif key in _STRING_FIELDS:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                        if row[key].is_integer() and key in {
                            "match_id",
                            "draw_number",
                            "home_matches_last_14_days",
                            "away_matches_last_14_days",
                            "home_short_rest",
                            "away_short_rest",
                            "home_extra_time_in_previous_match",
                            "away_extra_time_in_previous_match",
                            "extra_time_x_short_rest",
                        }:
                            row[key] = int(row[key])
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


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
) -> dict[str, Any]:
    """Train all grid combos; save the lowest validation-log-loss model."""
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total = (
        len(MAX_DEPTHS)
        * len(LEARNING_RATES)
        * len(MAX_ITERS)
        * len(LABEL_SMOOTHINGS)
        * len(VALIDATION_FRACTIONS)
    )
    trial_index = 0

    for max_depth in MAX_DEPTHS:
        for learning_rate in LEARNING_RATES:
            for max_iter in MAX_ITERS:
                for label_smoothing in LABEL_SMOOTHINGS:
                    for validation_fraction in VALIDATION_FRACTIONS:
                        trial_index += 1
                        train_rows, valid_rows = ResidualMLDatasetBuilder.time_split(
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

    if best is None:
        raise ValueError("Hyperparameter sweep produced no trials")

    train_rows, valid_rows = ResidualMLDatasetBuilder.time_split(
        rows,
        validation_fraction=best["validation_fraction"],
    )
    best_trainer = ResidualMLTrainer(
        market_weight=market_weight,
        dc_weight=dc_weight,
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        max_iter=best["max_iter"],
        label_smoothing=best["label_smoothing"],
    )
    final_result = best_trainer.fit(train_rows, valid_rows)
    saved = best_trainer.save(output_dir, version="sweep_best")
    best.update(
        {
            "train_log_loss": final_result.train_log_loss,
            "validation_log_loss": final_result.validation_log_loss,
            "market_validation_log_loss": final_result.market_validation_log_loss,
            "blend_validation_log_loss": final_result.blend_validation_log_loss,
            "model_path": str(saved.model_path),
        }
    )

    sweep_path = output_dir / "sweep_results.json"
    sweep_path.write_text(
        json.dumps({"best": best, "trials": trials}, indent=2),
        encoding="utf-8",
    )
    print(f"Best params: {best}", flush=True)
    print(f"Wrote sweep results to {sweep_path}", flush=True)
    print(f"Model saved to {saved.model_path}", flush=True)
    return best


def train_single(
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
) -> None:
    train_rows, valid_rows = ResidualMLDatasetBuilder.time_split(
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
    )
    result = trainer.fit(train_rows, valid_rows)
    saved = trainer.save(output_dir, version=version)
    print(f"Train rows: {result.train_rows}, validation rows: {result.validation_rows}")
    print(f"Train log loss: {result.train_log_loss:.4f}")
    print(f"Validation log loss: {result.validation_log_loss:.4f}")
    if result.market_validation_log_loss is not None:
        print(f"Market validation log loss: {result.market_validation_log_loss:.4f}")
    if result.blend_validation_log_loss is not None:
        print(f"Blend validation log loss: {result.blend_validation_log_loss:.4f}")
    print(f"Model saved to {saved.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/residual_ml/v1"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search hyperparameters and save the best model",
    )
    parser.add_argument(
        "--market-only-baseline",
        action="store_true",
        help="Train residuals vs market only (blend := market_norm; weights 1/0)",
    )
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output_dir)
    default_output = Path("models/residual_ml/v1")
    if args.output_dir == default_output:
        if args.market_only_baseline:
            output_dir = resolve_path(Path("models/residual_ml/market_only"))
        elif args.sweep:
            output_dir = resolve_path(Path("models/residual_ml/sweep_best"))

    config = DataSourceConfig()
    rows = load_dataset_csv(dataset_path)
    if not rows:
        raise SystemExit(f"No rows loaded from {dataset_path}")

    print(f"Loaded {len(rows)} rows from {dataset_path}", flush=True)

    if args.market_only_baseline:
        apply_market_only_baseline(rows)
        market_weight = 1.0
        dc_weight = 0.0
        print("Using market-only baseline (blend := market_norm)", flush=True)
    else:
        market_weight = config.residual_ml_market_weight
        dc_weight = config.residual_ml_dc_weight

    if args.sweep:
        run_hyperparameter_sweep(
            rows,
            market_weight=market_weight,
            dc_weight=dc_weight,
            output_dir=output_dir,
        )
    else:
        train_single(
            rows,
            market_weight=market_weight,
            dc_weight=dc_weight,
            output_dir=output_dir,
            version=args.version,
            validation_fraction=args.validation_fraction,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            label_smoothing=args.label_smoothing,
        )


if __name__ == "__main__":
    main()
