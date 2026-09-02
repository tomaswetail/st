#!/usr/bin/env python3
"""Train the residual 1X2 ML model from a prebuilt dataset CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from calc.residual_ml import (
    apply_market_only_baseline,
    load_dataset_rows,
    run_hyperparameter_sweep,
    train_and_save,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.repo_paths import resolve_repo_path
from utils.time_split import DEFAULT_VALIDATION_FRACTION


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
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
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

    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    default_output = Path("models/residual_ml/v1")
    if args.output_dir == default_output:
        if args.market_only_baseline:
            output_dir = resolve_repo_path(Path("models/residual_ml/market_only"))
        elif args.sweep:
            output_dir = resolve_repo_path(Path("models/residual_ml/sweep_best"))

    config = DataSourceConfig()
    rows = load_dataset_rows(dataset_path)
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
            validation_fraction=args.validation_fraction,
        )
    else:
        train_and_save(
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
