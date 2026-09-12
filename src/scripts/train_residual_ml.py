#!/usr/bin/env python3
"""Train the residual 1X2 ML model from a prebuilt dataset CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.calc.residual_ml import (
    load_dataset_rows,
    run_hyperparameter_sweep,
    train_and_save,
)
from src.utils.repo_paths import resolve_repo_path
from src.utils.time_split import DEFAULT_VALIDATION_FRACTION


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
        "--exclude-injury-features",
        action="store_true",
        help="Omit injury/availability columns from HGB training (ablation B1)",
    )
    parser.add_argument(
        "--injury-counts-only",
        action="store_true",
        help=(
            "Among injury columns, keep only unavailable counts and has_availability; "
            "drop the other INJURY_FEATURE_COLUMNS. Ignored if --exclude-injury-features."
        ),
    )
    parser.add_argument(
        "--train-recency-half-life-days",
        type=float,
        default=None,
        help=(
            "Exponential decay half-life in days for train sample weights "
            "(as-of = max train match_date). Default: unweighted."
        ),
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)
    default_output = Path("models/residual_ml/v1")
    if args.output_dir == default_output and args.sweep:
        output_dir = resolve_repo_path(Path("models/residual_ml/sweep_best"))

    rows = load_dataset_rows(dataset_path)
    if not rows:
        raise SystemExit(f"No rows loaded from {dataset_path}")

    print(f"Loaded {len(rows)} rows from {dataset_path}", flush=True)

    if args.sweep:
        run_hyperparameter_sweep(
            rows,
            output_dir=output_dir,
            validation_fraction=args.validation_fraction,
            exclude_injury_features=args.exclude_injury_features,
        )
    else:
        train_and_save(
            rows,
            output_dir=output_dir,
            version=args.version,
            validation_fraction=args.validation_fraction,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            label_smoothing=args.label_smoothing,
            exclude_injury_features=args.exclude_injury_features,
            injury_counts_only=args.injury_counts_only,
            train_recency_half_life_days=args.train_recency_half_life_days,
        )


if __name__ == "__main__":
    main()
