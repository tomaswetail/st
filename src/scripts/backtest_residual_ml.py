#!/usr/bin/env python3
"""Backtest market baseline and ML predictions (with optional market shrink) on a dataset.

By default evaluates only the time-split validation slice (same ordering as
training) so train rows are not scored. Holdout draws (4951–4960) are excluded
by default; pass --include-holdout for Phase 5. Pass --all-rows to score the
full CSV (optimistic / contaminated).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.calc.residual_ml import (
    ResidualMLModel,
    load_dataset_rows,
    resolve_validation_fraction,
    run_backtest_scoring,
    score_baseline_log_losses,
    select_backtest_rows,
)
from config.eval_protocol import TUNING_DRAW_MAX
from src.objects.schema.data_classes.data_sources import DataSourceConfig
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
        "--model",
        type=Path,
        default=Path("models/residual_ml/sweep_best/model.pkl"),
    )
    parser.add_argument(
        "--final-shrink",
        type=float,
        default=None,
        help="If set, evaluate raw ML (α=0) and this shrink α only",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Score every loaded row (includes train; optimistic)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "Time-split validation fraction (same as train). "
            f"Default {DEFAULT_VALIDATION_FRACTION} unless --all-rows"
        ),
    )
    parser.add_argument(
        "--min-draw",
        type=int,
        default=None,
        help="Keep only rows with draw_number >= this value",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=None,
        help="Keep only rows with draw_number <= this value",
    )
    parser.add_argument(
        "--exclude-holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"Exclude final holdout draws (>{TUNING_DRAW_MAX}; default). "
            "Use --no-exclude-holdout or --include-holdout for Phase 5."
        ),
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Include holdout draws (Phase 5 only; same as --no-exclude-holdout)",
    )
    args = parser.parse_args()

    if args.all_rows and args.validation_fraction is not None:
        parser.error("Use either --all-rows or --validation-fraction, not both")
    if args.include_holdout and args.max_draw is not None:
        parser.error("Use either --include-holdout or --max-draw, not both")

    exclude_holdout = args.exclude_holdout and not args.include_holdout
    max_draw = args.max_draw
    if exclude_holdout and not args.all_rows and max_draw is None:
        max_draw = TUNING_DRAW_MAX

    dataset_path = resolve_repo_path(args.dataset)
    model_path = resolve_repo_path(args.model)
    validation_fraction = resolve_validation_fraction(
        cli_fraction=args.validation_fraction,
        model_path=model_path,
    )
    if args.validation_fraction is None:
        sweep_path = model_path.parent / "sweep_results.json"
        if sweep_path.exists():
            print(
                f"Using validation_fraction={validation_fraction:g} from {sweep_path}",
                flush=True,
            )
    all_rows = load_dataset_rows(dataset_path)
    rows, filter_description = select_backtest_rows(
        all_rows,
        all_rows=args.all_rows,
        validation_fraction=None if args.all_rows else validation_fraction,
        min_draw=args.min_draw,
        max_draw=max_draw,
    )
    holdout_note = ""
    if exclude_holdout and not args.all_rows and args.max_draw is None:
        holdout_note = f", holdout excluded (>{TUNING_DRAW_MAX})"
    print(
        f"Backtest filter: {filter_description}{holdout_note} "
        f"({len(rows)}/{len(all_rows)} rows)",
        flush=True,
    )
    if not rows:
        print("No rows after filter; nothing to score")
        return

    for name, (loss, count) in score_baseline_log_losses(rows).items():
        if loss is None:
            print(f"{name}: no rows")
        else:
            print(f"{name} log loss: {loss:.4f} ({count} rows)")

    config = DataSourceConfig(
        residual_ml_enabled=True,
        residual_ml_model_path=model_path,
    )
    model = ResidualMLModel.load(model_path, config=config)
    if model is None:
        print("ML model not found; skipping ML backtest")
        return

    scoring = run_backtest_scoring(
        rows,
        model.trainer,
        final_shrink=args.final_shrink,
    )

    if scoring.ml_row_count == 0:
        print("ml: no rows")
        return

    print(
        f"market log loss (ML rows): {scoring.market_on_ml_rows:.4f} "
        f"({scoring.ml_row_count} rows)"
    )

    for alpha, loss in scoring.shrink_results:
        label = "ml" if alpha <= 0 else f"ml_shrink α={alpha:.1f}"
        print(f"{label} log loss: {loss:.4f} ({scoring.ml_row_count} rows)")

    beats_market = (
        scoring.best_loss is not None
        and scoring.market_on_ml_rows is not None
        and scoring.best_loss < scoring.market_on_ml_rows
    )
    print(
        f"best shrink α={scoring.best_alpha:.1f} log loss: {scoring.best_loss:.4f} "
        f"({'beats' if beats_market else 'does not beat'} market "
        f"{scoring.market_on_ml_rows:.4f})"
    )
    if scoring.best_alpha is not None and scoring.best_alpha > 0:
        print(
            f"Set RESIDUAL_ML_FINAL_SHRINK_TO_MARKET={scoring.best_alpha} for live inference"
        )


if __name__ == "__main__":
    main()
