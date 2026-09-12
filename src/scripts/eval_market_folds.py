#!/usr/bin/env python3
"""Score market-only 1X2 probabilities on chronological calendar-year folds.

No model is loaded or trained. Residual ML stays off.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.calc.residual_ml.folds import (
    Fold,
    FoldScores,
    Metrics,
    expanding_year_folds,
    rolling_year_folds,
    score_folds,
    score_market_only,
)
from src.calc.residual_ml.io import load_dataset_rows
from src.utils.repo_paths import resolve_repo_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/fixtures_dataset.csv"),
    )
    parser.add_argument(
        "--mode",
        choices=("expanding", "rolling", "both"),
        default="both",
    )
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=2,
        help="Rolling train window in calendar years (default 2)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write per-fold metrics and unweighted means to this path",
    )
    return parser.parse_args(argv)


def _format_n(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _print_fold_table(mode: str, scores: FoldScores) -> None:
    print(f"\n[{mode}]", flush=True)
    header = (
        f"{'fold':<16} {'n_val':>6} {'logloss':>9} {'brier':>9} "
        f"{'rps':>9} {'acc':>8} {'ece':>8}"
    )
    print(header, flush=True)
    for fold, metrics in scores.per_fold:
        print(_format_metrics_row(fold.name, metrics), flush=True)
    print(_format_metrics_row("mean", scores.mean), flush=True)


def _format_metrics_row(name: str, metrics: Metrics) -> str:
    return (
        f"{name:<16} {_format_n(metrics.n):>6} {metrics.log_loss:9.4f} "
        f"{metrics.brier:9.4f} {metrics.rps:9.4f} {metrics.accuracy:8.4f} "
        f"{metrics.ece:8.4f}"
    )


def _fold_payload(fold: Fold, metrics: Metrics) -> dict[str, Any]:
    return {
        "name": fold.name,
        "train_years": fold.train_years,
        "validate_year": fold.validate_year,
        "metrics": asdict(metrics),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    rows = load_dataset_rows(dataset_path)
    json_payload: dict[str, Any] = {
        "dataset": str(dataset_path),
        "train_window_years": args.train_window_years,
        "modes": {},
    }

    modes: list[str] = (
        ["expanding", "rolling"] if args.mode == "both" else [args.mode]
    )
    for mode in modes:
        if mode == "expanding":
            folds = expanding_year_folds(rows)
        else:
            folds = rolling_year_folds(
                rows, train_window_years=args.train_window_years
            )
        if not folds:
            print(f"\n[{mode}] no folds", flush=True)
            json_payload["modes"][mode] = {"folds": [], "mean": None}
            continue
        scores = score_folds(folds, score_market_only)
        _print_fold_table(mode, scores)
        json_payload["modes"][mode] = {
            "folds": [
                _fold_payload(fold, metrics) for fold, metrics in scores.per_fold
            ],
            "mean": asdict(scores.mean),
        }

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
