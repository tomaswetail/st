#!/usr/bin/env python3
"""Score post-hoc 1X2 calibration on expanding-year validate halves.

Fits temperature and/or isotonic on the earlier chronological half of each
validate year. Scores the later half only. Residual ML stays off. No model
files are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.calc.residual_ml.calibration import evaluate_calibration_on_folds
from src.calc.residual_ml.folds import expanding_year_folds
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
        "--methods",
        choices=("temperature", "isotonic", "both"),
        default="both",
    )
    parser.add_argument(
        "--source",
        choices=("market",),
        default="market",
        help="Probability columns to calibrate (default: p_*_market_norm)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write per-fold metrics and accept bits to this path",
    )
    return parser.parse_args(argv)


def _methods(choice: str) -> list[str]:
    if choice == "both":
        return ["temperature", "isotonic"]
    return [choice]


def _format_n(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _format_row(
    name: str,
    n_calib: float,
    n_score: float,
    raw_ll: float,
    cal_ll: float,
    delta_ll: float,
    raw_ece: float,
    cal_ece: float,
) -> str:
    return (
        f"{name:<16} {_format_n(n_calib):>8} {_format_n(n_score):>8} "
        f"{raw_ll:8.4f} {cal_ll:8.4f} {delta_ll:8.4f} "
        f"{raw_ece:8.4f} {cal_ece:8.4f}"
    )


def _print_method_table(result: Any) -> None:
    print(f"\n[{result.method}]", flush=True)
    header = (
        f"{'fold':<16} {'n_calib':>8} {'n_score':>8} "
        f"{'raw_ll':>8} {'cal_ll':>8} {'delta_ll':>8} "
        f"{'raw_ece':>8} {'cal_ece':>8}"
    )
    print(header, flush=True)
    for fold_score in result.per_fold:
        print(
            _format_row(
                fold_score.fold_name,
                fold_score.n_calib,
                fold_score.n_score,
                fold_score.raw_ll,
                fold_score.cal_ll,
                fold_score.delta_ll,
                fold_score.raw_ece,
                fold_score.cal_ece,
            ),
            flush=True,
        )
    print(
        _format_row(
            "mean",
            result.mean_n_calib,
            result.mean_n_score,
            result.mean_raw_ll,
            result.mean_cal_ll,
            result.mean_delta_ll,
            result.mean_raw_ece,
            result.mean_cal_ece,
        ),
        flush=True,
    )
    verdict = "ACCEPT" if result.accepted else "REJECT"
    print(
        f"{verdict}  mean calibrated log loss {result.mean_cal_ll:.4f} "
        f"{'<' if result.accepted else '>='} "
        f"mean raw log loss {result.mean_raw_ll:.4f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    rows = load_dataset_rows(dataset_path)
    folds = expanding_year_folds(rows)
    json_payload: dict[str, Any] = {
        "dataset": str(dataset_path),
        "source": args.source,
        "methods": {},
    }

    if not folds:
        print("no folds", flush=True)
        if args.json is not None:
            json_path = resolve_repo_path(args.json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
            print(f"\nWrote JSON report to {json_path}", flush=True)
        return 0

    for method in _methods(args.methods):
        result = evaluate_calibration_on_folds(folds, method)
        _print_method_table(result)
        json_payload["methods"][method] = {
            "accepted": result.accepted,
            "mean_raw_ll": result.mean_raw_ll,
            "mean_cal_ll": result.mean_cal_ll,
            "mean_delta_ll": result.mean_delta_ll,
            "mean_raw_ece": result.mean_raw_ece,
            "mean_cal_ece": result.mean_cal_ece,
            "mean_raw_brier": result.mean_raw_brier,
            "mean_cal_brier": result.mean_cal_brier,
            "folds": [asdict(fold_score) for fold_score in result.per_fold],
        }

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
