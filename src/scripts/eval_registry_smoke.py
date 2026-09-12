#!/usr/bin/env python3
"""Smoke-compare registry backends on one chronological fold.

Market-only is always the first row. Does not persist models. Residual ML stays off.
``vs_market_ll`` = model log loss − market log loss (negative = better than market).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.calc.probability_metrics import mean_accuracy, mean_log_loss, multiclass_brier
from src.calc.residual_ml.folds import expanding_year_folds, score_market_only
from src.calc.residual_ml.io import load_dataset_rows
from src.calc.residual_ml.models import available_backends, get_model, is_backend_available
from src.calc.residual_ml.models.registry import BACKENDS
from src.calc.residual_ml.trainer import market_from_row
from src.utils.repo_paths import resolve_repo_path

SMOKE_TREE_ITER = 100
SMOKE_LOGISTIC_MAX_ITER = 200
ML_BACKENDS = ("logistic", "hist_gradient", "lightgbm", "catboost")
VALID_LABELS = frozenset({"1", "X", "2"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/fixtures_dataset.csv"),
    )
    parser.add_argument(
        "--fold",
        default="expanding-2023",
        help="One expanding-year fold name (default expanding-2023)",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Backends to fit (default: all available). market_baseline is always scored.",
    )
    parser.add_argument(
        "--families",
        choices=("direct", "residual", "both"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the smoke table to this path",
    )
    return parser.parse_args(argv)


def _families(choice: str) -> list[str]:
    if choice == "both":
        return ["direct", "residual"]
    return [choice]


def _score_predictions(
    rows: list[dict[str, Any]],
    model: Any,
) -> tuple[int, float, float, float]:
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        market = market_from_row(row)
        if market is None:
            continue
        label = str(row.get("label", "")).strip().upper()
        if label not in VALID_LABELS:
            continue
        predicted = model.predict_proba(row, market)
        labels.append(label)
        probability_rows.append((predicted["1"], predicted["X"], predicted["2"]))
    if not labels:
        raise ValueError("no scorable predicted rows")
    return (
        len(labels),
        mean_log_loss(labels, probability_rows),
        multiclass_brier(labels, probability_rows),
        mean_accuracy(labels, probability_rows),
    )


def _format_row(
    backend: str,
    family: str,
    n_train: int,
    n_val: int,
    logloss: float,
    brier: float,
    accuracy: float,
    vs_market_ll: float,
) -> str:
    return (
        f"{backend:<16} {family:<10} {n_train:>7} {n_val:>6} "
        f"{logloss:8.4f} {brier:8.4f} {accuracy:7.4f} {vs_market_ll:12.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    rows = load_dataset_rows(dataset_path)
    folds = expanding_year_folds(rows)
    fold = next((item for item in folds if item.name == args.fold), None)
    if fold is None:
        available = ", ".join(item.name for item in folds) or "(none)"
        print(
            f"Fold not found: {args.fold!r}. Available expanding folds: {available}",
            file=sys.stderr,
        )
        return 1

    requested = list(args.backends) if args.backends else list(available_backends())
    families = _families(args.families)
    table_rows: list[dict[str, Any]] = []

    market_metrics = score_market_only(fold.validate_rows)
    market_row = {
        "backend": "market",
        "family": "market",
        "n_train": len(fold.train_rows),
        "n_val": market_metrics.n,
        "logloss": market_metrics.log_loss,
        "brier": market_metrics.brier,
        "acc": market_metrics.accuracy,
        "vs_market_ll": 0.0,
    }
    table_rows.append(market_row)

    header = (
        f"{'backend':<16} {'family':<10} {'n_train':>7} {'n_val':>6} "
        f"{'logloss':>8} {'brier':>8} {'acc':>7} {'vs_market_ll':>12}"
    )
    print(header, flush=True)
    print(
        _format_row(
            market_row["backend"],
            market_row["family"],
            market_row["n_train"],
            market_row["n_val"],
            market_row["logloss"],
            market_row["brier"],
            market_row["acc"],
            market_row["vs_market_ll"],
        ),
        flush=True,
    )

    for backend in requested:
        if backend == "market_baseline":
            continue
        if backend not in BACKENDS and backend not in ML_BACKENDS:
            print(f"Skipping unknown backend {backend!r}", flush=True)
            continue
        if not is_backend_available(backend):
            print(
                f"Skipping {backend}: optional package is not installed",
                flush=True,
            )
            continue
        for family in families:
            print(f"Fitting {backend} / {family} ...", flush=True)
            iter_cap = (
                SMOKE_LOGISTIC_MAX_ITER if backend == "logistic" else SMOKE_TREE_ITER
            )
            model = get_model(
                backend,
                family,
                random_state=args.seed,
                max_iter=iter_cap,
                n_estimators=SMOKE_TREE_ITER,
                iterations=SMOKE_TREE_ITER,
            )
            model.fit(fold.train_rows)
            n_val, logloss, brier, accuracy = _score_predictions(
                fold.validate_rows,
                model,
            )
            row = {
                "backend": backend,
                "family": family,
                "n_train": len(fold.train_rows),
                "n_val": n_val,
                "logloss": logloss,
                "brier": brier,
                "acc": accuracy,
                "vs_market_ll": logloss - market_metrics.log_loss,
            }
            table_rows.append(row)
            print(
                _format_row(
                    row["backend"],
                    row["family"],
                    row["n_train"],
                    row["n_val"],
                    row["logloss"],
                    row["brier"],
                    row["acc"],
                    row["vs_market_ll"],
                ),
                flush=True,
            )

    print(
        "vs_market_ll = model logloss − market logloss (negative = better than market)",
        flush=True,
    )

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(dataset_path),
            "fold": fold.name,
            "train_years": fold.train_years,
            "validate_year": fold.validate_year,
            "seed": args.seed,
            "tree_iter": SMOKE_TREE_ITER,
            "logistic_max_iter": SMOKE_LOGISTIC_MAX_ITER,
            "rows": table_rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
