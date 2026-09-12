#!/usr/bin/env python3
"""Fold-by-fold compare of registry probability sources (uncalibrated).

Expanding calendar-year folds only. Market-only is scored with
``score_market_only`` (no model fit). Default tables are uncalibrated —
Phase 6 temperature ACCEPT was a 0.0003 mean edge and mixed by year.
Does not persist models. Residual ML stays off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.calc.residual_ml.folds import expanding_year_folds, score_market_only
from src.calc.residual_ml.io import load_dataset_rows
from src.calc.residual_ml.models import available_backends, is_backend_available
from src.calc.residual_ml.models.registry import BACKENDS
from src.calc.residual_ml.p13 import score_p13_on_packs
from src.calc.residual_ml.registry_eval import (
    SMOKE_LOGISTIC_MAX_ITER,
    SMOKE_TREE_ITER,
    build_registry_model,
    collect_scored_rows,
    families_from_choice,
    is_market_backend,
    metrics_from_predictions,
)
from src.utils.repo_paths import resolve_repo_path

ML_BACKENDS = ("logistic", "hist_gradient", "lightgbm", "catboost")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/fixtures_dataset.csv"),
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Backends to score (default: all available). market is always first.",
    )
    parser.add_argument(
        "--families",
        choices=("direct", "residual", "both"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--p13-n",
        type=int,
        default=128,
        dest="p13_n",
        help="MAX_P13 row_count for consecutive 13-packs (default 128)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for P13 packs (default: all CPUs)",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="After the table, print native+permutation importance on expanding-2023",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the compare table to this path",
    )
    return parser.parse_args(argv)


def _format_optional(value: float | None, width: int = 10) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:{width}.4f}"


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['backend']:<16} {row['family']:<10} {row['fold']:<16} "
        f"{int(row['n_val']):>6} "
        f"{row['logloss']:8.4f} {row['brier']:8.4f} {row['acc']:7.4f} "
        f"{row['ece']:7.4f} "
        f"{_format_optional(row.get('p13'))} "
        f"{_format_optional(row.get('covered13'))}"
    )


def _header(p13_n: int) -> str:
    return (
        f"{'backend':<16} {'family':<10} {'fold':<16} {'n_val':>6} "
        f"{'logloss':>8} {'brier':>8} {'acc':>7} {'ece':>7} "
        f"{f'p13@{p13_n}':>10} {f'covered13@{p13_n}':>10}"
    )


def _mean_row(rows: list[dict[str, Any]], *, backend: str, family: str) -> dict[str, Any]:
    keys = ("n_val", "logloss", "brier", "acc", "ece")
    averaged: dict[str, Any] = {
        "backend": backend,
        "family": family,
        "fold": "mean",
    }
    for key in keys:
        averaged[key] = sum(row[key] for row in rows) / len(rows)
    for key in ("p13", "covered13"):
        values = [row[key] for row in rows if row.get(key) is not None]
        averaged[key] = sum(values) / len(values) if values else None
    return averaged


def _score_market_fold(fold, p13_n: int, workers: int | None) -> dict[str, Any]:
    market_metrics = score_market_only(fold.validate_rows)
    scored_rows, _labels, probability_rows = collect_scored_rows(fold.validate_rows)
    p13, covered13, _n_packs = score_p13_on_packs(
        scored_rows, probability_rows, row_count=p13_n, workers=workers
    )
    return {
        "backend": "market",
        "family": "market",
        "fold": fold.name,
        "n_val": float(market_metrics.n),
        "logloss": market_metrics.log_loss,
        "brier": market_metrics.brier,
        "acc": market_metrics.accuracy,
        "ece": market_metrics.ece,
        "p13": p13,
        "covered13": covered13,
    }


def _score_model_fold(
    fold, model, *, backend: str, family: str, p13_n: int, workers: int | None
) -> dict[str, Any]:
    scored_rows, labels, probability_rows = collect_scored_rows(
        fold.validate_rows, predict=model
    )
    metrics = metrics_from_predictions(labels, probability_rows)
    p13, covered13, _n_packs = score_p13_on_packs(
        scored_rows, probability_rows, row_count=p13_n, workers=workers
    )
    return {
        "backend": backend,
        "family": family,
        "fold": fold.name,
        "n_val": metrics["n_val"],
        "logloss": metrics["logloss"],
        "brier": metrics["brier"],
        "acc": metrics["acc"],
        "ece": metrics["ece"],
        "p13": p13,
        "covered13": covered13,
    }


def _requested_ml_backends(args: argparse.Namespace) -> list[str]:
    requested = list(args.backends) if args.backends else list(available_backends())
    ml_backends: list[str] = []
    for backend in requested:
        if is_market_backend(backend):
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
        ml_backends.append(backend)
    return ml_backends


def _print_importance_smoke(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any] | None:
    from src.calc.residual_ml.importance import importance_from_registry_model
    from src.calc.residual_ml.models import is_backend_available

    folds = expanding_year_folds(rows)
    fold = next((item for item in folds if item.name == "expanding-2023"), None)
    if fold is None:
        print("Importance smoke: expanding-2023 not found", flush=True)
        return None
    backend = "catboost" if is_backend_available("catboost") else "hist_gradient"
    family = "direct"
    print(f"Importance smoke: fitting {backend}/{family} on {fold.name} ...", flush=True)
    model = build_registry_model(backend, family, seed=seed)
    model.fit(fold.train_rows)
    tables = importance_from_registry_model(
        model, fold.validate_rows, random_state=seed
    )

    def _print_table(title: str, items: list[tuple[str, float]] | None) -> None:
        print(title, flush=True)
        if not items:
            print("  (none / skipped)", flush=True)
            return
        for name, value in items[:15]:
            print(f"  {name:<40} {value:10.6f}", flush=True)

    _print_table("Top 15 native importances", tables.get("native"))
    _print_table("Top 15 permutation importances (log-loss increase)", tables.get("permutation"))
    _print_table("Top 15 SHAP-style importances", tables.get("shap"))
    if tables.get("shap") is None:
        print("SHAP skipped (native CatBoost/LightGBM contrib unavailable).", flush=True)
    return {
        "backend": backend,
        "family": family,
        "fold": fold.name,
        "native": tables.get("native"),
        "permutation": tables.get("permutation"),
        "shap": tables.get("shap"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    rows = load_dataset_rows(dataset_path)
    folds = expanding_year_folds(rows)
    if not folds:
        print("No expanding-year folds in dataset", file=sys.stderr)
        return 1

    families = families_from_choice(args.families)
    ml_backends = _requested_ml_backends(args)
    table_rows: list[dict[str, Any]] = []

    print(
        "Default tables are uncalibrated (Phase 6 temperature ACCEPT was a "
        "0.0003 mean edge and mixed by year).",
        flush=True,
    )
    print(_header(args.p13_n), flush=True)

    market_fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        row = _score_market_fold(fold, args.p13_n, args.workers)
        market_fold_rows.append(row)
        table_rows.append(row)
        print(_format_row(row), flush=True)
    market_mean = _mean_row(market_fold_rows, backend="market", family="market")
    table_rows.append(market_mean)
    print(_format_row(market_mean), flush=True)

    for backend in ml_backends:
        for family in families:
            fold_rows: list[dict[str, Any]] = []
            for fold in folds:
                print(f"Fitting {backend} / {family} / {fold.name} ...", flush=True)
                model = build_registry_model(backend, family, seed=args.seed)
                model.fit(fold.train_rows)
                row = _score_model_fold(
                    fold,
                    model,
                    backend=backend,
                    family=family,
                    p13_n=args.p13_n,
                    workers=args.workers,
                )
                fold_rows.append(row)
                table_rows.append(row)
                print(_format_row(row), flush=True)
            mean_row = _mean_row(fold_rows, backend=backend, family=family)
            table_rows.append(mean_row)
            print(_format_row(mean_row), flush=True)

    print(
        "Fold mean is unweighted. P13@N uses consecutive 13-packs on "
        "sorted validate rows; leftover <13 dropped. Live ML stays off.",
        flush=True,
    )

    importance_payload: dict[str, Any] | None = None
    if args.importance:
        importance_payload = _print_importance_smoke(rows, seed=args.seed)

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(dataset_path),
            "seed": args.seed,
            "p13_n": args.p13_n,
            "uncalibrated": True,
            "tree_iter": SMOKE_TREE_ITER,
            "logistic_max_iter": SMOKE_LOGISTIC_MAX_ITER,
            "rows": table_rows,
            "importance": importance_payload,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
