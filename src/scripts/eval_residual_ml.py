#!/usr/bin/env python3
"""Multi-slice evaluation for residual ML (pooled + year/quarter/league slices).

By default scores the time-split validation slice and excludes final holdout draws
(draw_number > TUNING_DRAW_MAX). Use --include-holdout for Phase 5 evaluation only.

Also reports binary draw/home/away metrics and optional Phase 3 ablation A–D.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from calc.residual_ml import (
    ResidualMLModel,
    is_market_only_weights,
    load_dataset_rows,
    resolve_validation_fraction,
    run_backtest_scoring,
    select_backtest_rows,
)
from calc.residual_ml.evaluation import (
    BASELINE_KEY_GROUPS,
    SliceMetrics,
    build_multi_slice_report,
    run_phase3_ablation,
    score_outcome_metrics,
)
from config.eval_protocol import TUNING_DRAW_MAX, VALIDATION_FRACTION
from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.repo_paths import resolve_repo_path


def _format_loss(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _format_accuracy(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _print_slice_metrics(metrics: SliceMetrics) -> None:
    print(f"\n[{metrics.name}] rows={metrics.row_count}", flush=True)
    for baseline_name, (loss, count) in metrics.baselines.items():
        accuracy, accuracy_count = metrics.top_pick.get(baseline_name, (None, 0))
        print(
            f"  {baseline_name}: log_loss={_format_loss(loss)} ({count} rows) "
            f"top_pick={_format_accuracy(accuracy)} ({accuracy_count} rows)",
            flush=True,
        )
    if metrics.ml_log_loss is not None:
        print(
            f"  ml: log_loss={metrics.ml_log_loss:.4f} "
            f"market_on_ml_rows={_format_loss(metrics.market_on_ml_rows)}",
            flush=True,
        )
    if metrics.best_shrink_log_loss is not None:
        print(
            f"  ml_best_shrink: alpha={metrics.best_shrink_alpha:.1f} "
            f"log_loss={metrics.best_shrink_log_loss:.4f}",
            flush=True,
        )


def _print_outcome_metrics(name: str, outcome: dict[str, Any]) -> None:
    print(f"\n[outcome:{name}] rows={outcome.get('row_count', 0)}", flush=True)
    print(
        f"  draw_ll={_format_loss(outcome.get('draw_log_loss'))} "
        f"draw_brier={_format_loss(outcome.get('draw_brier'))} "
        f"home_ll={_format_loss(outcome.get('home_log_loss'))} "
        f"away_ll={_format_loss(outcome.get('away_log_loss'))} "
        f"pooled_ll={_format_loss(outcome.get('pooled_multiclass_log_loss'))}",
        flush=True,
    )


def _metrics_to_dict(metrics: SliceMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["baselines"] = {
        name: {"log_loss": loss, "row_count": count}
        for name, (loss, count) in metrics.baselines.items()
    }
    payload["top_pick"] = {
        name: {"accuracy": accuracy, "row_count": count}
        for name, (accuracy, count) in metrics.top_pick.items()
    }
    return payload


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
        "--validation-fraction",
        type=float,
        default=None,
        help=f"Time-split validation fraction (default {VALIDATION_FRACTION})",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Include final holdout draws (Phase 5 only)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run Phase 3 ablation rows A–D (blend / draw adj / HGB / shrink)",
    )
    parser.add_argument(
        "--shrink-alpha",
        type=float,
        default=0.3,
        help="Shrink alpha for ablation row D (default 0.3)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write structured metrics JSON to this path",
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    model_path = resolve_repo_path(args.model)
    validation_fraction = resolve_validation_fraction(
        cli_fraction=args.validation_fraction,
        model_path=model_path,
    )

    all_rows = load_dataset_rows(dataset_path)
    max_draw = None if args.include_holdout else TUNING_DRAW_MAX
    rows, filter_description = select_backtest_rows(
        all_rows,
        validation_fraction=validation_fraction,
        max_draw=max_draw,
    )
    holdout_note = "" if args.include_holdout else f", holdout excluded (>{TUNING_DRAW_MAX})"
    print(
        f"Eval filter: {filter_description}{holdout_note} "
        f"({len(rows)}/{len(all_rows)} rows)",
        flush=True,
    )
    if not rows:
        print("No rows after filter; nothing to score")
        return

    config = DataSourceConfig(
        residual_ml_enabled=True,
        residual_ml_model_path=model_path,
    )
    model = ResidualMLModel.load(model_path, config=config)
    scoring = None
    trainer = None
    if model is not None:
        trainer = model.trainer
        use_market_only = is_market_only_weights(
            model.trainer.market_weight,
            model.trainer.dc_weight,
        )
        scoring = run_backtest_scoring(
            rows,
            model.trainer,
            use_market_only_baseline=use_market_only,
        )
    else:
        print("ML model not found; reporting baseline slices only", flush=True)

    report = build_multi_slice_report(rows, scoring)
    for key in sorted(report.keys(), key=lambda name: (name != "pooled", name)):
        _print_slice_metrics(report[key])

    outcome_by_baseline = {
        name: score_outcome_metrics(rows, keys)
        for name, keys in BASELINE_KEY_GROUPS.items()
    }
    for name, outcome in outcome_by_baseline.items():
        _print_outcome_metrics(name, outcome)

    ablation = None
    if args.ablation:
        ablation = run_phase3_ablation(
            rows,
            trainer,
            shrink_alpha=args.shrink_alpha,
        )
        print("\n[ablation A–D]", flush=True)
        for name, metrics in ablation["rows"].items():
            print(
                f"  {name}: pooled_ll={_format_loss(metrics.get('pooled_log_loss'))} "
                f"draw_ll={_format_loss(metrics.get('draw_log_loss'))} "
                f"draw_brier={_format_loss(metrics.get('draw_brier'))} "
                f"home_ll={_format_loss(metrics.get('home_log_loss'))} "
                f"away_ll={_format_loss(metrics.get('away_log_loss'))} "
                f"n={metrics.get('row_count')}",
                flush=True,
            )

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "filter": filter_description + holdout_note,
            "validation_fraction": validation_fraction,
            "row_count": len(rows),
            "slices": {
                name: _metrics_to_dict(metrics) for name, metrics in report.items()
            },
            "outcome_metrics": outcome_by_baseline,
            "pipeline_order": "conditional blend → draw adjust → HGB",
        }
        if ablation is not None:
            payload["ablation"] = ablation
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {json_path}", flush=True)


if __name__ == "__main__":
    main()
