#!/usr/bin/env python3
"""Score log-loss components for one Stryktipset draw from dataset.csv.

No database required: filters ``data/residual_ml/dataset.csv`` by ``draw_number``
and reports market / DC / blend (and optional ML) multiclass LL plus binary
draw LL/Brier.

```bash
python -m src.scripts.monitor_round_ll --draw-number 4950
python -m src.scripts.monitor_round_ll --draw-number 4950 --model models/residual_ml/sweep_best/model.pkl
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml import load_dataset_rows
from src.calc.residual_ml.evaluation import score_baseline_log_losses, score_outcome_metrics
from src.calc.residual_ml.trainer import ResidualMLTrainer
from src.utils.repo_paths import resolve_repo_path

LABEL_TO_INDEX = {"1": 0, "X": 1, "2": 2}


def _format_loss(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def filter_rows_for_draw(
    rows: list[dict[str, Any]],
    draw_number: int,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("draw_number")
        if raw in (None, ""):
            continue
        try:
            if int(float(raw)) == draw_number:
                matched.append(row)
        except (TypeError, ValueError):
            continue
    return matched


def score_ml_log_loss(
    rows: list[dict[str, Any]],
    trainer: ResidualMLTrainer,
) -> tuple[float | None, int]:
    y_true: list[int] = []
    y_prob: list[list[float]] = []
    for row in rows:
        label = str(row.get("label", "")).strip().upper()
        if label not in LABEL_TO_INDEX:
            continue
        probs = trainer.predict_match_proba(row)
        if probs is None:
            continue
        y_true.append(LABEL_TO_INDEX[label])
        y_prob.append([probs["1"], probs["X"], probs["2"]])
    if not y_true:
        return None, 0
    return multiclass_log_loss(y_true, y_prob), len(y_true)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-number", type=int, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ResidualMLTrainer pickle for ML LL on the draw",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    rows = filter_rows_for_draw(load_dataset_rows(dataset_path), args.draw_number)
    print(
        f"Draw {args.draw_number}: {len(rows)} rows from {dataset_path}",
        flush=True,
    )
    if not rows:
        print("No rows for draw; nothing to score", flush=True)
        return

    baselines = score_baseline_log_losses(rows)
    blend_outcome = score_outcome_metrics(
        rows, ("p_home_blend", "p_draw_blend", "p_away_blend")
    )
    payload: dict[str, Any] = {
        "draw_number": args.draw_number,
        "row_count": len(rows),
        "baselines": {
            name: {"log_loss": loss, "row_count": count}
            for name, (loss, count) in baselines.items()
        },
        "blend_outcome": blend_outcome,
        "notes": (
            "Scores settled labels in dataset.csv for this draw_number. "
            "Market probs are ST odds stored on the match (often start odds). "
            "Injury features use snapshot_at <= kickoff when present."
        ),
    }

    for name, (loss, count) in baselines.items():
        print(
            f"  {name}: log_loss={_format_loss(loss)} ({count} rows)",
            flush=True,
        )
    print(
        f"  blend draw_ll={_format_loss(blend_outcome.get('draw_log_loss'))} "
        f"draw_brier={_format_loss(blend_outcome.get('draw_brier'))} "
        f"home_ll={_format_loss(blend_outcome.get('home_log_loss'))} "
        f"away_ll={_format_loss(blend_outcome.get('away_log_loss'))}",
        flush=True,
    )

    if args.model is not None:
        model_path = resolve_repo_path(args.model)
        trainer = ResidualMLTrainer.load(model_path)
        ml_loss, ml_count = score_ml_log_loss(rows, trainer)
        payload["ml"] = {"log_loss": ml_loss, "row_count": ml_count}
        print(f"  ml: log_loss={_format_loss(ml_loss)} ({ml_count} rows)", flush=True)

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
