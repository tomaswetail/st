#!/usr/bin/env python3
"""Score official 518 + locked injury slice at global shrink α (trial gate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml import (
    ResidualMLModel,
    load_dataset_rows,
    select_backtest_rows,
    shrink_toward_market,
)
from src.calc.residual_ml.evaluation import (
    LABEL_TO_INDEX,
    slice_rows_by_availability,
    slice_rows_by_unavailable_count_differential,
)
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path


def _market_probs(row: dict[str, Any]) -> dict[str, float] | None:
    keys = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return {
        "1": float(row["p_home_market_norm"]),
        "X": float(row["p_draw_market_norm"]),
        "2": float(row["p_away_market_norm"]),
    }


def score_global_shrink(
    rows: list[dict[str, Any]],
    trainer: Any,
    *,
    alpha: float,
) -> dict[str, Any]:
    vectors: list[list[float]] = []
    blend_vectors: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        ml_probs = trainer.predict_match_proba(row)
        market = _market_probs(row)
        if ml_probs is None or market is None:
            continue
        shrunk = shrink_toward_market(ml_probs, market, alpha=alpha)
        vectors.append([shrunk["1"], shrunk["X"], shrunk["2"]])
        labels.append(LABEL_TO_INDEX[row["label"]])
        blend_keys = ("p_home_blend", "p_draw_blend", "p_away_blend")
        if all(row.get(key) not in (None, "") for key in blend_keys):
            blend_vectors.append([float(row[key]) for key in blend_keys])
    return {
        "row_count": len(labels),
        "shrink_log_loss": (
            multiclass_log_loss(labels, vectors) if labels else None
        ),
        "blend_log_loss": (
            multiclass_log_loss(labels[: len(blend_vectors)], blend_vectors)
            if blend_vectors and len(blend_vectors) == len(labels)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--trial", default="")
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    model_path = resolve_repo_path(args.model)
    all_rows = load_dataset_rows(dataset_path)
    official_rows, filter_description = select_backtest_rows(
        all_rows,
        validation_fraction=0.20,
        max_draw=None,
    )
    config = DataSourceConfig(
        residual_ml_enabled=True,
        residual_ml_model_path=model_path,
    )
    model = ResidualMLModel.load(model_path, config=config)
    if model is None:
        raise SystemExit(f"Model not found: {model_path}")
    alpha = float(config.residual_ml_final_shrink_to_market)
    availability = slice_rows_by_availability(official_rows) or {}
    injury_slice = slice_rows_by_unavailable_count_differential(official_rows)
    payload = {
        "trial": args.trial,
        "filter": filter_description,
        "dataset": str(dataset_path),
        "model": str(model_path),
        "shrink_alpha": alpha,
        "official_518": score_global_shrink(
            official_rows, model.trainer, alpha=alpha
        ),
        "injury_slice": {
            "n": len(injury_slice),
            **score_global_shrink(injury_slice, model.trainer, alpha=alpha),
        },
        "availability:0": score_global_shrink(
            availability.get("availability:0", []),
            model.trainer,
            alpha=alpha,
        ),
        "availability:1": score_global_shrink(
            availability.get("availability:1", []),
            model.trainer,
            alpha=alpha,
        ),
        "production_unchanged": True,
    }
    json_path = resolve_repo_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    official = payload["official_518"]
    print(
        f"Official 518: n={official['row_count']} "
        f"shrink@{alpha:g}={official['shrink_log_loss']:.4f} "
        f"blend={official['blend_log_loss']:.4f}",
        flush=True,
    )
    print(
        f"Injury slice: n={payload['injury_slice']['n']} "
        f"shrink={payload['injury_slice']['shrink_log_loss']:.4f}",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
