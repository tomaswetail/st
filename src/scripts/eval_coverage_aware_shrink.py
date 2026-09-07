#!/usr/bin/env python3
"""Eval-only coverage-aware post-ML shrink (T4). Does not change production scoring."""

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
    shrink_toward_market_by_coverage,
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


def _log_loss_for_vectors(
    labels: list[int],
    vectors: list[list[float]],
) -> float | None:
    if not labels:
        return None
    return multiclass_log_loss(labels, vectors)


def score_shrink_variants(
    rows: list[dict[str, Any]],
    trainer: Any,
    *,
    ship_alpha: float,
) -> dict[str, Any]:
    ship_vectors: list[list[float]] = []
    t4_vectors: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        ml_probs = trainer.predict_match_proba(row)
        market = _market_probs(row)
        if ml_probs is None or market is None:
            continue
        ship = shrink_toward_market(ml_probs, market, alpha=ship_alpha)
        t4 = shrink_toward_market_by_coverage(
            ml_probs, market, row.get("has_availability")
        )
        ship_vectors.append([ship["1"], ship["X"], ship["2"]])
        t4_vectors.append([t4["1"], t4["X"], t4["2"]])
        labels.append(LABEL_TO_INDEX[row["label"]])
    return {
        "row_count": len(labels),
        "ship_shrink_log_loss": _log_loss_for_vectors(labels, ship_vectors),
        "t4_shrink_log_loss": _log_loss_for_vectors(labels, t4_vectors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/dc_rho_mle_promotion/dataset.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Production HGB path (do not write sweep_best from this script)",
    )
    parser.add_argument("--json", type=Path, required=True)
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
    trainer = model.trainer
    ship_alpha = float(config.residual_ml_final_shrink_to_market)

    availability = slice_rows_by_availability(official_rows) or {}
    injury_slice = slice_rows_by_unavailable_count_differential(official_rows)
    payload: dict[str, Any] = {
        "filter": filter_description,
        "dataset": str(dataset_path),
        "model": str(model_path),
        "ship_alpha": ship_alpha,
        "t4_alphas": {"has_availability==1": 0.7, "has_availability!=1": 0.85},
        "official_518": score_shrink_variants(
            official_rows, trainer, ship_alpha=ship_alpha
        ),
        "injury_slice": {
            "n": len(injury_slice),
            **score_shrink_variants(injury_slice, trainer, ship_alpha=ship_alpha),
        },
        "availability:0": score_shrink_variants(
            availability.get("availability:0", []),
            trainer,
            ship_alpha=ship_alpha,
        ),
        "availability:1": score_shrink_variants(
            availability.get("availability:1", []),
            trainer,
            ship_alpha=ship_alpha,
        ),
        "production_unchanged": True,
    }
    json_path = resolve_repo_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    official = payload["official_518"]
    print(
        f"Official 518: n={official['row_count']} "
        f"ship={official['ship_shrink_log_loss']:.4f} "
        f"t4={official['t4_shrink_log_loss']:.4f}",
        flush=True,
    )
    print(
        f"Injury slice: n={payload['injury_slice']['n']} "
        f"ship={payload['injury_slice']['ship_shrink_log_loss']:.4f} "
        f"t4={payload['injury_slice']['t4_shrink_log_loss']:.4f}",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
