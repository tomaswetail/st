#!/usr/bin/env python3
"""Backtest market, DC, blend, and ML probabilities on a dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import log_loss

from calc.residual_ml_model import ResidualMLModel
from objects.schema.data_classes.data_sources import DataSourceConfig


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prob_vector(row: dict, keys: tuple[str, str, str]) -> list[float] | None:
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return [float(row[key]) for key in keys]


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
        default=Path("models/residual_ml/v1/model.pkl"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    label_map = {"1": 0, "X": 1, "2": 2}
    y_true = [label_map[row["label"]] for row in rows]

    baselines = {
        "market": ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm"),
        "dc": ("p_home_dc_norm", "p_draw_dc_norm", "p_away_dc_norm"),
        "blend": ("p_home_blend", "p_draw_blend", "p_away_blend"),
    }
    for name, keys in baselines.items():
        vectors = [_prob_vector(row, keys) for row in rows]
        valid_indices = [index for index, vec in enumerate(vectors) if vec is not None]
        if not valid_indices:
            print(f"{name}: no rows")
            continue
        loss = log_loss(
            [y_true[index] for index in valid_indices],
            [vectors[index] for index in valid_indices],
        )
        print(f"{name} log loss: {loss:.4f} ({len(valid_indices)} rows)")

    config = DataSourceConfig(residual_ml_enabled=True, residual_ml_model_path=args.model)
    model = ResidualMLModel.load(args.model, config=config)
    if model is None:
        print("ML model not found; skipping ML backtest")
        return

    ml_vectors = []
    ml_labels = []
    for row in rows:
        probs = model.trainer.predict_match_proba(row)
        if probs is None:
            continue
        ml_vectors.append([probs["1"], probs["X"], probs["2"]])
        ml_labels.append(label_map[row["label"]])
    if not ml_vectors:
        print("ml: no rows")
        return
    print(f"ml log loss: {log_loss(ml_labels, ml_vectors):.4f} ({len(ml_labels)} rows)")


if __name__ == "__main__":
    main()
