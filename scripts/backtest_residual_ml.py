#!/usr/bin/env python3
"""Backtest market, DC, blend, and ML probabilities on a dataset.

By default evaluates only the time-split validation slice (same ordering as
training) so train rows are not scored. Pass --all-rows to score the full CSV
(optimistic / contaminated).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import log_loss

from calc.residual_ml_baseline import (
    apply_market_only_baseline,
    is_market_only_weights,
    shrink_toward_market,
)
from calc.residual_ml_dataset import ResidualMLDatasetBuilder
from calc.residual_ml_model import ResidualMLModel
from objects.schema.data_classes.data_sources import DataSourceConfig

SHRINK_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MARKET_KEYS = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
DEFAULT_VALIDATION_FRACTION = 0.15


def repo_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "scripts":
        return cwd.parent
    return cwd


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {
                key: (None if value == "" else value)
                for key, value in raw.items()
                if key is not None
            }
            rows.append(row)
        return rows


def _draw_number(row: dict[str, Any]) -> int | None:
    value = row.get("draw_number")
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def filter_rows_by_draw(
    rows: list[dict[str, Any]],
    *,
    min_draw: int | None = None,
    max_draw: int | None = None,
) -> list[dict[str, Any]]:
    """Keep rows whose draw_number falls within an inclusive range."""
    if min_draw is None and max_draw is None:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        draw_number = _draw_number(row)
        if draw_number is None:
            continue
        if min_draw is not None and draw_number < min_draw:
            continue
        if max_draw is not None and draw_number > max_draw:
            continue
        filtered.append(row)
    return filtered


def select_backtest_rows(
    rows: list[dict[str, Any]],
    *,
    all_rows: bool = False,
    validation_fraction: float | None = None,
    min_draw: int | None = None,
    max_draw: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return rows to score and a short description of the filter applied."""
    draw_filtered = filter_rows_by_draw(
        rows, min_draw=min_draw, max_draw=max_draw
    )
    draw_note = ""
    if min_draw is not None or max_draw is not None:
        draw_note = (
            f", draws {min_draw if min_draw is not None else '…'}"
            f"–{max_draw if max_draw is not None else '…'}"
        )

    if all_rows:
        return draw_filtered, f"all rows{draw_note} (includes train; optimistic)"

    fraction = (
        DEFAULT_VALIDATION_FRACTION
        if validation_fraction is None
        else validation_fraction
    )
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {fraction}"
        )
    _, validation_rows = ResidualMLDatasetBuilder.time_split(
        draw_filtered,
        validation_fraction=fraction,
    )
    return (
        validation_rows,
        f"time-split validation fraction={fraction:g}{draw_note}",
    )


def _prob_vector(row: dict, keys: tuple[str, str, str]) -> list[float] | None:
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return [float(row[key]) for key in keys]


def _market_probs(row: dict) -> dict[str, float] | None:
    vector = _prob_vector(row, MARKET_KEYS)
    if vector is None:
        return None
    return {"1": vector[0], "X": vector[1], "2": vector[2]}


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
        "--market-only-baseline",
        action="store_true",
        help="Force blend := market_norm before ML predict",
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
    args = parser.parse_args()

    if args.all_rows and args.validation_fraction is not None:
        parser.error("Use either --all-rows or --validation-fraction, not both")

    dataset_path = resolve_path(args.dataset)
    model_path = resolve_path(args.model)
    all_rows = _load_rows(dataset_path)
    rows, filter_description = select_backtest_rows(
        all_rows,
        all_rows=args.all_rows,
        validation_fraction=args.validation_fraction,
        min_draw=args.min_draw,
        max_draw=args.max_draw,
    )
    print(
        f"Backtest filter: {filter_description} "
        f"({len(rows)}/{len(all_rows)} rows)",
        flush=True,
    )
    if not rows:
        print("No rows after filter; nothing to score")
        return

    label_map = {"1": 0, "X": 1, "2": 2}
    y_true = [label_map[row["label"]] for row in rows]

    baselines = {
        "market": MARKET_KEYS,
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

    config = DataSourceConfig(
        residual_ml_enabled=True,
        residual_ml_model_path=model_path,
    )
    model = ResidualMLModel.load(model_path, config=config)
    if model is None:
        print("ML model not found; skipping ML backtest")
        return

    use_market_only = args.market_only_baseline or is_market_only_weights(
        model.trainer.market_weight,
        model.trainer.dc_weight,
    )
    if use_market_only:
        apply_market_only_baseline(rows)
        print("Using market-only baseline (blend := market_norm)", flush=True)
        # Re-print blend after rewrite so it matches market.
        blend_keys = ("p_home_blend", "p_draw_blend", "p_away_blend")
        vectors = [_prob_vector(row, blend_keys) for row in rows]
        valid_indices = [index for index, vec in enumerate(vectors) if vec is not None]
        if valid_indices:
            loss = log_loss(
                [y_true[index] for index in valid_indices],
                [vectors[index] for index in valid_indices],
            )
            print(
                f"blend log loss (after market-only): {loss:.4f} "
                f"({len(valid_indices)} rows)"
            )

    raw_ml: list[dict[str, float]] = []
    markets: list[dict[str, float]] = []
    ml_labels: list[int] = []
    for row in rows:
        probs = model.trainer.predict_match_proba(row)
        market = _market_probs(row)
        if probs is None or market is None:
            continue
        raw_ml.append(probs)
        markets.append(market)
        ml_labels.append(label_map[row["label"]])
    if not raw_ml:
        print("ml: no rows")
        return

    market_on_ml_rows = log_loss(
        ml_labels,
        [[market["1"], market["X"], market["2"]] for market in markets],
    )
    print(
        f"market log loss (ML rows): {market_on_ml_rows:.4f} ({len(ml_labels)} rows)"
    )

    if args.final_shrink is None:
        alphas = list(SHRINK_ALPHAS)
    else:
        alphas = [0.0]
        if abs(args.final_shrink) > 1e-12:
            alphas.append(float(args.final_shrink))

    best_alpha = None
    best_loss = None
    for alpha in alphas:
        vectors = []
        for ml_probs, market in zip(raw_ml, markets):
            if alpha <= 0:
                shrunk = ml_probs
            else:
                shrunk = shrink_toward_market(ml_probs, market, alpha=alpha)
            vectors.append([shrunk["1"], shrunk["X"], shrunk["2"]])
        loss = log_loss(ml_labels, vectors)
        label = "ml" if alpha <= 0 else f"ml_shrink α={alpha:.1f}"
        print(f"{label} log loss: {loss:.4f} ({len(ml_labels)} rows)")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    beats_market = best_loss is not None and best_loss < market_on_ml_rows
    print(
        f"best shrink α={best_alpha:.1f} log loss: {best_loss:.4f} "
        f"({'beats' if beats_market else 'does not beat'} market "
        f"{market_on_ml_rows:.4f})"
    )
    if best_alpha is not None and best_alpha > 0:
        print(
            f"Set RESIDUAL_ML_FINAL_SHRINK_TO_MARKET={best_alpha} for live inference"
        )


if __name__ == "__main__":
    main()
