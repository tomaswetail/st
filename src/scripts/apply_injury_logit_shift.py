#!/usr/bin/env python3
"""T3: write a side CSV with post-DC count-differential logit shift + 70/30 reblend."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml import (
    apply_count_differential_logit_shift,
    blend_baselines,
    load_dataset_rows,
    select_backtest_rows,
)
from src.calc.residual_ml.baseline import engine_baseline
from src.calc.residual_ml.evaluation import LABEL_TO_INDEX
from src.utils.repo_paths import resolve_repo_path


def _engine_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    norm_keys = ("p_home_dc_norm", "p_draw_dc_norm", "p_away_dc_norm")
    if all(row.get(key) not in (None, "") for key in norm_keys):
        return {
            "1": float(row["p_home_dc_norm"]),
            "X": float(row["p_draw_dc_norm"]),
            "2": float(row["p_away_dc_norm"]),
        }
    return engine_baseline(
        p_home_dc=row.get("p_home_dc"),
        p_draw_dc=row.get("p_draw_dc"),
        p_away_dc=row.get("p_away_dc"),
    )


def _market_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    keys = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return {
        "1": float(row["p_home_market_norm"]),
        "X": float(row["p_draw_market_norm"]),
        "2": float(row["p_away_market_norm"]),
    }


def apply_t3_to_row(row: dict[str, Any]) -> dict[str, Any]:
    engine = _engine_from_row(row)
    if engine is None:
        return dict(row)
    adjusted = apply_count_differential_logit_shift(
        engine,
        home_unavailable_count=row.get("home_unavailable_count"),
        away_unavailable_count=row.get("away_unavailable_count"),
        has_availability=row.get("has_availability"),
    )
    copy = dict(row)
    copy["p_home_dc"] = adjusted["1"]
    copy["p_draw_dc"] = adjusted["X"]
    copy["p_away_dc"] = adjusted["2"]
    copy["p_home_dc_norm"] = adjusted["1"]
    copy["p_draw_dc_norm"] = adjusted["X"]
    copy["p_away_dc_norm"] = adjusted["2"]
    market = _market_from_row(row)
    blend = blend_baselines(market, adjusted, market_weight=0.7, dc_weight=0.3)
    if blend is not None:
        copy["p_home_blend"] = blend["1"]
        copy["p_draw_blend"] = blend["X"]
        copy["p_away_blend"] = blend["2"]
    return copy


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _blend_log_loss(rows: list[dict[str, Any]]) -> float | None:
    labels: list[int] = []
    vectors: list[list[float]] = []
    for row in rows:
        keys = ("p_home_blend", "p_draw_blend", "p_away_blend")
        if any(row.get(key) in (None, "") for key in keys):
            continue
        labels.append(LABEL_TO_INDEX[row["label"]])
        vectors.append([float(row[key]) for key in keys])
    if not labels:
        return None
    return multiclass_log_loss(labels, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    source_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    if output_path.resolve() == source_path.resolve():
        raise SystemExit("Refusing to overwrite the source dataset")

    rows = [apply_t3_to_row(row) for row in load_dataset_rows(source_path)]
    with source_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})

    official_rows, filter_description = select_backtest_rows(
        rows,
        validation_fraction=0.20,
        max_draw=None,
    )
    blend_ll = _blend_log_loss(official_rows)
    payload = {
        "filter": filter_description,
        "source_dataset": str(source_path),
        "side_dataset": str(output_path),
        "official_518_blend_log_loss": blend_ll,
        "ship_blend": 1.0083,
        "blend_kill_line": 1.0103,
        "row_count": len(rows),
        "official_518_n": len(official_rows),
    }
    json_path = resolve_repo_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {output_path} ({len(rows)} rows). "
        f"Official 518 blend LL={blend_ll:.4f}"
        if blend_ll is not None
        else f"Wrote {output_path}; blend LL unavailable",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
