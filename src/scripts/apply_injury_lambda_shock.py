#!/usr/bin/env python3
"""T2: write a side CSV with DC λ shock + paired same-ρ recompute + 70/30 reblend."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml import blend_baselines, load_dataset_rows, select_backtest_rows
from src.calc.residual_ml.evaluation import LABEL_TO_INDEX
from src.calc.residual_ml.lambda_shock import (
    dixon_coles_1x2,
    rho_for_league,
    shock_expected_goals,
)
from src.data_sources.classic_dc_config import load_league_params
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path


def _market_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    keys = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return {
        "1": float(row["p_home_market_norm"]),
        "X": float(row["p_draw_market_norm"]),
        "2": float(row["p_away_market_norm"]),
    }


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def apply_t2_to_row(
    row: dict[str, Any],
    *,
    league_params: dict[int, Any],
    default_rho: float,
) -> tuple[dict[str, Any], dict[str, float] | None, dict[str, float] | None]:
    lambda_home = _optional_float(row.get("expected_home_goals"))
    lambda_away = _optional_float(row.get("expected_away_goals"))
    if lambda_home is None or lambda_away is None:
        return dict(row), None, None
    rho = rho_for_league(
        row.get("league_external_id"),
        default_rho=default_rho,
        league_params=league_params,
    )
    unadjusted = dixon_coles_1x2(lambda_home, lambda_away, rho)
    shocked_home, shocked_away = shock_expected_goals(
        lambda_home,
        lambda_away,
        home_unavailable_count=row.get("home_unavailable_count"),
        away_unavailable_count=row.get("away_unavailable_count"),
        has_availability=row.get("has_availability"),
    )
    adjusted = dixon_coles_1x2(shocked_home, shocked_away, rho)
    copy = dict(row)
    copy["expected_home_goals"] = shocked_home
    copy["expected_away_goals"] = shocked_away
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
    return copy, unadjusted, adjusted


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


def _engine_log_loss(
    rows: list[dict[str, Any]],
    engines: list[dict[str, float] | None],
) -> float | None:
    labels: list[int] = []
    vectors: list[list[float]] = []
    for row, engine in zip(rows, engines):
        if engine is None:
            continue
        labels.append(LABEL_TO_INDEX[row["label"]])
        vectors.append([engine["1"], engine["X"], engine["2"]])
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

    league_params = load_league_params()
    default_rho = DataSourceConfig().dixon_coles_rho
    source_rows = load_dataset_rows(source_path)
    applied: list[dict[str, Any]] = []
    unadjusted_engines: list[dict[str, float] | None] = []
    adjusted_engines: list[dict[str, float] | None] = []
    for row in source_rows:
        copy, unadjusted, adjusted = apply_t2_to_row(
            row,
            league_params=league_params,
            default_rho=default_rho,
        )
        applied.append(copy)
        unadjusted_engines.append(unadjusted)
        adjusted_engines.append(adjusted)

    with source_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in applied:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})

    official_rows, filter_description = select_backtest_rows(
        applied,
        validation_fraction=0.20,
        max_draw=None,
    )
    official_ids = {
        (row.get("match_date"), row.get("match_id")) for row in official_rows
    }
    official_pairs = [
        (source_row, unadjusted, adjusted)
        for source_row, unadjusted, adjusted in zip(
            source_rows, unadjusted_engines, adjusted_engines
        )
        if (source_row.get("match_date"), source_row.get("match_id")) in official_ids
    ]
    official_source = [item[0] for item in official_pairs]
    official_unadjusted = [item[1] for item in official_pairs]
    official_adjusted = [item[2] for item in official_pairs]
    blend_ll = _blend_log_loss(official_rows)
    payload = {
        "filter": filter_description,
        "source_dataset": str(source_path),
        "side_dataset": str(output_path),
        "official_518_blend_log_loss": blend_ll,
        "paired_unadjusted_dc_log_loss": _engine_log_loss(
            official_source, official_unadjusted
        ),
        "paired_adjusted_dc_log_loss": _engine_log_loss(
            official_source, official_adjusted
        ),
        "ship_blend": 1.0083,
        "blend_kill_line": 1.0103,
        "default_rho": default_rho,
        "row_count": len(applied),
        "official_518_n": len(official_rows),
    }
    json_path = resolve_repo_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {output_path} ({len(applied)} rows). "
        f"Official 518 blend LL={blend_ll:.4f}"
        if blend_ll is not None
        else f"Wrote {output_path}; blend LL unavailable",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
