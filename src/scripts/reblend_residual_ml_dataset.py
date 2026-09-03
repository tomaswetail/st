#!/usr/bin/env python3
"""Recompute blend columns in a residual ML dataset CSV (no feature rebuild).

Useful for injury ablations B0/B2/B3 where only conditional blend weights differ.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from calc.draw_adjustment import load_draw_adjustment_config
from calc.residual_ml.baseline import apply_draw_adjustment, blend_baselines
from calc.residual_ml.blend_weights import (
    load_blend_weights_config,
    load_dc_league_quality,
    select_blend_weights,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.repo_paths import resolve_repo_path


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    return float(raw)


_COUNT_PROXY_MAX = 1e-4


def _patch_injury_value_columns(row: dict[str, str]) -> dict[str, str]:
    patched = dict(row)
    for key in ("home_missing_player_value", "away_missing_player_value"):
        value = _float(patched, key)
        if value is not None and abs(value) <= _COUNT_PROXY_MAX:
            patched[key] = ""
    diff = _float(patched, "missing_value_difference")
    if diff is not None and abs(diff) <= _COUNT_PROXY_MAX:
        patched["missing_value_difference"] = ""
    return patched


def reblend_rows(
    rows: list[dict[str, str]],
    *,
    blend_config_path: Path,
    fallback_market_weight: float,
    fallback_dc_weight: float,
    patch_injury_values: bool = True,
) -> list[dict[str, str]]:
    blend_config = load_blend_weights_config(blend_config_path)
    draw_config = load_draw_adjustment_config()
    dc_quality = load_dc_league_quality()
    updated: list[dict[str, str]] = []

    for row in rows:
        row = _patch_injury_value_columns(row) if patch_injury_values else row
        market = {
            "1": _float(row, "p_home_market_norm"),
            "X": _float(row, "p_draw_market_norm"),
            "2": _float(row, "p_away_market_norm"),
        }
        engine = {
            "1": _float(row, "p_home_dc_norm"),
            "X": _float(row, "p_draw_dc_norm"),
            "2": _float(row, "p_away_dc_norm"),
        }
        if any(market[outcome] is None for outcome in market):
            updated.append(row)
            continue

        engine_missing = any(engine[outcome] is None for outcome in engine)
        if engine_missing and not blend_config.enabled:
            new_row = dict(row)
            new_row["blend_market_weight"] = str(fallback_market_weight)
            new_row["blend_dc_weight"] = str(fallback_dc_weight)
            new_row["p_home_blend_pre_draw"] = str(market["1"])
            new_row["p_draw_blend_pre_draw"] = str(market["X"])
            new_row["p_away_blend_pre_draw"] = str(market["2"])
            new_row["p_home_blend"] = str(market["1"])
            new_row["p_draw_blend"] = str(market["X"])
            new_row["p_away_blend"] = str(market["2"])
            updated.append(new_row)
            continue
        if engine_missing:
            updated.append(row)
            continue

        market_weight, dc_weight = select_blend_weights(
            row,
            blend_config,
            league_external_id=row.get("league_external_id"),
            dc_league_log_loss=dc_quality,
            fallback_market_weight=fallback_market_weight,
            fallback_dc_weight=fallback_dc_weight,
        )
        blend = blend_baselines(
            market,
            engine,
            market_weight=market_weight,
            dc_weight=dc_weight,
        )
        if blend is None:
            updated.append(row)
            continue

        new_row = dict(row)
        new_row["blend_market_weight"] = str(market_weight)
        new_row["blend_dc_weight"] = str(dc_weight)
        new_row["p_home_blend_pre_draw"] = str(blend["1"])
        new_row["p_draw_blend_pre_draw"] = str(blend["X"])
        new_row["p_away_blend_pre_draw"] = str(blend["2"])
        adjusted = apply_draw_adjustment(blend, row, draw_config)
        assert adjusted is not None
        new_row["p_home_blend"] = str(adjusted["1"])
        new_row["p_draw_blend"] = str(adjusted["X"])
        new_row["p_away_blend"] = str(adjusted["2"])
        updated.append(new_row)

    return updated


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--blend-config",
        type=Path,
        default=Path("config/blend_weights.json"),
    )
    parser.add_argument(
        "--no-patch-injury-values",
        action="store_true",
        help="Keep tiny count-proxy missing_value columns unchanged",
    )
    args = parser.parse_args()

    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    blend_config_path = resolve_repo_path(args.blend_config)
    config = DataSourceConfig()

    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    updated = reblend_rows(
        rows,
        blend_config_path=blend_config_path,
        fallback_market_weight=config.residual_ml_market_weight,
        fallback_dc_weight=config.residual_ml_dc_weight,
        patch_injury_values=not args.no_patch_injury_values,
    )
    write_csv(output_path, updated)
    print(f"Reblended {len(updated)} rows → {output_path}", flush=True)


if __name__ == "__main__":
    main()
