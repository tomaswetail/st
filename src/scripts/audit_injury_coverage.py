#!/usr/bin/env python3
"""Audit injury / availability coverage in a residual ML dataset CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from utils.repo_paths import resolve_repo_path


def _float_or_none(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int_or_none(raw: str | None) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def audit_injury_coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}

    has_col = "has_availability" in rows[0]
    availability_flags = [_int_or_none(row.get("has_availability")) for row in rows]
    with_availability = sum(1 for flag in availability_flags if flag == 1)
    without_availability = sum(1 for flag in availability_flags if flag == 0)

    missing_diffs = [
        _float_or_none(row.get("missing_value_difference"))
        for row in rows
        if _int_or_none(row.get("has_availability")) == 1
    ]
    non_null_diffs = [value for value in missing_diffs if value is not None]

    home_unavailable = [
        _int_or_none(row.get("home_unavailable_count"))
        for row in rows
        if _int_or_none(row.get("has_availability")) == 1
    ]
    away_unavailable = [
        _int_or_none(row.get("away_unavailable_count"))
        for row in rows
        if _int_or_none(row.get("has_availability")) == 1
    ]

    by_draw: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "with": 0})
    for row in rows:
        draw = _int_or_none(row.get("draw_number"))
        if draw is None:
            continue
        by_draw[draw]["total"] += 1
        if _int_or_none(row.get("has_availability")) == 1:
            by_draw[draw]["with"] += 1

    blend_cross: Counter[str] = Counter()
    for row in rows:
        flag = _int_or_none(row.get("has_availability"))
        market_weight = row.get("blend_market_weight", "")
        key = f"has_availability={flag}|blend_market_weight={market_weight}"
        blend_cross[key] += 1

    tiny_missing_value_rows = 0
    for row in rows:
        if _int_or_none(row.get("has_availability")) != 1:
            continue
        for key in ("home_missing_player_value", "away_missing_player_value"):
            value = _float_or_none(row.get(key))
            if value is not None and abs(value) < 1e-4:
                tiny_missing_value_rows += 1
                break

    draw_summary = {
        str(draw): {
            "row_count": stats["total"],
            "has_availability_count": stats["with"],
            "has_availability_pct": round(
                100.0 * stats["with"] / stats["total"], 2
            )
            if stats["total"]
            else 0.0,
        }
        for draw, stats in sorted(by_draw.items())
    }

    return {
        "row_count": len(rows),
        "has_availability_pct": round(100.0 * with_availability / len(rows), 2),
        "has_availability_counts": {
            "1": with_availability,
            "0": without_availability,
        },
        "missing_value_difference": {
            "non_null_count": len(non_null_diffs),
            "mean": round(mean(non_null_diffs), 6) if non_null_diffs else None,
            "median": round(median(non_null_diffs), 6) if non_null_diffs else None,
            "min": round(min(non_null_diffs), 6) if non_null_diffs else None,
            "max": round(max(non_null_diffs), 6) if non_null_diffs else None,
        },
        "unavailable_counts_when_available": {
            "home_mean": round(mean(v for v in home_unavailable if v is not None), 3)
            if any(v is not None for v in home_unavailable)
            else None,
            "away_mean": round(mean(v for v in away_unavailable if v is not None), 3)
            if any(v is not None for v in away_unavailable)
            else None,
        },
        "tiny_scaled_missing_value_rows": tiny_missing_value_rows,
        "has_availability_x_blend_market_weight": dict(
            sorted(blend_cross.items(), key=lambda item: (-item[1], item[0]))
        ),
        "by_draw": draw_summary,
        "columns_present": {
            "has_availability": has_col,
            "blend_market_weight": "blend_market_weight" in rows[0],
        },
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("artifacts/injury_coverage_audit.json"),
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    rows = load_csv_rows(dataset_path)
    report = audit_injury_coverage(rows)
    report["dataset_path"] = str(dataset_path)

    json_path = resolve_repo_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {report['row_count']}", flush=True)
    print(f"has_availability=1: {report['has_availability_pct']}%", flush=True)
    print(
        f"tiny scaled missing_value rows: {report['tiny_scaled_missing_value_rows']}",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
