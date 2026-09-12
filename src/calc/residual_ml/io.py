"""Load residual ML dataset rows from CSV or JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_STRING_FIELDS = frozenset(
    {
        "label",
        "match_date",
        "feature_cutoff_date",
        "market_price_type",
    }
)

_INT_FIELDS = frozenset(
    {
        "match_id",
        "draw_number",
        "league_external_id",
        "home_matches_last_14_days",
        "away_matches_last_14_days",
        "home_short_rest",
        "away_short_rest",
        "home_extra_time_in_previous_match",
        "away_extra_time_in_previous_match",
        "extra_time_x_short_rest",
    }
)


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    """Load dataset rows from CSV or JSON."""
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return payload

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                if value is None or value == "":
                    row[key] = None
                elif key in _STRING_FIELDS:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                        if row[key].is_integer() and key in _INT_FIELDS:
                            row[key] = int(row[key])
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows
