"""Tests for residual ML dataset I/O."""

from __future__ import annotations

import csv
from pathlib import Path

from calc.residual_ml.io import load_dataset_rows


def test_load_dataset_rows_csv_coerces_int_fields(tmp_path: Path):
    path = tmp_path / "dataset.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "draw_number",
                "label",
                "match_date",
                "home_short_rest",
                "feature_value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_id": "42",
                "draw_number": "4960",
                "label": "1",
                "match_date": "2024-01-01",
                "home_short_rest": "1",
                "feature_value": "0.5",
            }
        )

    rows = load_dataset_rows(path)
    assert len(rows) == 1
    assert rows[0]["match_id"] == 42
    assert rows[0]["draw_number"] == 4960
    assert rows[0]["label"] == "1"
    assert rows[0]["home_short_rest"] == 1
    assert rows[0]["feature_value"] == 0.5


def test_load_dataset_rows_empty_csv(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("match_id,label\n", encoding="utf-8")
    assert load_dataset_rows(path) == []
