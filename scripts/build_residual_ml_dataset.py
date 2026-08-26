#!/usr/bin/env python3
"""Build residual ML training dataset from historical Stryktipset matches."""

from __future__ import annotations

from pathlib import Path

from calc.residual_ml_dataset import ResidualMLDatasetBuilder
from database import SessionLocal, init_db

MIN_DRAW = 4760
MAX_DRAW = 4960
CSV_OUTPUT = Path("data/residual_ml/dataset.csv")
JSON_OUTPUT = Path("data/residual_ml/dataset.json")


def build_residual_ml_dataset() -> list[dict]:
    """Build and export the residual ML dataset for the default draw window."""
    init_db()
    session = SessionLocal()
    try:
        builder = ResidualMLDatasetBuilder(session)
        rows = builder.build(min_draw_number=MIN_DRAW, max_draw_number=MAX_DRAW)
        builder.export_csv(CSV_OUTPUT, rows)
        builder.export_json(JSON_OUTPUT, rows)
        print(f"Wrote {len(rows)} rows to {CSV_OUTPUT} and {JSON_OUTPUT}", flush=True)
        return rows
    finally:
        session.close()


if __name__ == "__main__":
    build_residual_ml_dataset()
