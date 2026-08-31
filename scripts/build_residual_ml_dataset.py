#!/usr/bin/env python3
"""Build residual ML training dataset from historical Stryktipset matches."""

from __future__ import annotations

from pathlib import Path

from calc.residual_ml_dataset import ResidualMLDatasetBuilder
from database import SessionLocal, init_db

MIN_DRAW = 4760
MAX_DRAW = 4960
CSV_OUTPUT = Path("data/residual_ml/dataset.csv")


def build_residual_ml_dataset() -> int:
    """Stream the residual ML dataset CSV for the default draw window."""
    init_db()
    session = SessionLocal()
    try:
        builder = ResidualMLDatasetBuilder(session)
        row_count = builder.export_csv_from_iter(
            CSV_OUTPUT,
            builder.iter_rows(min_draw_number=MIN_DRAW, max_draw_number=MAX_DRAW),
        )
        print(f"Wrote {row_count} rows to {CSV_OUTPUT}", flush=True)
        return row_count
    finally:
        session.close()


if __name__ == "__main__":
    build_residual_ml_dataset()
