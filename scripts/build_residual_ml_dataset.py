#!/usr/bin/env python3
"""Build residual ML training dataset from historical Stryktipset matches.

Classic DC engine (default) + fast HA rebuild example:

    RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast \\
    RESIDUAL_ML_DC_ENGINE=classic \\
    PYTHONPATH=. python scripts/build_residual_ml_dataset.py

Per-league DC hyperparameters (run before rebuild when tuning):

    PYTHONPATH=. python scripts/optimize_classic_dixon_coles.py \\
      --validation-fraction 0.15 \\
      --output config/classic_dc_league_params.json

    PYTHONPATH=. python scripts/train_residual_ml.py --sweep
    PYTHONPATH=. python scripts/backtest_residual_ml.py \\
      --model models/residual_ml/sweep_best/model.pkl
"""

from __future__ import annotations

from pathlib import Path

from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from calc.residual_ml import ResidualMLDatasetBuilder
from database import SessionLocal, init_db

CSV_OUTPUT = Path("data/residual_ml/dataset.csv")


def build_residual_ml_dataset() -> int:
    """Stream the residual ML dataset CSV for the default draw window."""
    init_db()
    session = SessionLocal()
    try:
        builder = ResidualMLDatasetBuilder(session)
        row_count = builder.export_csv_from_iter(
            CSV_OUTPUT,
            builder.iter_rows(
                min_draw_number=STRYKETIPSET_DRAW_MIN,
                max_draw_number=STRYKETIPSET_DRAW_MAX,
            ),
        )
        print(f"Wrote {row_count} rows to {CSV_OUTPUT}", flush=True)
        return row_count
    finally:
        session.close()


if __name__ == "__main__":
    build_residual_ml_dataset()
