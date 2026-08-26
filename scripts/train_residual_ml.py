#!/usr/bin/env python3
"""Train the residual 1X2 ML model from historical DB matches."""

from __future__ import annotations

import argparse
from pathlib import Path

from calc.residual_ml_dataset import ResidualMLDatasetBuilder
from calc.residual_ml_trainer import ResidualMLTrainer
from database import SessionLocal, init_db
from objects.schema.data_classes.data_sources import DataSourceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/residual_ml/v1"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-draw", type=int, default=None)
    parser.add_argument("--max-draw", type=int, default=None)
    args = parser.parse_args()

    config = DataSourceConfig()
    init_db()
    session = SessionLocal()
    try:
        rows = ResidualMLDatasetBuilder(session, config=config).build(
            min_draw_number=args.min_draw or 4760,
            max_draw_number=args.max_draw or 4960,
        )
        train_rows, valid_rows = ResidualMLDatasetBuilder.time_split(
            rows,
            validation_fraction=args.validation_fraction,
        )
        print(
            f"Fitting model on {len(train_rows)} train / {len(valid_rows)} validation rows",
            flush=True,
        )
        trainer = ResidualMLTrainer(
            market_weight=config.residual_ml_market_weight,
            dc_weight=config.residual_ml_dc_weight,
        )
        result = trainer.fit(train_rows, valid_rows)
        saved = trainer.save(args.output_dir, version=args.version)
    finally:
        session.close()

    print(f"Train rows: {result.train_rows}, validation rows: {result.validation_rows}")
    print(f"Train log loss: {result.train_log_loss:.4f}")
    print(f"Validation log loss: {result.validation_log_loss:.4f}")
    if result.market_validation_log_loss is not None:
        print(f"Market validation log loss: {result.market_validation_log_loss:.4f}")
    if result.blend_validation_log_loss is not None:
        print(f"Blend validation log loss: {result.blend_validation_log_loss:.4f}")
    print(f"Model saved to {saved.model_path}")


if __name__ == "__main__":
    main()
