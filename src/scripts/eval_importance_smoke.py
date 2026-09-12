#!/usr/bin/env python3
"""Print native + permutation importances on expanding-2023 (CatBoost direct)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.calc.residual_ml.folds import expanding_year_folds
from src.calc.residual_ml.io import load_dataset_rows
from src.scripts.compare_probability_models import _print_importance_smoke
from src.utils.repo_paths import resolve_repo_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/fixtures_dataset.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_repo_path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    rows = load_dataset_rows(dataset_path)
    expanding_year_folds(rows)  # validate the CSV has folds
    _print_importance_smoke(rows, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
