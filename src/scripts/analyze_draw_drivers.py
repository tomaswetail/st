#!/usr/bin/env python3
"""Run Phase 3.1 draw-driver discovery on residual ML dataset.csv.

```bash
export PYTHONPATH=src
python src/scripts/analyze_draw_drivers.py \\
  --dataset data/residual_ml/dataset.csv \\
  --output-dir artifacts/draw_analysis \\
  --write-doc docs/reports/ml_draw/draw_driver_analysis.md
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calc.draw_driver_analysis import (
    load_dataset_rows,
    render_draw_driver_markdown,
    run_draw_discovery,
    write_discovery_artifacts,
)
from config.eval_protocol import VALIDATION_FRACTION
from utils.repo_paths import repo_root, resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/draw_analysis"),
    )
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=Path("docs/reports/ml_draw/draw_driver_analysis.md"),
        help="Markdown report path (set empty string to skip)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Include draws ≥ HOLDOUT_DRAW_MIN (default: exclude)",
    )
    parser.add_argument("--l1-C", type=float, default=0.5)
    parser.add_argument("--elastic-C", type=float, default=0.5)
    args = parser.parse_args()

    root = repo_root()
    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir)

    rows = load_dataset_rows(dataset_path)
    result = run_draw_discovery(
        rows,
        validation_fraction=args.validation_fraction,
        exclude_holdout=not args.include_holdout,
        l1_C=args.l1_C,
        elastic_C=args.elastic_C,
    )
    paths = write_discovery_artifacts(result, output_dir)

    doc_arg = str(args.write_doc).strip() if args.write_doc is not None else ""
    if doc_arg:
        doc_path = resolve_repo_path(Path(doc_arg))
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(render_draw_driver_markdown(result), encoding="utf-8")
        paths["doc"] = doc_path

    print(
        f"Draw discovery: train={result.n_train} val={result.n_validation} "
        f"stable_drivers={len(result.stable_drivers)}",
        flush=True,
    )
    print(
        f"Val draw Brier blend={result.validation_draw_brier_blend:.4f} "
        f"L1={result.validation_draw_brier_model:.4f} "
        f"Δ={result.validation_draw_brier_model - result.validation_draw_brier_blend:+.4f}",
        flush=True,
    )
    print(
        f"Val draw LL blend={result.validation_draw_log_loss_blend:.4f} "
        f"L1={result.validation_draw_log_loss_model:.4f} "
        f"Δ={result.validation_draw_log_loss_model - result.validation_draw_log_loss_blend:+.4f}",
        flush=True,
    )
    if result.stable_drivers:
        print("Stable drivers:", ", ".join(result.stable_drivers), flush=True)
    else:
        print("Stable drivers: (none)", flush=True)
    for key, path in paths.items():
        print(f"  {key}: {path}", flush=True)


if __name__ == "__main__":
    main()
