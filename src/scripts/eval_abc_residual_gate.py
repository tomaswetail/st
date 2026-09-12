#!/usr/bin/env python3
"""Train/gate family B residual models on opening (research) and ST universes.

Does not enable live ML. Does not write models/residual_ml/sweep_best/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.calc.residual_ml.abc_gate import run_universe_gate
from src.calc.residual_ml.io import load_dataset_rows
from src.utils.repo_paths import repo_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-csv",
        type=Path,
        default=Path("data/residual_ml/fixtures_dataset.csv"),
    )
    parser.add_argument(
        "--st-csv",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--opening-artifact",
        type=Path,
        default=Path("artifacts/abc_opening_gate.json"),
    )
    parser.add_argument(
        "--st-artifact",
        type=Path,
        default=Path("artifacts/abc_st_gate.json"),
    )
    parser.add_argument(
        "--skip-opening",
        action="store_true",
        help="Skip the opening universe (ST still runs).",
    )
    parser.add_argument(
        "--skip-st",
        action="store_true",
        help="Skip the ST universe.",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_absolute():
        target = path
    else:
        target = repo_root() / path
    if target.resolve().as_posix().endswith("models/residual_ml/sweep_best") or (
        "models/residual_ml/sweep_best" in target.resolve().as_posix()
    ):
        raise SystemExit("Refusing to write models/residual_ml/sweep_best/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {target}", flush=True)


def _print_summary(payload: dict[str, Any]) -> None:
    chosen = payload.get("chosen") or {}
    print(
        f"\n=== {payload['universe']} {payload['decision']} ({payload['reason']}) ===",
        flush=True,
    )
    print(
        f"n_rows={payload['n_rows']} backend={chosen.get('backend')} "
        f"hyper={chosen.get('hyperparams')} "
        f"chosen α={chosen.get('alpha')} threshold={chosen.get('threshold')}",
        flush=True,
    )
    print(
        f"mean model LL={chosen.get('mean_model_ll')} "
        f"mean baseline LL={chosen.get('mean_baseline_ll')}",
        flush=True,
    )
    years = chosen.get("years") or []
    for year in years:
        print(
            f"  year={year['year']} eligible={year['eligible']} "
            f"n_score={year['n_score']} model={year['model_ll']} "
            f"baseline={year['baseline_ll']} α={year['alpha']} "
            f"thr={year['threshold']} closing={year.get('closing_ll')} "
            f"{year.get('note', '')}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_opening:
        fixture_rows = load_dataset_rows(args.fixtures_csv)
        opening = run_universe_gate(fixture_rows, universe="opening")
        _print_summary(opening)
        _write_json(args.opening_artifact, opening)
    if not args.skip_st:
        st_rows = load_dataset_rows(args.st_csv)
        st_gate = run_universe_gate(st_rows, universe="st")
        _print_summary(st_gate)
        _write_json(args.st_artifact, st_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
