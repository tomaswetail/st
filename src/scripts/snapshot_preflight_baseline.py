#!/usr/bin/env python3
"""Snapshot pre-DC-tune baseline artifacts to a dated artifacts folder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from utils.repo_paths import repo_root, resolve_repo_path

LEAGUE_PARAMS_NOTE = """\
classic_dc_league_params.json is NOT included in this snapshot.

Reason: if the snapshot script runs after the DC optimization pipeline, copying
config/classic_dc_league_params.json would store post-tune params while labeling
the folder as "pre-DC-tune".

Reliable pre-tune artifacts in this snapshot:
- dataset.csv (copied from data/residual_ml/dataset_classic.csv — classic DC features
  before the full per-league grid tune)
- sweep_best/ (model + sweep_results.json copied at snapshot time)
- backtest.txt (re-run on dataset_classic.csv at 20% validation)

For league params history, rely on git history of config/classic_dc_league_params.json
or take a snapshot before running src/scripts/run_classic_dc_ml_pipeline.sh.
"""


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def _load_sweep_metrics(sweep_results_path: Path) -> dict[str, float | int] | None:
    if not sweep_results_path.exists():
        return None
    payload = json.loads(sweep_results_path.read_text(encoding="utf-8"))
    best = payload.get("best")
    if not isinstance(best, dict):
        return None
    return {
        "validation_rows": int(best.get("validation_rows", 0)),
        "market": float(best["market_validation_log_loss"]),
        "blend": float(best["blend_validation_log_loss"]),
        "ml": float(best["validation_log_loss"]),
    }


def _build_readme(
    *,
    metrics: dict[str, float | int] | None,
    snapshot_date: str,
) -> str:
    if metrics:
        metrics_block = f"""\
## Copied sweep metrics (`sweep_best/sweep_results.json` best trial)

These numbers reflect the model copied at snapshot time ({snapshot_date}), not
recomputed on `dataset.csv` in this folder.

| System | Log loss | Rows |
|--------|----------|------|
| Market | {metrics['market']:.4f} | {metrics['validation_rows']} |
| Blend 70/30 | {metrics['blend']:.4f} | {metrics['validation_rows']} |
| ML (no shrink) | {metrics['ml']:.4f} | {metrics['validation_rows']} |
"""
    else:
        metrics_block = """\
## Copied sweep metrics

`sweep_best/sweep_results.json` was not available at snapshot time.
See `backtest.txt` for a re-run on `dataset.csv`.
"""

    return f"""\
# Pre-DC-tune baseline snapshot ({snapshot_date})

**Post-hoc proxy snapshot:** the full DC pipeline had already run in this repo when
this folder was created. Only `dataset.csv` (from `dataset_classic.csv`) preserves
pre-tune DC feature columns reliably.

{metrics_block}
## Files in this snapshot

| File | Source | Notes |
|------|--------|-------|
| `dataset.csv` | `data/residual_ml/dataset_classic.csv` | **Pre-tune proxy** — classic DC before full grid tune |
| `sweep_best/` | `models/residual_ml/sweep_best/` | Copied at snapshot time (may be post-pipeline model) |
| `backtest.txt` | generated | Backtest on `dataset.csv` @ 20% validation (`--include-holdout`) |
| `classic_dc_league_params.pre_tune_note.txt` | generated | Why league params are omitted |

`classic_dc_league_params.json` is **not** copied — see the note file above.

Reproduce `backtest.txt`:

```bash
export PYTHONPATH=src
python src/scripts/backtest_residual_ml.py \\
  --dataset artifacts/baseline_pre_dc_tune_{snapshot_date}/dataset.csv \\
  --model artifacts/baseline_pre_dc_tune_{snapshot_date}/sweep_best/model.pkl \\
  --validation-fraction 0.20 \\
  --include-holdout
```
"""


def _run_backtest(snapshot_dir: Path, root: Path) -> bool:
    dataset_path = snapshot_dir / "dataset.csv"
    model_path = snapshot_dir / "sweep_best" / "model.pkl"
    if not dataset_path.exists() or not model_path.exists():
        return False

    backtest_script = root / "src" / "scripts" / "backtest_residual_ml.py"
    command = [
        sys.executable,
        str(backtest_script),
        "--dataset",
        str(dataset_path),
        "--model",
        str(model_path),
        "--validation-fraction",
        "0.20",
        "--include-holdout",
    ]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}".strip()
    backtest_path = snapshot_dir / "backtest.txt"
    backtest_path.write_text(output + "\n", encoding="utf-8")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=str,
        default=date.today().strftime("%Y%m%d"),
        help="Snapshot date suffix YYYYMMDD (default: today)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts"),
        help="Parent directory for snapshot folders",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip generating backtest.txt",
    )
    args = parser.parse_args()

    root = repo_root()
    snapshot_dir = resolve_repo_path(args.output_root) / f"baseline_pre_dc_tune_{args.date}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    league_params_dest = snapshot_dir / "classic_dc_league_params.json"
    if league_params_dest.exists():
        league_params_dest.unlink()
        skipped.append("classic_dc_league_params.json (removed — not a reliable pre-tune artifact)")

    note_path = snapshot_dir / "classic_dc_league_params.pre_tune_note.txt"
    note_path.write_text(LEAGUE_PARAMS_NOTE, encoding="utf-8")
    copied.append("classic_dc_league_params.pre_tune_note.txt")

    dataset_source = root / "data" / "residual_ml" / "dataset_classic.csv"
    if _copy_if_exists(dataset_source, snapshot_dir / "dataset.csv"):
        copied.append("dataset.csv (from dataset_classic.csv)")
    else:
        skipped.append("dataset.csv (dataset_classic.csv missing)")

    sweep_source = root / "models" / "residual_ml" / "sweep_best"
    if _copy_if_exists(sweep_source, snapshot_dir / "sweep_best"):
        copied.append("sweep_best/")
    else:
        skipped.append("sweep_best/ (missing)")

    metrics = _load_sweep_metrics(snapshot_dir / "sweep_best" / "sweep_results.json")
    readme_path = snapshot_dir / "README.md"
    readme_path.write_text(
        _build_readme(metrics=metrics, snapshot_date=args.date),
        encoding="utf-8",
    )
    copied.append("README.md")

    if args.skip_backtest:
        skipped.append("backtest.txt (--skip-backtest)")
    elif _run_backtest(snapshot_dir, root):
        copied.append("backtest.txt")
    else:
        skipped.append("backtest.txt (dataset.csv or model missing, or backtest failed)")

    print(f"Snapshot directory: {snapshot_dir}", flush=True)
    print("Copied:", flush=True)
    for item in copied:
        print(f"  - {item}", flush=True)
    if skipped:
        print("Skipped:", flush=True)
        for item in skipped:
            print(f"  - {item}", flush=True)


if __name__ == "__main__":
    main()
