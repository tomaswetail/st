#!/usr/bin/env python3
"""Snapshot post-DC-tune baseline artifacts to a dated artifacts folder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from src.utils.repo_paths import repo_root, resolve_repo_path


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
    git_commit: str,
    dataset_rows: int,
) -> str:
    if metrics:
        metrics_block = f"""\
## Copied sweep metrics (`sweep_best/sweep_results.json` best trial)

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
# Post-DC-tune baseline snapshot ({snapshot_date})

Git commit at snapshot: `{git_commit}`

Dataset rows: {dataset_rows} (expect ~2589)

Full ablation and production recommendation: [`docs/reports/ml_residual/baseline_after_dc_tune.md`](../../docs/reports/ml_residual/baseline_after_dc_tune.md)

{metrics_block}
## Files in this snapshot

| File | Source | Notes |
|------|--------|-------|
| `dataset.csv` | `data/residual_ml/dataset.csv` | Post-tune DC features |
| `classic_dc_league_params.json` | `config/classic_dc_league_params.json` | Per-league tuned params |
| `sweep_best/` | `models/residual_ml/sweep_best/` | Best ML model + sweep results |
| `backtest.txt` | generated | 518-row validation (`--include-holdout`) |
| `backtest_holdout_excluded.txt` | generated | 492-row tuning slice (default) |
| `verify_dc_coverage.txt` | copied if present | League coverage report |
| `eval_post_dc.json` | copied if present | Multi-slice eval JSON |

Reproduce `backtest.txt`:

```bash
python -m src.scripts.backtest_residual_ml \\
  --dataset artifacts/baseline_post_dc_tune_{snapshot_date}/dataset.csv \\
  --model artifacts/baseline_post_dc_tune_{snapshot_date}/sweep_best/model.pkl \\
  --validation-fraction 0.20 \\
  --include-holdout
```
"""


def _run_backtest(
    snapshot_dir: Path,
    root: Path,
    *,
    output_name: str,
    include_holdout: bool,
) -> bool:
    dataset_path = snapshot_dir / "dataset.csv"
    model_path = snapshot_dir / "sweep_best" / "model.pkl"
    if not dataset_path.exists() or not model_path.exists():
        return False

    command = [
        sys.executable,
        "-m",
        "src.scripts.backtest_residual_ml",
        "--dataset",
        str(dataset_path),
        "--model",
        str(model_path),
        "--validation-fraction",
        "0.20",
    ]
    if include_holdout:
        command.append("--include-holdout")

    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}".strip()
    backtest_path = snapshot_dir / output_name
    backtest_path.write_text(output + "\n", encoding="utf-8")
    return result.returncode == 0


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _dataset_row_count(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return 0
    with dataset_path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


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
        help="Skip generating backtest outputs",
    )
    args = parser.parse_args()

    root = repo_root()
    snapshot_dir = resolve_repo_path(args.output_root) / f"baseline_post_dc_tune_{args.date}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    dataset_source = root / "data" / "residual_ml" / "dataset.csv"
    if _copy_if_exists(dataset_source, snapshot_dir / "dataset.csv"):
        copied.append("dataset.csv")
    else:
        skipped.append("dataset.csv (missing)")

    league_params_source = root / "config" / "classic_dc_league_params.json"
    if _copy_if_exists(league_params_source, snapshot_dir / "classic_dc_league_params.json"):
        copied.append("classic_dc_league_params.json")
    else:
        skipped.append("classic_dc_league_params.json (missing)")

    sweep_source = root / "models" / "residual_ml" / "sweep_best"
    if _copy_if_exists(sweep_source, snapshot_dir / "sweep_best"):
        copied.append("sweep_best/")
    else:
        skipped.append("sweep_best/ (missing)")

    for artifact_name in (
        f"verify_dc_coverage_{args.date}.txt",
        f"eval_post_dc_{args.date}.json",
    ):
        source = root / "artifacts" / artifact_name
        if _copy_if_exists(source, snapshot_dir / artifact_name.replace(f"_{args.date}", "")):
            copied.append(artifact_name)
        else:
            skipped.append(f"{artifact_name} (missing — run verify/eval first)")

    git_commit = _git_commit(root)
    dataset_rows = _dataset_row_count(snapshot_dir / "dataset.csv")
    metrics = _load_sweep_metrics(snapshot_dir / "sweep_best" / "sweep_results.json")
    readme_path = snapshot_dir / "README.md"
    readme_path.write_text(
        _build_readme(
            metrics=metrics,
            snapshot_date=args.date,
            git_commit=git_commit,
            dataset_rows=dataset_rows,
        ),
        encoding="utf-8",
    )
    copied.append("README.md")

    if args.skip_backtest:
        skipped.append("backtest outputs (--skip-backtest)")
    else:
        if _run_backtest(
            snapshot_dir,
            root,
            output_name="backtest.txt",
            include_holdout=True,
        ):
            copied.append("backtest.txt")
        else:
            skipped.append("backtest.txt (failed)")

        if _run_backtest(
            snapshot_dir,
            root,
            output_name="backtest_holdout_excluded.txt",
            include_holdout=False,
        ):
            copied.append("backtest_holdout_excluded.txt")
        else:
            skipped.append("backtest_holdout_excluded.txt (failed)")

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
