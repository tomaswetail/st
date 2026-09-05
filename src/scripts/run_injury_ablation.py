#!/usr/bin/env python3
"""Run injury regression ablations B0–B3 (rebuild → sweep → eval).

Uses RESIDUAL_ML_BLEND_WEIGHTS_PATH so production config is untouched.
Each ablation writes dataset/model/eval artifacts under data/ and artifacts/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.repo_paths import repo_root, resolve_repo_path

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "B0": {
        "blend_config": "config/injury_ablation/B0_blend_weights.json",
        "exclude_injury_features": False,
        "reuse_dataset_from": None,
    },
    "B1": {
        "blend_config": "config/injury_ablation/B0_blend_weights.json",
        "exclude_injury_features": True,
        "reuse_dataset_from": "B0",
    },
    "B2": {
        "blend_config": "config/injury_ablation/B2_blend_weights.json",
        "exclude_injury_features": False,
        "reuse_dataset_from": None,
    },
    "B3": {
        "blend_config": "config/injury_ablation/B3_blend_weights.json",
        "exclude_injury_features": False,
        "reuse_dataset_from": None,
    },
}


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=repo_root())


def _paths_for(ablation_id: str) -> dict[str, Path]:
    root = repo_root()
    return {
        "dataset": root / "data" / "residual_ml" / "ablation" / ablation_id / "dataset.csv",
        "model_dir": root / "models" / "residual_ml" / "ablation" / ablation_id,
        "eval_json": root / "artifacts" / f"injury_ablation_{ablation_id}.json",
        "blend_config": root / ABLATION_CONFIGS[ablation_id]["blend_config"],
    }


def run_ablation(
    ablation_id: str,
    *,
    skip_rebuild: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    use_reblend: bool = True,
    source_dataset: Path | None = None,
) -> None:
    if ablation_id not in ABLATION_CONFIGS:
        raise SystemExit(f"Unknown ablation {ablation_id!r}; choose from {list(ABLATION_CONFIGS)}")

    spec = ABLATION_CONFIGS[ablation_id]
    paths = _paths_for(ablation_id)
    paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RESIDUAL_ML_BLEND_WEIGHTS_PATH"] = str(paths["blend_config"])

    reuse_from = spec["reuse_dataset_from"]
    if reuse_from and not skip_rebuild:
        source = _paths_for(reuse_from)["dataset"]
        if not source.is_file():
            raise SystemExit(
                f"{ablation_id} reuses dataset from {reuse_from}; run {reuse_from} first "
                f"(missing {source})"
            )
        paths["dataset"].write_bytes(source.read_bytes())
        print(f"Reused dataset from {reuse_from}: {paths['dataset']}", flush=True)
    elif not skip_rebuild:
        base_dataset = source_dataset or resolve_repo_path(
            Path("data/residual_ml/dataset.csv")
        )
        if use_reblend:
            _run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.reblend_residual_ml_dataset",
                    "--input",
                    str(base_dataset),
                    "--output",
                    str(paths["dataset"]),
                    "--blend-config",
                    str(paths["blend_config"]),
                ],
                env=env,
            )
        else:
            _run(
                [sys.executable, "-u", "-m", "src.scripts.build_residual_ml_dataset"],
                env=env,
            )
            built = resolve_repo_path(Path("data/residual_ml/dataset.csv"))
            paths["dataset"].write_bytes(built.read_bytes())
            print(f"Copied rebuilt dataset to {paths['dataset']}", flush=True)

    if not skip_train:
        train_cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.scripts.train_residual_ml",
            "--dataset",
            str(paths["dataset"]),
            "--sweep",
            "--output-dir",
            str(paths["model_dir"]),
        ]
        if spec["exclude_injury_features"]:
            train_cmd.append("--exclude-injury-features")
        _run(train_cmd, env=env)

    if not skip_eval:
        _run(
            [
                sys.executable,
                "-m",
                "src.scripts.eval_residual_ml",
                "--dataset",
                str(paths["dataset"]),
                "--model",
                str(paths["model_dir"] / "model.pkl"),
                "--include-holdout",
                "--ablation",
                "--json",
                str(paths["eval_json"]),
            ],
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["B0", "B1", "B2", "B3", "all"],
        default="all",
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Run full feature rebuild instead of fast CSV reblend",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    steps = ["B0", "B1", "B2", "B3"] if args.step == "all" else [args.step]
    summary: dict[str, Any] = {}

    for ablation_id in steps:
        print(f"\n=== Ablation {ablation_id} ===", flush=True)
        run_ablation(
            ablation_id,
            skip_rebuild=args.skip_rebuild,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            use_reblend=not args.full_rebuild,
        )
        eval_path = _paths_for(ablation_id)["eval_json"]
        if eval_path.is_file():
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            pooled = payload.get("slices", {}).get("pooled", {})
            summary[ablation_id] = {
                "best_shrink_log_loss": pooled.get("best_shrink_log_loss"),
                "best_shrink_alpha": pooled.get("best_shrink_alpha"),
                "ml_log_loss": pooled.get("ml_log_loss"),
                "blend_log_loss": pooled.get("baselines", {}).get("blend", {}).get(
                    "log_loss"
                ),
                "market_log_loss": pooled.get("baselines", {}).get("market", {}).get(
                    "log_loss"
                ),
            }

    if summary:
        summary_path = repo_root() / "artifacts" / "injury_ablation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
