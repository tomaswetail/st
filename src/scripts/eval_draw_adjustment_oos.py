#!/usr/bin/env python3
"""Cheap offline OOS gate for Phase 3.2 draw adjustment.

Loads existing ``data/residual_ml/dataset.csv`` (no rebuild), time-splits
validation 0.20, excludes holdout draws ≥ HOLDOUT_DRAW_MIN, and compares
draw Brier / draw log-loss for blend vs blend+draw_adj.

Tries train-L1 coefficients and re-fit sparse offset betas on subsets of the
five stable discovery drivers. Writes ``artifacts/draw_adjustment_oos_gate.json``
and optionally updates ``config/draw_adjustment.json`` (enabled only if OOS improves).

```bash
export PYTHONPATH=src
python src/scripts/eval_draw_adjustment_oos.py
python src/scripts/eval_draw_adjustment_oos.py --write-config
```
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from calc.draw_adjustment import (
    DrawAdjustmentConfig,
    apply_draw_adjustment,
    load_draw_adjustment_config,
)
from calc.draw_driver_analysis import filter_tuning_rows, load_dataset_rows
from calc.probability_metrics import PROB_EPSILON
from config.eval_protocol import HOLDOUT_DRAW_MIN, VALIDATION_FRACTION
from utils.repo_paths import repo_root, resolve_repo_path
from utils.time_split import time_split_dataset_rows

# Stable L1 drivers from docs/reports/ml_draw/draw_driver_analysis.md (train coefs).
STABLE_L1_TRAIN_COEFS: dict[str, float] = {
    "away_npxg_for": -0.10955333695681589,
    "away_short_rest": 0.048328047446036984,
    "away_npxg_against": 0.04719263877508896,
    "combined_low_scoring_rate": -0.024298498667375875,
    "congestion_difference": 0.010077914452130719,
}

# Prefer larger |train| first when forming sparse subsets.
STABLE_DRIVER_ORDER = sorted(
    STABLE_L1_TRAIN_COEFS.keys(),
    key=lambda name: abs(STABLE_L1_TRAIN_COEFS[name]),
    reverse=True,
)


def _clip(probability: float) -> float:
    return min(1.0 - PROB_EPSILON, max(PROB_EPSILON, probability))


def _logit(probability: float) -> float:
    clipped = _clip(probability)
    return math.log(clipped / (1.0 - clipped))


def _inv_logit(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_draw(label: Any) -> int:
    return 1 if str(label).strip().upper() == "X" else 0


def _draw_brier(y_true: np.ndarray, p_draw: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p_draw))


def _draw_log_loss(y_true: np.ndarray, p_draw: np.ndarray) -> float:
    clipped = np.clip(p_draw, 1e-15, 1.0 - 1e-15)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def _feature_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=float)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(feature_names):
            value = _to_float(row.get(name))
            matrix[row_index, feature_index] = 0.0 if value is None else value
    return matrix


def _blend_draw(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [_clip(_to_float(row.get("p_draw_blend")) or PROB_EPSILON) for row in rows],
        dtype=float,
    )


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([_is_draw(row.get("label")) for row in rows], dtype=int)


def _adjusted_draw_probs(
    rows: list[dict[str, Any]],
    config: DrawAdjustmentConfig,
) -> np.ndarray:
    probs: list[float] = []
    for row in rows:
        blend = {
            "1": _clip(_to_float(row.get("p_home_blend")) or PROB_EPSILON),
            "X": _clip(_to_float(row.get("p_draw_blend")) or PROB_EPSILON),
            "2": _clip(_to_float(row.get("p_away_blend")) or PROB_EPSILON),
        }
        # Use pre-draw columns when present (future rebuilds); else current blend.
        pre_home = _to_float(row.get("p_home_blend_pre_draw"))
        pre_draw = _to_float(row.get("p_draw_blend_pre_draw"))
        pre_away = _to_float(row.get("p_away_blend_pre_draw"))
        if pre_home is not None and pre_draw is not None and pre_away is not None:
            blend = {
                "1": _clip(pre_home),
                "X": _clip(pre_draw),
                "2": _clip(pre_away),
            }
        adjusted = apply_draw_adjustment(blend, row, config)
        assert adjusted is not None
        probs.append(adjusted["X"])
    return np.array(probs, dtype=float)


def fit_offset_betas(
    train_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[float, dict[str, float]]:
    """Fit logit(p') = logit(p_blend) + β0 + Xβ on train (MLE)."""
    if not feature_names:
        return 0.0, {}

    x_train = _feature_matrix(train_rows, feature_names)
    p_blend = _blend_draw(train_rows)
    y_train = _labels(train_rows).astype(float)
    offsets = np.array([_logit(float(p)) for p in p_blend], dtype=float)

    def negative_log_likelihood(params: np.ndarray) -> float:
        intercept = float(params[0])
        coefs = params[1:]
        logits = offsets + intercept + x_train @ coefs
        probs = np.array([_inv_logit(float(value)) for value in logits], dtype=float)
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        return float(
            -np.mean(
                y_train * np.log(probs) + (1.0 - y_train) * np.log(1.0 - probs)
            )
        )

    initial = np.zeros(1 + len(feature_names), dtype=float)
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
    )
    params = result.x
    intercept = float(params[0])
    betas = {
        name: float(params[index + 1])
        for index, name in enumerate(feature_names)
    }
    return intercept, betas


@dataclass
class CandidateResult:
    name: str
    method: str
    feature_names: list[str]
    intercept: float
    betas: dict[str, float]
    validation_draw_brier_blend: float
    validation_draw_brier_adj: float
    validation_draw_ll_blend: float
    validation_draw_ll_adj: float
    train_draw_brier_blend: float
    train_draw_brier_adj: float
    train_draw_ll_blend: float
    train_draw_ll_adj: float
    improves_oos: bool

    @property
    def brier_delta(self) -> float:
        return self.validation_draw_brier_adj - self.validation_draw_brier_blend

    @property
    def ll_delta(self) -> float:
        return self.validation_draw_ll_adj - self.validation_draw_ll_blend


def evaluate_candidate(
    name: str,
    method: str,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    feature_names: list[str],
    intercept: float,
    betas: dict[str, float],
) -> CandidateResult:
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=intercept,
        features=betas,
    )
    y_train = _labels(train_rows)
    y_valid = _labels(valid_rows)
    p_blend_train = _blend_draw(train_rows)
    p_blend_valid = _blend_draw(valid_rows)
    p_adj_train = _adjusted_draw_probs(train_rows, config)
    p_adj_valid = _adjusted_draw_probs(valid_rows, config)

    val_brier_blend = _draw_brier(y_valid, p_blend_valid)
    val_brier_adj = _draw_brier(y_valid, p_adj_valid)
    val_ll_blend = _draw_log_loss(y_valid, p_blend_valid)
    val_ll_adj = _draw_log_loss(y_valid, p_adj_valid)
    improves = val_brier_adj < val_brier_blend or val_ll_adj < val_ll_blend

    return CandidateResult(
        name=name,
        method=method,
        feature_names=list(feature_names),
        intercept=intercept,
        betas=dict(betas),
        validation_draw_brier_blend=val_brier_blend,
        validation_draw_brier_adj=val_brier_adj,
        validation_draw_ll_blend=val_ll_blend,
        validation_draw_ll_adj=val_ll_adj,
        train_draw_brier_blend=_draw_brier(y_train, p_blend_train),
        train_draw_brier_adj=_draw_brier(y_train, p_adj_train),
        train_draw_ll_blend=_draw_log_loss(y_train, p_blend_train),
        train_draw_ll_adj=_draw_log_loss(y_train, p_adj_train),
        improves_oos=improves,
    )


def run_oos_gate(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    tuning_rows = filter_tuning_rows(rows, exclude_holdout=True)
    train_rows, valid_rows = time_split_dataset_rows(
        tuning_rows,
        validation_fraction=validation_fraction,
    )
    if len(train_rows) < 50 or len(valid_rows) < 20:
        raise ValueError(
            f"Insufficient rows after split: train={len(train_rows)} "
            f"valid={len(valid_rows)}"
        )

    candidates: list[CandidateResult] = []

    # (a) discovery train L1 coefs as raw offset betas (intercept 0)
    for size in (5, 4, 3, 2):
        names = STABLE_DRIVER_ORDER[:size]
        betas = {name: STABLE_L1_TRAIN_COEFS[name] for name in names}
        candidates.append(
            evaluate_candidate(
                name=f"l1_train_coefs_top{size}",
                method="discovery_l1_train_coefs",
                train_rows=train_rows,
                valid_rows=valid_rows,
                feature_names=names,
                intercept=0.0,
                betas=betas,
            )
        )

    # (b) re-fit exact offset form on train only
    for size in (5, 4, 3, 2, 1):
        names = STABLE_DRIVER_ORDER[:size]
        intercept, betas = fit_offset_betas(train_rows, names)
        candidates.append(
            evaluate_candidate(
                name=f"refit_offset_top{size}",
                method="train_mle_offset",
                train_rows=train_rows,
                valid_rows=valid_rows,
                feature_names=names,
                intercept=intercept,
                betas=betas,
            )
        )

    improving = [item for item in candidates if item.improves_oos]

    def score(item: CandidateResult) -> tuple[float, float]:
        # Prefer lower validation draw LL, then lower Brier.
        return (item.validation_draw_ll_adj, item.validation_draw_brier_adj)

    if improving:
        best = min(improving, key=score)
        enabled = True
        gate_result = "PASS"
    else:
        best = min(candidates, key=score)
        enabled = False
        gate_result = "FAIL"

    config_payload = {
        "enabled": enabled,
        "intercept": best.intercept,
        "beta0": best.intercept,
        "features": [
            {"name": name, "beta": best.betas[name]} for name in best.feature_names
        ],
        "missing_feature_default": 0.0,
        "notes": (
            f"OOS gate {gate_result}: candidate={best.name} method={best.method}. "
            f"Val draw Brier blend={best.validation_draw_brier_blend:.6f} "
            f"adj={best.validation_draw_brier_adj:.6f} "
            f"(Δ={best.brier_delta:+.6f}); "
            f"Val draw LL blend={best.validation_draw_ll_blend:.6f} "
            f"adj={best.validation_draw_ll_adj:.6f} "
            f"(Δ={best.ll_delta:+.6f}). "
            "Phase 3.1 discovery full-L1 OOS remains FAIL. "
            "Full dataset rebuild + HGB retrain deferred unless run separately."
        ),
    }

    return {
        "gate_result": gate_result,
        "enabled": enabled,
        "holdout_draw_min": HOLDOUT_DRAW_MIN,
        "validation_fraction": validation_fraction,
        "n_train": len(train_rows),
        "n_validation": len(valid_rows),
        "n_tuning": len(tuning_rows),
        "best_candidate": asdict(best),
        "candidates": [asdict(item) for item in candidates],
        "recommended_config": config_payload,
    }


def write_config(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/residual_ml/dataset.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/draw_adjustment_oos_gate.json"),
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path("config/draw_adjustment.json"),
        help="Config path updated when --write-config is set",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write recommended config (enabled only if OOS improves)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION,
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    config_path = resolve_repo_path(args.config_out)

    rows = load_dataset_rows(dataset_path)
    report = run_oos_gate(rows, validation_fraction=args.validation_fraction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.write_config:
        write_config(report["recommended_config"], config_path)

    best = report["best_candidate"]
    print(
        f"OOS gate: {report['gate_result']} enabled={report['enabled']} "
        f"candidate={best['name']}",
        flush=True,
    )
    print(
        f"Val draw Brier blend={best['validation_draw_brier_blend']:.6f} "
        f"adj={best['validation_draw_brier_adj']:.6f} "
        f"Δ={best['validation_draw_brier_adj'] - best['validation_draw_brier_blend']:+.6f}",
        flush=True,
    )
    print(
        f"Val draw LL blend={best['validation_draw_ll_blend']:.6f} "
        f"adj={best['validation_draw_ll_adj']:.6f} "
        f"Δ={best['validation_draw_ll_adj'] - best['validation_draw_ll_blend']:+.6f}",
        flush=True,
    )
    print(f"Wrote {output_path}", flush=True)
    if args.write_config:
        print(f"Wrote {config_path}", flush=True)
    else:
        # Still ensure a disabled-safe default config exists for runtime.
        existing = load_draw_adjustment_config(config_path)
        if not config_path.exists():
            write_config(report["recommended_config"], config_path)
            print(f"Wrote missing config {config_path}", flush=True)
        else:
            print(
                f"Config unchanged ({config_path}); enabled={existing.enabled}. "
                "Pass --write-config to apply gate recommendation.",
                flush=True,
            )


if __name__ == "__main__":
    main()
