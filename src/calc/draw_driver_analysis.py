"""Interpretable draw-driver discovery (L1 logistic + draw-error regression).

Analysis-only: finds features that move draw probability beyond the blend
baseline. Does not change production prediction math (that is Phase 3.2 ship).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from config.eval_protocol import HOLDOUT_DRAW_MIN, VALIDATION_FRACTION
from src.utils.time_split import time_split_dataset_rows

# Baselines always included so coefficients are incremental vs odds+DC blend.
BASELINE_FEATURES = ("p_draw_blend",)

CANDIDATE_DRIVER_FEATURES = (
    "league_draw_rate",
    "combined_draw_rate",
    "combined_close_match_rate",
    "combined_low_scoring_rate",
    "combined_one_goal_match_rate",
    "expected_goal_total",
    "expected_goal_difference",
    "market_vs_dc_draw",
    "market_balance",
    "favourite_strength",
    "home_npxg_for",
    "away_npxg_for",
    "home_npxg_against",
    "away_npxg_against",
    "rest_day_difference",
    "home_short_rest",
    "away_short_rest",
    "congestion_difference",
    "home_advantage_log",
    "league_avg_goals",
    "league_competitive_balance",
)

METADATA_COLUMNS = ("match_id", "draw_number", "match_date", "label")


@dataclass
class DriverCoefficient:
    feature: str
    train_coef: float
    validation_coef: float | None
    stable: bool
    same_sign: bool


@dataclass
class DrawDiscoveryResult:
    n_train: int
    n_validation: int
    feature_names: list[str]
    l1_coefficients: list[DriverCoefficient]
    elastic_net_coefficients: list[DriverCoefficient]
    draw_error_coefficients: list[DriverCoefficient]
    train_draw_brier_blend: float
    validation_draw_brier_blend: float
    train_draw_brier_model: float
    validation_draw_brier_model: float
    train_draw_log_loss_blend: float
    validation_draw_log_loss_blend: float
    train_draw_log_loss_model: float
    validation_draw_log_loss_model: float
    stable_drivers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "feature_names": self.feature_names,
            "l1_coefficients": [asdict(item) for item in self.l1_coefficients],
            "elastic_net_coefficients": [
                asdict(item) for item in self.elastic_net_coefficients
            ],
            "draw_error_coefficients": [
                asdict(item) for item in self.draw_error_coefficients
            ],
            "metrics": {
                "train_draw_brier_blend": self.train_draw_brier_blend,
                "validation_draw_brier_blend": self.validation_draw_brier_blend,
                "train_draw_brier_model": self.train_draw_brier_model,
                "validation_draw_brier_model": self.validation_draw_brier_model,
                "train_draw_log_loss_blend": self.train_draw_log_loss_blend,
                "validation_draw_log_loss_blend": self.validation_draw_log_loss_blend,
                "train_draw_log_loss_model": self.train_draw_log_loss_model,
                "validation_draw_log_loss_model": self.validation_draw_log_loss_model,
            },
            "stable_drivers": self.stable_drivers,
            "notes": self.notes,
        }


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def filter_tuning_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_holdout: bool = True,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("label")
        if label not in {"1", "X", "2"}:
            continue
        if _to_float(row.get("p_draw_blend")) is None:
            continue
        if exclude_holdout:
            draw_number = _to_float(row.get("draw_number"))
            if draw_number is not None and int(draw_number) >= HOLDOUT_DRAW_MIN:
                continue
        filtered.append(row)
    return filtered


def build_analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach is_draw and draw_surprise to each row (analysis table)."""
    analysis: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("label")
        is_draw = 1 if label == "X" else 0
        p_draw_blend = _to_float(row.get("p_draw_blend"))
        if p_draw_blend is None:
            continue
        enriched = dict(row)
        enriched["is_draw"] = is_draw
        enriched["draw_surprise"] = float(is_draw) - p_draw_blend
        analysis.append(enriched)
    return analysis


def resolve_feature_names(
    rows: list[dict[str, Any]],
    *,
    candidate_features: tuple[str, ...] = CANDIDATE_DRIVER_FEATURES,
) -> list[str]:
    present = []
    for name in BASELINE_FEATURES:
        if any(_to_float(row.get(name)) is not None for row in rows):
            present.append(name)
    for name in candidate_features:
        if name in BASELINE_FEATURES:
            continue
        if any(_to_float(row.get(name)) is not None for row in rows):
            present.append(name)
    return present


def _median_fill(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for column_index in range(filled.shape[1]):
        column = filled[:, column_index]
        valid = column[~np.isnan(column)]
        fill = float(np.median(valid)) if len(valid) else 0.0
        column[np.isnan(column)] = fill
        filled[:, column_index] = column
    return filled


def design_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.full((len(rows), len(feature_names)), np.nan, dtype=float)
    labels = np.zeros(len(rows), dtype=int)
    surprises = np.zeros(len(rows), dtype=float)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(feature_names):
            value = _to_float(row.get(name))
            if value is not None:
                features[row_index, feature_index] = value
        labels[row_index] = int(row["is_draw"])
        surprises[row_index] = float(row["draw_surprise"])
    return _median_fill(features), labels, surprises


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l1_ratio: float,
    C: float = 0.5,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit saga logistic; l1_ratio=1 is L1, (0,1) is elastic-net (sklearn ≥1.8)."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver="saga",
        max_iter=5000,
        random_state=42,
    )
    model.fit(x_scaled, y_train)
    return model, scaler


def _coef_pairs(
    feature_names: list[str],
    train_coef: np.ndarray,
    validation_coef: np.ndarray | None,
    *,
    stability_ratio: float = 0.5,
) -> list[DriverCoefficient]:
    rows: list[DriverCoefficient] = []
    for index, name in enumerate(feature_names):
        train_value = float(train_coef[index])
        val_value = (
            float(validation_coef[index]) if validation_coef is not None else None
        )
        same_sign = False
        stable = False
        if val_value is not None and train_value != 0.0 and val_value != 0.0:
            same_sign = (train_value > 0) == (val_value > 0)
            magnitude_ok = abs(val_value) >= stability_ratio * abs(train_value) or abs(
                train_value
            ) >= stability_ratio * abs(val_value)
            # Non-baseline sparse drivers: both non-zero and same sign
            stable = same_sign and magnitude_ok and name not in BASELINE_FEATURES
        rows.append(
            DriverCoefficient(
                feature=name,
                train_coef=train_value,
                validation_coef=val_value,
                stable=stable,
                same_sign=same_sign,
            )
        )
    return sorted(rows, key=lambda item: abs(item.train_coef), reverse=True)


def _draw_brier(y_true: np.ndarray, p_draw: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p_draw))


def _draw_log_loss(y_true: np.ndarray, p_draw: np.ndarray) -> float:
    clipped = np.clip(p_draw, 1e-15, 1.0 - 1e-15)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def run_draw_discovery(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    exclude_holdout: bool = True,
    l1_C: float = 0.5,
    elastic_C: float = 0.5,
) -> DrawDiscoveryResult:
    tuning_rows = filter_tuning_rows(rows, exclude_holdout=exclude_holdout)
    analysis_rows = build_analysis_rows(tuning_rows)
    if len(analysis_rows) < 50:
        raise ValueError(
            f"Need at least 50 analysis rows, got {len(analysis_rows)}"
        )

    train_rows, valid_rows = time_split_dataset_rows(
        analysis_rows,
        validation_fraction=validation_fraction,
    )
    feature_names = resolve_feature_names(train_rows)
    if "p_draw_blend" not in feature_names:
        raise ValueError("p_draw_blend required for incremental draw discovery")

    x_train, y_train, surprise_train = design_matrix(train_rows, feature_names)
    x_valid, y_valid, surprise_valid = design_matrix(valid_rows, feature_names)

    # --- L1 logistic ---
    l1_model, l1_scaler = _fit_logistic(x_train, y_train, l1_ratio=1.0, C=l1_C)
    l1_val_model, _ = _fit_logistic(x_valid, y_valid, l1_ratio=1.0, C=l1_C)
    l1_coefs = _coef_pairs(
        feature_names,
        l1_model.coef_.ravel(),
        l1_val_model.coef_.ravel(),
    )

    # --- Elastic net ---
    en_model, en_scaler = _fit_logistic(
        x_train, y_train, l1_ratio=0.5, C=elastic_C
    )
    en_val_model, _ = _fit_logistic(
        x_valid, y_valid, l1_ratio=0.5, C=elastic_C
    )
    en_coefs = _coef_pairs(
        feature_names,
        en_model.coef_.ravel(),
        en_val_model.coef_.ravel(),
    )

    # --- Draw-error regression (surprise vs drivers, excluding raw blend) ---
    driver_indices = [
        index
        for index, name in enumerate(feature_names)
        if name not in BASELINE_FEATURES
    ]
    notes: list[str] = []
    if driver_indices:
        x_train_err = x_train[:, driver_indices]
        x_valid_err = x_valid[:, driver_indices]
        err_names = [feature_names[index] for index in driver_indices]
        err_scaler = StandardScaler()
        x_train_err_s = err_scaler.fit_transform(x_train_err)
        x_valid_err_s = err_scaler.transform(x_valid_err)
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(x_train_err_s, surprise_train)
        ridge_val = Ridge(alpha=1.0, random_state=42)
        ridge_val.fit(x_valid_err_s, surprise_valid)
        error_coefs = _coef_pairs(
            err_names,
            ridge.coef_.ravel(),
            ridge_val.coef_.ravel(),
        )
    else:
        error_coefs = []
        notes.append("No non-baseline drivers available for draw-error regression")

    # Metrics: blend-only vs L1 predicted P(draw)
    p_blend_train = np.array(
        [_to_float(row["p_draw_blend"]) or 0.0 for row in train_rows],
        dtype=float,
    )
    p_blend_valid = np.array(
        [_to_float(row["p_draw_blend"]) or 0.0 for row in valid_rows],
        dtype=float,
    )
    p_model_train = l1_model.predict_proba(l1_scaler.transform(x_train))[:, 1]
    p_model_valid = l1_model.predict_proba(l1_scaler.transform(x_valid))[:, 1]

    stable = [
        item.feature
        for item in l1_coefs
        if item.stable and abs(item.train_coef) > 1e-6
    ]

    return DrawDiscoveryResult(
        n_train=len(train_rows),
        n_validation=len(valid_rows),
        feature_names=feature_names,
        l1_coefficients=l1_coefs,
        elastic_net_coefficients=en_coefs,
        draw_error_coefficients=error_coefs,
        train_draw_brier_blend=_draw_brier(y_train, p_blend_train),
        validation_draw_brier_blend=_draw_brier(y_valid, p_blend_valid),
        train_draw_brier_model=_draw_brier(y_train, p_model_train),
        validation_draw_brier_model=_draw_brier(y_valid, p_model_valid),
        train_draw_log_loss_blend=_draw_log_loss(y_train, p_blend_train),
        validation_draw_log_loss_blend=_draw_log_loss(y_valid, p_blend_valid),
        train_draw_log_loss_model=_draw_log_loss(y_train, p_model_train),
        validation_draw_log_loss_model=_draw_log_loss(y_valid, p_model_valid),
        stable_drivers=stable,
        notes=notes,
    )


def write_discovery_artifacts(
    result: DrawDiscoveryResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary_path = output_dir / "discovery_summary.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    paths["summary"] = summary_path

    l1_path = output_dir / "l1_coefficients.json"
    l1_path.write_text(
        json.dumps([asdict(item) for item in result.l1_coefficients], indent=2),
        encoding="utf-8",
    )
    paths["l1"] = l1_path

    en_path = output_dir / "elastic_net_coefficients.json"
    en_path.write_text(
        json.dumps(
            [asdict(item) for item in result.elastic_net_coefficients],
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["elastic_net"] = en_path

    err_path = output_dir / "draw_error_coefficients.json"
    err_path.write_text(
        json.dumps(
            [asdict(item) for item in result.draw_error_coefficients],
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["draw_error"] = err_path

    # Flat CSV for permutation-style ranking (L1 abs coef as importance proxy)
    importance_path = output_dir / "permutation_importance.csv"
    with importance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature",
                "importance_abs_train_coef",
                "train_coef",
                "validation_coef",
                "stable",
            ],
        )
        writer.writeheader()
        for item in result.l1_coefficients:
            writer.writerow(
                {
                    "feature": item.feature,
                    "importance_abs_train_coef": abs(item.train_coef),
                    "train_coef": item.train_coef,
                    "validation_coef": item.validation_coef,
                    "stable": item.stable,
                }
            )
    paths["importance"] = importance_path
    return paths


def render_draw_driver_markdown(result: DrawDiscoveryResult) -> str:
    lines = [
        "# Draw driver analysis",
        "",
        "Phase 3.1 discovery (interpretable models only). "
        "HGB residual engine is unchanged.",
        "",
        "## Protocol",
        "",
        f"- Train / validation time split: last **{VALIDATION_FRACTION:.0%}** by `match_date`",
        f"- Holdout draws ≥ {HOLDOUT_DRAW_MIN} excluded from discovery",
        "- Models always include `p_draw_blend` (incremental vs odds+DC)",
        "",
        "## Sample sizes",
        "",
        f"| Split | Rows |",
        f"|-------|------|",
        f"| Train | {result.n_train} |",
        f"| Validation | {result.n_validation} |",
        "",
        "## Draw metrics (binary X vs not-X)",
        "",
        "| Metric | Blend-only | L1 logistic | Δ |",
        "|--------|------------|-------------|---|",
        (
            f"| Val draw Brier | {result.validation_draw_brier_blend:.4f} | "
            f"{result.validation_draw_brier_model:.4f} | "
            f"{result.validation_draw_brier_model - result.validation_draw_brier_blend:+.4f} |"
        ),
        (
            f"| Val draw log loss | {result.validation_draw_log_loss_blend:.4f} | "
            f"{result.validation_draw_log_loss_model:.4f} | "
            f"{result.validation_draw_log_loss_model - result.validation_draw_log_loss_blend:+.4f} |"
        ),
        (
            f"| Train draw Brier | {result.train_draw_brier_blend:.4f} | "
            f"{result.train_draw_brier_model:.4f} | "
            f"{result.train_draw_brier_model - result.train_draw_brier_blend:+.4f} |"
        ),
        "",
        "## Discovery gate",
        "",
    ]
    oos_brier_delta = (
        result.validation_draw_brier_model - result.validation_draw_brier_blend
    )
    oos_ll_delta = (
        result.validation_draw_log_loss_model
        - result.validation_draw_log_loss_blend
    )
    stable_ok = len(result.stable_drivers) >= 3
    oos_ok = oos_brier_delta < 0 or oos_ll_delta < 0
    lines.extend(
        [
            f"| Gate | Result |",
            f"|------|--------|",
            (
                f"| ≥3 stable drivers | "
                f"{'PASS' if stable_ok else 'FAIL'} "
                f"({len(result.stable_drivers)}) |"
            ),
            (
                f"| OOS draw Brier or LL improves vs blend | "
                f"{'PASS' if oos_ok else 'FAIL'} "
                f"(Brier Δ={oos_brier_delta:+.4f}, LL Δ={oos_ll_delta:+.4f}) |"
            ),
            "",
            (
                "Home/away LL regression gate applies only after Phase 3.2 ships "
                "an explicit draw adjustment."
            ),
            "",
            "## Stable L1 drivers (same sign train+val, non-baseline)",
            "",
        ]
    )
    if result.stable_drivers:
        lines.append("| Feature | Train coef | Val coef |")
        lines.append("|---------|------------|----------|")
        by_name = {item.feature: item for item in result.l1_coefficients}
        for name in result.stable_drivers:
            item = by_name[name]
            lines.append(
                f"| `{name}` | {item.train_coef:+.4f} | "
                f"{item.validation_coef:+.4f} |"
            )
    else:
        lines.append("_No drivers met the stability gate._")
    lines.extend(
        [
            "",
            f"**Stable count:** {len(result.stable_drivers)} "
            f"(discovery gate wants ≥ 3)",
            "",
            "## Top L1 coefficients (by |train|)",
            "",
            "| Feature | Train | Val | Stable |",
            "|---------|-------|-----|--------|",
        ]
    )
    for item in result.l1_coefficients[:15]:
        val = (
            f"{item.validation_coef:+.4f}"
            if item.validation_coef is not None
            else "—"
        )
        lines.append(
            f"| `{item.feature}` | {item.train_coef:+.4f} | {val} | "
            f"{'yes' if item.stable else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Top draw-error (surprise) Ridge coefficients",
            "",
            "Target: `is_draw - p_draw_blend` (positive = blend undercalls X).",
            "",
            "| Feature | Train | Val |",
            "|---------|-------|-----|",
        ]
    )
    for item in result.draw_error_coefficients[:10]:
        val = (
            f"{item.validation_coef:+.4f}"
            if item.validation_coef is not None
            else "—"
        )
        lines.append(
            f"| `{item.feature}` | {item.train_coef:+.4f} | {val} |"
        )
    if result.notes:
        lines.extend(["", "## Notes", ""])
        for note in result.notes:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Next step (Phase 3.2)",
            "",
            (
                "Ship only if OOS gate passes (or after a re-tuned sparse subset "
                "beats blend). Preferred: 3–8 stable terms into "
                "`config/draw_adjustment.json` via `calc/draw_adjustment.py` "
                "(logit adjust on blend, then renormalize)."
            ),
            "",
        ]
    )
    return "\n".join(lines)
