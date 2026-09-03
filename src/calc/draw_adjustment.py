"""Explicit draw (X) logit adjustment on blended 1X2 probabilities.

Form:
    logit(p_draw') = logit(p_draw_blend) + β0 + Σ βi * feature_i
then renormalize {1, X, 2} to sum to 1.

Missing feature values default to 0.0 (never raise). Disabled config is a no-op.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from calc.probability_metrics import PROB_EPSILON
from utils.common import OUTCOMES
from utils.repo_paths import repo_root, resolve_repo_path

DEFAULT_CONFIG_PATH = repo_root() / "config" / "draw_adjustment.json"
MISSING_FEATURE_DEFAULT = 0.0


@dataclass(frozen=True)
class DrawAdjustmentConfig:
    """Coefficients for blend draw-probability adjustment."""

    enabled: bool = False
    intercept: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    missing_feature_default: float = MISSING_FEATURE_DEFAULT
    notes: str = ""

    @property
    def beta0(self) -> float:
        return self.intercept


def _clip_prob(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    return min(1.0 - epsilon, max(epsilon, probability))


def _logit(probability: float, *, epsilon: float = PROB_EPSILON) -> float:
    clipped = _clip_prob(probability, epsilon=epsilon)
    return math.log(clipped / (1.0 - clipped))


def _inv_logit(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _normalize(probabilities: Mapping[str, float]) -> dict[str, float] | None:
    present = {
        key: float(value)
        for key, value in probabilities.items()
        if key in OUTCOMES
    }
    if len(present) != 3:
        return None
    total = sum(present.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in present.items()}


def _parse_features(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): float(beta) for name, beta in raw.items()}
    if isinstance(raw, list):
        parsed: dict[str, float] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid feature entry: {item!r}")
            name = item.get("name")
            if name is None:
                raise ValueError(f"Feature entry missing name: {item!r}")
            if "beta" in item:
                beta = item["beta"]
            elif "coef" in item:
                beta = item["coef"]
            else:
                raise ValueError(f"Feature entry missing beta: {item!r}")
            parsed[str(name)] = float(beta)
        return parsed
    raise ValueError(f"Unsupported features payload type: {type(raw)!r}")


def load_draw_adjustment_config(
    path: str | Path | None = None,
) -> DrawAdjustmentConfig:
    """Load draw-adjustment config from JSON (repo-relative or absolute)."""
    config_path = resolve_repo_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return DrawAdjustmentConfig(enabled=False, notes="config file missing")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    intercept = payload.get("intercept", payload.get("beta0", 0.0))
    return DrawAdjustmentConfig(
        enabled=bool(payload.get("enabled", False)),
        intercept=float(intercept if intercept is not None else 0.0),
        features=_parse_features(payload.get("features")),
        missing_feature_default=float(
            payload.get("missing_feature_default", MISSING_FEATURE_DEFAULT)
        ),
        notes=str(payload.get("notes", "") or ""),
    )


def feature_value(
    features: Any,
    name: str,
    *,
    default: float = MISSING_FEATURE_DEFAULT,
) -> float:
    """Read a numeric feature from a mapping or object; missing → default."""
    raw: Any
    if isinstance(features, Mapping):
        raw = features.get(name, default)
    else:
        raw = getattr(features, name, default)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def draw_logit_delta(
    features: Any,
    config: DrawAdjustmentConfig,
) -> float:
    """β0 + Σ βi * feature_i (missing features use configured default)."""
    delta = float(config.intercept)
    default = config.missing_feature_default
    for name, beta in config.features.items():
        delta += float(beta) * feature_value(features, name, default=default)
    return delta


def apply_draw_adjustment(
    blend: Mapping[str, float] | None,
    features: Any,
    config: DrawAdjustmentConfig | None = None,
) -> dict[str, float] | None:
    """Adjust blend draw logit, then renormalize 1/X/2.

    When ``config`` is None, loads ``config/draw_adjustment.json``.
    When disabled or blend is None, returns blend unchanged (copy if present).
    """
    if blend is None:
        return None
    resolved = config if config is not None else load_draw_adjustment_config()
    if not resolved.enabled:
        return {outcome: float(blend[outcome]) for outcome in OUTCOMES if outcome in blend}

    if any(outcome not in blend for outcome in OUTCOMES):
        return {key: float(value) for key, value in blend.items()}

    delta = draw_logit_delta(features, resolved)
    adjusted_draw = _inv_logit(_logit(float(blend["X"])) + delta)
    renormalized = _normalize(
        {
            "1": float(blend["1"]),
            "X": adjusted_draw,
            "2": float(blend["2"]),
        }
    )
    if renormalized is None:
        return {outcome: float(blend[outcome]) for outcome in OUTCOMES}
    return renormalized
