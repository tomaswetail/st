"""Vectorize residual ML feature rows (numpy-only; no trainer import)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def vectorize_row_values(
    row: dict[str, Any],
    *,
    feature_names: list[str],
    global_medians: dict[str, float],
) -> np.ndarray:
    values: list[float] = []
    for name in feature_names:
        raw = row.get(name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or _is_nan(raw):
            values.append(float(global_medians.get(name, 0.0)))
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=float)


def vectorize_features(
    features: ResidualMLFeatures,
    *,
    feature_names: list[str],
    global_medians: dict[str, float],
) -> np.ndarray:
    row = features.to_dict()
    for key, value in row.items():
        if isinstance(value, date):
            row[key] = value.isoformat()
    return vectorize_row_values(
        row,
        feature_names=feature_names,
        global_medians=global_medians,
    )
