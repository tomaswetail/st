"""Shared registry fit/score helpers for smoke and compare CLIs."""

from __future__ import annotations

from typing import Any

from src.calc.probability_metrics import (
    mean_accuracy,
    mean_log_loss,
    multiclass_brier,
    top_label_ece,
)
from src.calc.residual_ml.models import get_model
from src.calc.residual_ml.trainer import market_from_row

SMOKE_TREE_ITER = 100
SMOKE_LOGISTIC_MAX_ITER = 200
VALID_LABELS = frozenset({"1", "X", "2"})
MARKET_BACKENDS = frozenset({"market", "market_baseline"})


def families_from_choice(choice: str) -> list[str]:
    if choice == "both":
        return ["direct", "residual"]
    return [choice]


def build_registry_model(backend: str, family: str, *, seed: int = 42) -> Any:
    iter_cap = SMOKE_LOGISTIC_MAX_ITER if backend == "logistic" else SMOKE_TREE_ITER
    return get_model(
        backend,
        family,
        random_state=seed,
        max_iter=iter_cap,
        n_estimators=SMOKE_TREE_ITER,
        iterations=SMOKE_TREE_ITER,
    )


def collect_scored_rows(
    rows: list[dict[str, Any]],
    predict: Any | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[tuple[float, float, float]]]:
    """Score rows that have a market baseline and a 1X2 label.

    ``predict`` is ``None`` for market-only (use stored market columns).
    Otherwise it is a registry model with ``predict_proba(row, market)``.
    """
    scored_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        market = market_from_row(row)
        if market is None:
            continue
        label = str(row.get("label", "")).strip().upper()
        if label not in VALID_LABELS:
            continue
        if predict is None:
            predicted = market
        else:
            predicted = predict.predict_proba(row, market)
        scored_rows.append(row)
        labels.append(label)
        probability_rows.append(
            (float(predicted["1"]), float(predicted["X"]), float(predicted["2"]))
        )
    return scored_rows, labels, probability_rows


def metrics_from_predictions(
    labels: list[str],
    probability_rows: list[tuple[float, float, float]],
) -> dict[str, float]:
    if not labels:
        raise ValueError("no scorable predicted rows")
    return {
        "n_val": float(len(labels)),
        "logloss": mean_log_loss(labels, probability_rows),
        "brier": multiclass_brier(labels, probability_rows),
        "acc": mean_accuracy(labels, probability_rows),
        "ece": top_label_ece(labels, probability_rows),
    }


def is_market_backend(backend: str) -> bool:
    return backend in MARKET_BACKENDS
