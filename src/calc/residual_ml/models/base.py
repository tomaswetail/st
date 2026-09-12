"""Shared registry contract helpers (features, medians, clip+normalize)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from src.calc.probability_metrics import clip_and_normalize_probs
from src.calc.residual_ml.baseline import OUTCOMES
from src.calc.residual_ml.trainer import ResidualMLTrainer, market_from_row
from src.calc.residual_ml.vectorize import vectorize_row_values

Family = str  # "market" | "direct" | "residual"


class RegistryModel(Protocol):
    """Seeded backend with a shared 1X2 ``predict_proba`` contract."""

    name: str
    family: str
    feature_names: list[str]

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        ...

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        ...


def as_unit_probs(probabilities: Mapping[str, float]) -> dict[str, float]:
    p_home, p_draw, p_away = clip_and_normalize_probs(
        float(probabilities["1"]),
        float(probabilities["X"]),
        float(probabilities["2"]),
    )
    return {"1": p_home, "X": p_draw, "2": p_away}


def resolve_baseline(
    row: dict[str, Any],
    baseline: Mapping[str, float] | None,
) -> dict[str, float]:
    if baseline is not None:
        return as_unit_probs(baseline)
    market = market_from_row(row)
    if market is None:
        raise ValueError("Missing market baseline for predict_proba")
    return as_unit_probs(market)


def prepare_feature_state(
    train_rows: list[dict[str, Any]],
    *,
    include_market_norm: bool,
) -> tuple[list[str], dict[str, float]]:
    if not train_rows:
        raise ValueError("Training dataset is empty")
    feature_names = ResidualMLTrainer.feature_names_from_rows(
        train_rows,
        include_market_norm=include_market_norm,
    )
    medians = ResidualMLTrainer._compute_medians(train_rows, feature_names)
    return feature_names, medians


def rows_to_matrix(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    global_medians: dict[str, float],
):
    import numpy as np

    return np.vstack(
        [
            vectorize_row_values(
                row,
                feature_names=feature_names,
                global_medians=global_medians,
            )
            for row in rows
        ]
    )


def classifier_predict_proba(estimator: Any, vector) -> dict[str, float]:
    raw = estimator.predict_proba(vector.reshape(1, -1))[0]
    by_class = {str(label): float(probability) for label, probability in zip(estimator.classes_, raw)}
    return as_unit_probs(
        {
            outcome: by_class.get(outcome, 0.0)
            for outcome in OUTCOMES
        }
    )
