"""Native, permutation, and optional built-in SHAP-style importances."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.calc.probability_metrics import mean_log_loss
from src.calc.residual_ml.models.base import rows_to_matrix
from src.utils.common import OUTCOMES


def fitted_estimator(registry_model: Any) -> Any | None:
    """Return the sklearn/boosting estimator stored on a registry model."""
    estimator = getattr(registry_model, "model", None)
    if estimator is not None:
        return estimator
    trainer = getattr(registry_model, "_trainer", None)
    if trainer is not None:
        return getattr(trainer, "model", None)
    return None


def _mean_importances(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array
    return np.mean(np.abs(array), axis=0)


def native_feature_importance(
    estimator: Any,
    feature_names: list[str],
) -> list[tuple[str, float]] | None:
    """Native importances when the estimator exposes them."""
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "estimators_") and estimator.estimators_:
        child_values = []
        for child in estimator.estimators_:
            if hasattr(child, "feature_importances_"):
                child_values.append(np.asarray(child.feature_importances_, dtype=float))
        if child_values:
            values = np.mean(np.vstack(child_values), axis=0)
    elif hasattr(estimator, "get_feature_importance"):
        try:
            values = np.asarray(estimator.get_feature_importance(), dtype=float)
        except Exception:
            values = None
    if values is None or values.size == 0:
        return None
    if values.ndim > 1:
        values = _mean_importances(values)
    if values.size != len(feature_names):
        feature_names = feature_names[: values.size]
    return sorted(
        zip(feature_names, (float(value) for value in values)),
        key=lambda item: item[1],
        reverse=True,
    )


class _LogLossProbaAdapter:
    """Adapter so permutation importance can call ``predict_proba(X)``."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.estimator.predict_proba(X), dtype=float)
        classes = [str(label) for label in getattr(self.estimator, "classes_", OUTCOMES)]
        mapped = np.zeros((raw.shape[0], 3), dtype=float)
        for column, outcome in enumerate(OUTCOMES):
            if outcome in classes:
                mapped[:, column] = raw[:, classes.index(outcome)]
        totals = mapped.sum(axis=1, keepdims=True)
        totals[totals <= 0] = 1.0
        return mapped / totals


def _neg_log_loss_scorer(estimator: Any, X, y) -> float:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    triples = [
        (float(row[0]), float(row[1]), float(row[2])) for row in probabilities
    ]
    labels = [str(label) for label in y]
    return -mean_log_loss(labels, triples)


def permutation_log_loss_importance(
    estimator: Any,
    feature_matrix: np.ndarray,
    labels: list[str],
    feature_names: list[str],
    *,
    random_state: int = 42,
    n_repeats: int = 5,
) -> list[tuple[str, float]]:
    """Permutation importance = increase in log loss when a feature is shuffled."""
    from sklearn.inspection import permutation_importance

    adapter = estimator
    if not isinstance(estimator, _LogLossProbaAdapter):
        if hasattr(estimator, "predict_proba") and hasattr(estimator, "classes_"):
            adapter = _LogLossProbaAdapter(estimator)
    result = permutation_importance(
        adapter,
        feature_matrix,
        np.asarray(labels, dtype=object),
        scoring=_neg_log_loss_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    names = feature_names[: len(result.importances_mean)]
    return sorted(
        zip(names, (float(value) for value in result.importances_mean)),
        key=lambda item: item[1],
        reverse=True,
    )


def shap_feature_importance(
    estimator: Any,
    feature_matrix: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]] | None:
    """Thin CatBoost/LightGBM SHAP wrapper. Does not import the ``shap`` package."""
    try:
        if hasattr(estimator, "get_feature_importance"):
            from catboost import EFstrType, Pool

            shap_values = estimator.get_feature_importance(
                data=Pool(feature_matrix),
                type=EFstrType.ShapValues,
            )
            array = np.asarray(shap_values, dtype=float)
            if array.ndim == 3:
                # (n, classes, features+bias)
                array = np.mean(np.abs(array[:, :, :-1]), axis=(0, 1))
            elif array.ndim == 2:
                array = np.mean(np.abs(array[:, :-1]), axis=0)
            else:
                return None
            names = feature_names[: array.size]
            return sorted(
                zip(names, (float(value) for value in array)),
                key=lambda item: item[1],
                reverse=True,
            )
        if hasattr(estimator, "predict"):
            contrib = estimator.predict(feature_matrix, pred_contrib=True)
            array = np.asarray(contrib, dtype=float)
            if array.ndim == 3:
                array = np.mean(np.abs(array[:, :, :-1]), axis=(0, 1))
            elif array.ndim == 2:
                array = np.mean(np.abs(array[:, :-1]), axis=0)
            else:
                return None
            names = feature_names[: array.size]
            return sorted(
                zip(names, (float(value) for value in array)),
                key=lambda item: item[1],
                reverse=True,
            )
    except Exception:
        return None
    return None


def importance_from_registry_model(
    registry_model: Any,
    validate_rows: list[dict[str, Any]],
    *,
    random_state: int = 42,
    n_repeats: int = 5,
) -> dict[str, list[tuple[str, float]] | None]:
    """Native + permutation (+ SHAP if free) on one fold's validate matrix."""
    estimator = fitted_estimator(registry_model)
    feature_names = list(getattr(registry_model, "feature_names", []) or [])
    medians = getattr(registry_model, "global_medians", None)
    if estimator is None or not feature_names or medians is None:
        trainer = getattr(registry_model, "_trainer", None)
        if trainer is not None:
            feature_names = list(trainer.feature_names)
            medians = trainer.global_medians
            estimator = fitted_estimator(registry_model)
    if estimator is None or not feature_names or not medians:
        raise ValueError("registry model has no fitted estimator / feature matrix")

    matrix = rows_to_matrix(
        validate_rows,
        feature_names=feature_names,
        global_medians=medians,
    )
    labels = [str(row.get("label", "")).strip().upper() for row in validate_rows]
    native = native_feature_importance(estimator, feature_names)
    permutation = None
    if hasattr(estimator, "predict_proba") and hasattr(estimator, "classes_"):
        permutation = permutation_log_loss_importance(
            estimator,
            matrix,
            labels,
            feature_names,
            random_state=random_state,
            n_repeats=n_repeats,
        )
    shap_values = shap_feature_importance(estimator, matrix, feature_names)
    return {
        "native": native,
        "permutation": permutation,
        "shap": shap_values,
    }
