"""Registry backends: market passthrough, family A classifiers, family B residuals."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.calc.residual_ml.baseline import OUTCOMES, apply_residual_deltas, target_logit_deltas
from src.calc.residual_ml.models.base import (
    as_unit_probs,
    classifier_predict_proba,
    prepare_feature_state,
    resolve_baseline,
    rows_to_matrix,
)
from src.calc.residual_ml.trainer import ResidualMLTrainer, market_from_row
from src.calc.residual_ml.vectorize import vectorize_row_values

DEFAULT_MAX_ITER = 150


def _label_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(row["label"]).strip().upper() for row in rows], dtype=object)


def _delta_matrix(
    rows: list[dict[str, Any]],
    *,
    label_smoothing: float = 0.05,
) -> np.ndarray:
    targets: list[list[float]] = []
    for row in rows:
        market = market_from_row(row)
        if market is None:
            raise ValueError(f"Missing market probabilities for match id={row.get('match_id')}")
        deltas = target_logit_deltas(
            row["label"],
            market,
            label_smoothing=label_smoothing,
        )
        targets.append([float(deltas[outcome]) for outcome in OUTCOMES])
    return np.asarray(targets, dtype=float)


class _FeatureModel:
    name: str
    family: str

    def __init__(self, *, include_market_norm: bool, random_state: int = 42) -> None:
        self.random_state = random_state
        self.include_market_norm = include_market_norm
        self.feature_names: list[str] = []
        self.global_medians: dict[str, float] = {}

    def _fit_features(self, train_rows: list[dict[str, Any]]) -> np.ndarray:
        self.feature_names, self.global_medians = prepare_feature_state(
            train_rows,
            include_market_norm=self.include_market_norm,
        )
        return rows_to_matrix(
            train_rows,
            feature_names=self.feature_names,
            global_medians=self.global_medians,
        )

    def _vectorize(self, row: dict[str, Any]) -> np.ndarray:
        return vectorize_row_values(
            row,
            feature_names=self.feature_names,
            global_medians=self.global_medians,
        )


class MarketBaselineModel:
    name = "market"
    family = "market"

    def __init__(self, *, random_state: int = 42, **_ignored: Any) -> None:
        self.random_state = random_state
        self.feature_names: list[str] = []

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        return None

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return resolve_baseline(features_or_row, baseline)


class DirectLogistic(_FeatureModel):
    name = "logistic"
    family = "direct"

    def __init__(self, *, random_state: int = 42, max_iter: int = 200, **_ignored: Any) -> None:
        super().__init__(include_market_norm=True, random_state=random_state)
        self.max_iter = max_iter
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from sklearn.linear_model import LogisticRegression

        x_train = self._fit_features(train_rows)
        self.model = LogisticRegression(
            solver="lbfgs",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.model.fit(x_train, _label_vector(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return classifier_predict_proba(self.model, self._vectorize(features_or_row))


class DirectHistGradient(_FeatureModel):
    name = "hist_gradient"
    family = "direct"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        super().__init__(include_market_norm=True, random_state=random_state)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        x_train = self._fit_features(train_rows)
        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        self.model.fit(x_train, _label_vector(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return classifier_predict_proba(self.model, self._vectorize(features_or_row))


class DirectLightGBM(_FeatureModel):
    name = "lightgbm"
    family = "direct"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        n_estimators: int | None = None,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        super().__init__(include_market_norm=True, random_state=random_state)
        self.n_estimators = n_estimators if n_estimators is not None else max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from lightgbm import LGBMClassifier

        x_train = self._fit_features(train_rows)
        self.model = LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            num_class=3,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1,
        )
        self.model.fit(x_train, _label_vector(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return classifier_predict_proba(self.model, self._vectorize(features_or_row))


class DirectCatBoost(_FeatureModel):
    name = "catboost"
    family = "direct"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        iterations: int | None = None,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        super().__init__(include_market_norm=True, random_state=random_state)
        self.iterations = iterations if iterations is not None else max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from catboost import CatBoostClassifier

        x_train = self._fit_features(train_rows)
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=self.iterations,
            depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            verbose=False,
        )
        self.model.fit(x_train, _label_vector(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return classifier_predict_proba(self.model, self._vectorize(features_or_row))


class ResidualLogistic(_FeatureModel):
    name = "logistic"
    family = "residual"

    def __init__(self, *, random_state: int = 42, **_ignored: Any) -> None:
        super().__init__(include_market_norm=False, random_state=random_state)
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor

        x_train = self._fit_features(train_rows)
        self.model = MultiOutputRegressor(LinearRegression())
        self.model.fit(x_train, _delta_matrix(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return _predict_residual(self.model, self._vectorize(features_or_row), features_or_row, baseline)


class ResidualHistGradient:
    """Family B HGB residual: wraps ResidualMLTrainer (no second HGB)."""

    name = "hist_gradient"
    family = "residual"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        self._trainer = ResidualMLTrainer(
            random_state=random_state,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

    @property
    def feature_names(self) -> list[str]:
        return self._trainer.feature_names

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        self._trainer.fit(train_rows, [])

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        market = resolve_baseline(features_or_row, baseline)
        vector = vectorize_row_values(
            features_or_row,
            feature_names=self._trainer.feature_names,
            global_medians=self._trainer.global_medians,
        )
        deltas = self._trainer.predict_deltas(vector)
        return as_unit_probs(apply_residual_deltas(market, deltas))


class ResidualLightGBM(_FeatureModel):
    name = "lightgbm"
    family = "residual"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        n_estimators: int | None = None,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        super().__init__(include_market_norm=False, random_state=random_state)
        self.n_estimators = n_estimators if n_estimators is not None else max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        x_train = self._fit_features(train_rows)
        self.model = MultiOutputRegressor(
            LGBMRegressor(
                objective="regression",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=-1,
            )
        )
        self.model.fit(x_train, _delta_matrix(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return _predict_residual(self.model, self._vectorize(features_or_row), features_or_row, baseline)


class ResidualCatBoost(_FeatureModel):
    name = "catboost"
    family = "residual"

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = DEFAULT_MAX_ITER,
        iterations: int | None = None,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **_ignored: Any,
    ) -> None:
        super().__init__(include_market_norm=False, random_state=random_state)
        self.iterations = iterations if iterations is not None else max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor

        x_train = self._fit_features(train_rows)
        self.model = MultiOutputRegressor(
            CatBoostRegressor(
                loss_function="RMSE",
                iterations=self.iterations,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_seed=self.random_state,
                verbose=False,
            )
        )
        self.model.fit(x_train, _delta_matrix(train_rows))

    def predict_proba(
        self,
        features_or_row: dict[str, Any],
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return _predict_residual(self.model, self._vectorize(features_or_row), features_or_row, baseline)


def _predict_residual(
    estimator: Any,
    vector: np.ndarray,
    row: dict[str, Any],
    baseline: Mapping[str, float] | None,
) -> dict[str, float]:
    market = resolve_baseline(row, baseline)
    predicted = estimator.predict(vector.reshape(1, -1))[0]
    deltas = {
        outcome: float(predicted[index])
        for index, outcome in enumerate(OUTCOMES)
    }
    return as_unit_probs(apply_residual_deltas(market, deltas))
