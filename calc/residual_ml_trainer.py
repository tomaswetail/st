"""Train and persist the residual 1X2 ML model."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputRegressor

from calc.residual_ml_baseline import (
    OUTCOMES,
    apply_residual_deltas,
    target_logit_deltas,
)
from objects.schema.data_classes.residual_ml_features import ResidualMLFeatures

MODEL_TYPE = "residual_logit_v1"

_DATASET_ONLY_FIELDS = frozenset(
    {
        "label",
        "match_date",
        "match_id",
        "draw_number",
        "feature_cutoff_date",
        "p_home_blend",
        "p_draw_blend",
        "p_away_blend",
        "p_home_market_norm",
        "p_draw_market_norm",
        "p_away_market_norm",
        "p_home_dc_norm",
        "p_draw_dc_norm",
        "p_away_dc_norm",
    }
)


@dataclass
class ResidualMLTrainingResult:
    version: str
    train_rows: int
    validation_rows: int
    train_log_loss: float
    validation_log_loss: float
    market_validation_log_loss: float | None
    blend_validation_log_loss: float | None
    model_path: Path
    feature_schema_path: Path
    baseline_weights_path: Path


class ResidualMLTrainer:
    """Train a multi-output regressor predicting logit deltas vs blended baseline."""

    def __init__(
        self,
        *,
        market_weight: float = 0.7,
        dc_weight: float = 0.3,
        random_state: int = 42,
        label_smoothing: float = 0.05,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        max_iter: int = 300,
    ) -> None:
        self.market_weight = market_weight
        self.dc_weight = dc_weight
        self.random_state = random_state
        self.label_smoothing = label_smoothing
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.feature_names: list[str] = []
        self.global_medians: dict[str, float] = {}
        self.version = "v1"
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=random_state,
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
            )
        )

    @classmethod
    def feature_names_from_rows(cls, rows: list[dict[str, Any]]) -> list[str]:
        if not rows:
            return []
        return [
            key for key in rows[0].keys() if key not in _DATASET_ONLY_FIELDS
        ]

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
    ) -> ResidualMLTrainingResult:
        if not train_rows:
            raise ValueError("Training dataset is empty")

        self.feature_names = self.feature_names_from_rows(train_rows)
        self.global_medians = self._compute_medians(train_rows)

        x_train = np.vstack([self._vectorize_row(row) for row in train_rows])
        y_train = np.vstack(
            [self._target_deltas(row) for row in train_rows]
        )
        x_valid = np.vstack([self._vectorize_row(row) for row in validation_rows])
        y_valid_labels = [row["label"] for row in validation_rows]

        self.model.fit(x_train, y_train)

        train_loss = self._log_loss_from_rows(train_rows)
        valid_loss = self._log_loss_from_rows(validation_rows, labels=y_valid_labels)
        market_loss = self._baseline_log_loss(validation_rows, prefix="p_home_market_norm")
        blend_loss = self._baseline_log_loss(validation_rows, prefix="p_home_blend")

        return ResidualMLTrainingResult(
            version="",
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
            train_log_loss=train_loss,
            validation_log_loss=valid_loss,
            market_validation_log_loss=market_loss,
            blend_validation_log_loss=blend_loss,
            model_path=Path(),
            feature_schema_path=Path(),
            baseline_weights_path=Path(),
        )

    def save(self, directory: Path, *, version: str = "v1") -> ResidualMLTrainingResult:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.pkl"
        feature_schema_path = directory / "feature_schema.json"
        baseline_weights_path = directory / "baseline_weights.json"

        artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
            "global_medians": self.global_medians,
            "market_weight": self.market_weight,
            "dc_weight": self.dc_weight,
            "label_smoothing": self.label_smoothing,
            "model_type": MODEL_TYPE,
            "version": version,
        }
        with model_path.open("wb") as handle:
            pickle.dump(artifact, handle)

        feature_schema_path.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "global_medians": self.global_medians,
                    "model_type": MODEL_TYPE,
                    "version": version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        baseline_weights_path.write_text(
            json.dumps(
                {
                    "market_weight": self.market_weight,
                    "dc_weight": self.dc_weight,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return ResidualMLTrainingResult(
            version=version,
            train_rows=0,
            validation_rows=0,
            train_log_loss=0.0,
            validation_log_loss=0.0,
            market_validation_log_loss=None,
            blend_validation_log_loss=None,
            model_path=model_path,
            feature_schema_path=feature_schema_path,
            baseline_weights_path=baseline_weights_path,
        )

    @classmethod
    def load(cls, model_path: Path) -> ResidualMLTrainer:
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        model_type = artifact.get("model_type")
        if model_type != MODEL_TYPE:
            raise ValueError(
                f"Unsupported model type {model_type!r}; expected {MODEL_TYPE!r}. Retrain required."
            )
        trainer = cls(
            market_weight=float(artifact.get("market_weight", 0.7)),
            dc_weight=float(artifact.get("dc_weight", 0.3)),
            label_smoothing=float(artifact.get("label_smoothing", 0.05)),
        )
        trainer.model = artifact["model"]
        trainer.feature_names = list(artifact["feature_names"])
        trainer.global_medians = dict(artifact.get("global_medians", {}))
        trainer.market_weight = float(artifact.get("market_weight", trainer.market_weight))
        trainer.dc_weight = float(artifact.get("dc_weight", trainer.dc_weight))
        trainer.version = str(artifact.get("version", "v1"))
        return trainer

    def predict_deltas(self, vector: np.ndarray) -> dict[str, float]:
        predicted = self.model.predict(vector.reshape(1, -1))[0]
        return {
            outcome: float(predicted[index])
            for index, outcome in enumerate(OUTCOMES)
        }

    def predict_match_proba(self, row: dict[str, Any]) -> dict[str, float] | None:
        blend = blend_from_row(row)
        if blend is None:
            return None
        vector = self._vectorize_row(row)
        deltas = self.predict_deltas(vector)
        return apply_residual_deltas(blend, deltas)

    def _vectorize_row(self, row: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        for name in self.feature_names:
            raw = row.get(name)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                values.append(float(self.global_medians.get(name, 0.0)))
            else:
                values.append(float(raw))
        return np.asarray(values, dtype=float)

    def _target_deltas(self, row: dict[str, Any]) -> np.ndarray:
        blend = blend_from_row(row)
        if blend is None:
            raise ValueError(f"Missing blend probabilities for match id={row.get('match_id')}")
        deltas = target_logit_deltas(
            row["label"],
            blend,
            label_smoothing=self.label_smoothing,
        )
        return np.asarray([deltas[outcome] for outcome in OUTCOMES], dtype=float)

    def _log_loss_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        labels: list[str] | None = None,
    ) -> float:
        if not rows:
            return 0.0
        label_to_index = {"1": 0, "X": 1, "2": 2}
        y_true: list[int] = []
        y_prob: list[list[float]] = []
        for index, row in enumerate(rows):
            probs = self.predict_match_proba(row)
            if probs is None:
                continue
            label = labels[index] if labels is not None else row["label"]
            y_true.append(label_to_index[label])
            y_prob.append([probs[outcome] for outcome in OUTCOMES])
        if not y_true:
            return 0.0
        return float(log_loss(y_true, y_prob))

    @staticmethod
    def _compute_medians(rows: list[dict[str, Any]]) -> dict[str, float]:
        medians: dict[str, float] = {}
        if not rows:
            return medians
        for key in rows[0].keys():
            if key in _DATASET_ONLY_FIELDS:
                continue
            values = [
                float(row[key])
                for row in rows
                if row.get(key) is not None and not _is_nan(row.get(key))
            ]
            if values:
                medians[key] = float(np.median(values))
        return medians

    @staticmethod
    def _baseline_log_loss(
        rows: list[dict[str, Any]], *, prefix: str
    ) -> float | None:
        if not rows:
            return None
        if prefix == "p_home_blend":
            prob_keys = ("p_home_blend", "p_draw_blend", "p_away_blend")
        else:
            prob_keys = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
        y_true = []
        y_prob = []
        label_to_index = {"1": 0, "X": 1, "2": 2}
        for row in rows:
            if any(row.get(key) is None for key in prob_keys):
                continue
            y_true.append(label_to_index[row["label"]])
            y_prob.append([float(row[key]) for key in prob_keys])
        if not y_true:
            return None
        return float(log_loss(y_true, y_prob))


def blend_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    blend = {
        "1": row.get("p_home_blend"),
        "X": row.get("p_draw_blend"),
        "2": row.get("p_away_blend"),
    }
    if any(blend[outcome] is None for outcome in OUTCOMES):
        return None
    return {outcome: float(blend[outcome]) for outcome in OUTCOMES}  # type: ignore[arg-type]


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def vectorize_features(
    features: ResidualMLFeatures,
    *,
    feature_names: list[str],
    global_medians: dict[str, float],
) -> np.ndarray:
    from datetime import date

    row = features.to_dict()
    for key, value in row.items():
        if isinstance(value, date):
            row[key] = value.isoformat()
    trainer = ResidualMLTrainer()
    trainer.feature_names = feature_names
    trainer.global_medians = global_medians
    return trainer._vectorize_row(row)
