"""Post-hoc 1X2 calibration: temperature scaling and one-vs-rest isotonic.

Fit only on a chronological calibration half of validate rows. Score the later
half. Train rows never enter the calibrator.

Odd-n split: the extra row goes to the score half.
``n_calib = n // 2``, ``n_score = n - n_calib``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Sequence

from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from src.calc.probability_metrics import (
    clip_and_normalize_probs,
    mean_log_loss,
    multiclass_brier,
    top_label_ece,
)
from src.calc.residual_ml.folds import Fold
from src.utils.common import OUTCOMES

TEMPERATURE_BOUNDS = (0.05, 10.0)
VALID_LABELS = frozenset(OUTCOMES)
MARKET_PROB_KEYS = (
    "p_home_market_norm",
    "p_draw_market_norm",
    "p_away_market_norm",
)


def accept_calibration(
    mean_calibrated_log_loss: float,
    mean_raw_log_loss: float,
) -> bool:
    """Accept only if mean OOS calibrated log loss is strictly better."""
    return mean_calibrated_log_loss < mean_raw_log_loss


def split_validate_for_calibration(
    validate_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronological half-split. Extra odd row goes to score.

    ``n_calib = n // 2``, ``n_score = n - n_calib``.
    Rows must already be sorted by ``(match_date, match_id)``.
    """
    row_count = len(validate_rows)
    n_calib = row_count // 2
    return list(validate_rows[:n_calib]), list(validate_rows[n_calib:])


def _as_prob_tuple(
    item: tuple[float, float, float] | dict[str, float],
) -> tuple[float, float, float]:
    if isinstance(item, dict):
        return float(item["1"]), float(item["X"]), float(item["2"])
    return float(item[0]), float(item[1]), float(item[2])


def _tuple_to_probs(values: tuple[float, float, float]) -> dict[str, float]:
    return {"1": values[0], "X": values[1], "2": values[2]}


def _validate_fit_inputs(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> None:
    if not labels:
        raise ValueError("empty labels")
    if len(labels) != len(probability_rows):
        raise ValueError("labels and probability_rows length mismatch")
    for label in labels:
        if label not in VALID_LABELS:
            raise ValueError(f"missing or invalid label: {label!r}")


def apply_temperature(
    probs: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    """Guo-style ``softmax(log(p_k) / T)``, then clip+normalize."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    p_home, p_draw, p_away = clip_and_normalize_probs(*_as_prob_tuple(probs))
    log_over_t = [
        math.log(p_home) / temperature,
        math.log(p_draw) / temperature,
        math.log(p_away) / temperature,
    ]
    max_logit = max(log_over_t)
    exps = [math.exp(value - max_logit) for value in log_over_t]
    total = sum(exps)
    softmaxed = (exps[0] / total, exps[1] / total, exps[2] / total)
    return _tuple_to_probs(clip_and_normalize_probs(*softmaxed))


@dataclass
class FittedCalibrator:
    method: str
    temperature: float | None = None
    isotonic_regressors: dict[str, IsotonicRegression] | None = None

    def apply(self, probs: dict[str, float]) -> dict[str, float]:
        """Return clip-normalized ``{1, X, 2}`` probabilities that sum to 1."""
        if self.method == "temperature":
            if self.temperature is None:
                raise ValueError("temperature calibrator missing T")
            return apply_temperature(probs, self.temperature)
        if self.method == "isotonic":
            if not self.isotonic_regressors:
                raise ValueError("isotonic calibrator missing maps")
            p_home, p_draw, p_away = clip_and_normalize_probs(*_as_prob_tuple(probs))
            mapped = (
                float(self.isotonic_regressors["1"].predict([p_home])[0]),
                float(self.isotonic_regressors["X"].predict([p_draw])[0]),
                float(self.isotonic_regressors["2"].predict([p_away])[0]),
            )
            return _tuple_to_probs(clip_and_normalize_probs(*mapped))
        raise ValueError(f"unknown method: {self.method}")


def fit_temperature(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float] | dict[str, float]],
) -> FittedCalibrator:
    """Fit T on the calibration slice only by minimizing multiclass log loss."""
    tuples = [_as_prob_tuple(item) for item in probability_rows]
    _validate_fit_inputs(labels, tuples)

    def objective(temperature: float) -> float:
        calibrated = [
            _as_prob_tuple(apply_temperature(_tuple_to_probs(probs), temperature))
            for probs in tuples
        ]
        return mean_log_loss(labels, calibrated)

    result = minimize_scalar(
        objective,
        bounds=TEMPERATURE_BOUNDS,
        method="bounded",
    )
    return FittedCalibrator(method="temperature", temperature=float(result.x))


def fit_isotonic(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float] | dict[str, float]],
) -> FittedCalibrator:
    """Fit one-vs-rest isotonic maps on the calibration slice only."""
    tuples = [_as_prob_tuple(item) for item in probability_rows]
    _validate_fit_inputs(labels, tuples)
    regressors: dict[str, IsotonicRegression] = {}
    for index, outcome in enumerate(OUTCOMES):
        predicted = [row[index] for row in tuples]
        targets = [1.0 if label == outcome else 0.0 for label in labels]
        regressor = IsotonicRegression(increasing=True, out_of_bounds="clip")
        regressor.fit(predicted, targets)
        regressors[outcome] = regressor
    return FittedCalibrator(method="isotonic", isotonic_regressors=regressors)


def fit_calibrator(
    method: str,
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float] | dict[str, float]],
) -> FittedCalibrator:
    if method == "temperature":
        return fit_temperature(labels, probability_rows)
    if method == "isotonic":
        return fit_isotonic(labels, probability_rows)
    raise ValueError(f"unknown method: {method}")


def parse_market_label_and_probs(
    row: dict[str, Any],
) -> tuple[str, tuple[float, float, float]] | None:
    if any(row.get(key) in (None, "") for key in MARKET_PROB_KEYS):
        return None
    label = str(row.get("label", "")).strip().upper()
    if label not in VALID_LABELS:
        return None
    try:
        probabilities = (
            float(row["p_home_market_norm"]),
            float(row["p_draw_market_norm"]),
            float(row["p_away_market_norm"]),
        )
    except (TypeError, ValueError):
        return None
    return label, probabilities


def labels_and_probs_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    require_all: bool = False,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Extract market 1X2 labels and probabilities.

    ``require_all=True`` raises ``ValueError`` on empty input or any missing
    label / probability. CLI scoring skips unscorable rows instead.
    """
    if require_all and not rows:
        raise ValueError("empty labels")
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        parsed = parse_market_label_and_probs(row)
        if parsed is None:
            if require_all:
                raise ValueError("missing or invalid label")
            continue
        label, probabilities = parsed
        labels.append(label)
        probability_rows.append(probabilities)
    if require_all and not labels:
        raise ValueError("empty labels")
    return labels, probability_rows


def fit_calibrator_from_rows(
    method: str,
    rows: Sequence[dict[str, Any]],
) -> FittedCalibrator:
    """Fit from calibration rows only. Missing / empty labels → ValueError."""
    labels, probability_rows = labels_and_probs_from_rows(rows, require_all=True)
    return fit_calibrator(method, labels, probability_rows)


def _score_probability_rows(
    labels: Sequence[str],
    probability_rows: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    return (
        mean_log_loss(labels, probability_rows),
        multiclass_brier(labels, probability_rows),
        top_label_ece(labels, probability_rows),
    )


@dataclass
class CalibrationFoldScore:
    fold_name: str
    n_calib: int
    n_score: int
    raw_ll: float
    cal_ll: float
    delta_ll: float
    raw_ece: float
    cal_ece: float
    raw_brier: float
    cal_brier: float
    temperature: float | None = None


@dataclass
class CalibrationEvalResult:
    method: str
    per_fold: list[CalibrationFoldScore]
    mean_n_calib: float
    mean_n_score: float
    mean_raw_ll: float
    mean_cal_ll: float
    mean_delta_ll: float
    mean_raw_ece: float
    mean_cal_ece: float
    mean_raw_brier: float
    mean_cal_brier: float
    accepted: bool


def _unweighted_mean_fold_scores(
    scores: list[CalibrationFoldScore],
) -> dict[str, float]:
    fold_count = len(scores)
    if fold_count == 0:
        raise ValueError("no folds to average")
    averages: dict[str, float] = {}
    for field in fields(CalibrationFoldScore):
        if field.name in {"fold_name", "temperature"}:
            continue
        averages[field.name] = (
            sum(getattr(item, field.name) for item in scores) / fold_count
        )
    return averages


def evaluate_calibration_on_folds(
    folds: Sequence[Fold],
    method: str,
) -> CalibrationEvalResult:
    """Fit on each fold's earlier validate half; score the later half only."""
    if method not in {"temperature", "isotonic"}:
        raise ValueError(f"unknown method: {method}")
    per_fold: list[CalibrationFoldScore] = []
    for fold in folds:
        calib_rows, score_rows = split_validate_for_calibration(fold.validate_rows)
        if not calib_rows or not score_rows:
            continue
        calib_labels, calib_probs = labels_and_probs_from_rows(calib_rows)
        score_labels, score_probs = labels_and_probs_from_rows(score_rows)
        if not calib_labels or not score_labels:
            continue
        calibrator = fit_calibrator(method, calib_labels, calib_probs)
        calibrated_score = [
            _as_prob_tuple(calibrator.apply(_tuple_to_probs(probs)))
            for probs in score_probs
        ]
        raw_ll, raw_brier, raw_ece = _score_probability_rows(
            score_labels, score_probs
        )
        cal_ll, cal_brier, cal_ece = _score_probability_rows(
            score_labels, calibrated_score
        )
        per_fold.append(
            CalibrationFoldScore(
                fold_name=fold.name,
                n_calib=len(calib_labels),
                n_score=len(score_labels),
                raw_ll=raw_ll,
                cal_ll=cal_ll,
                delta_ll=cal_ll - raw_ll,
                raw_ece=raw_ece,
                cal_ece=cal_ece,
                raw_brier=raw_brier,
                cal_brier=cal_brier,
                temperature=calibrator.temperature,
            )
        )
    averages = _unweighted_mean_fold_scores(per_fold)
    return CalibrationEvalResult(
        method=method,
        per_fold=per_fold,
        mean_n_calib=averages["n_calib"],
        mean_n_score=averages["n_score"],
        mean_raw_ll=averages["raw_ll"],
        mean_cal_ll=averages["cal_ll"],
        mean_delta_ll=averages["delta_ll"],
        mean_raw_ece=averages["raw_ece"],
        mean_cal_ece=averages["cal_ece"],
        mean_raw_brier=averages["raw_brier"],
        mean_cal_brier=averages["cal_brier"],
        accepted=accept_calibration(averages["cal_ll"], averages["raw_ll"]),
    )
