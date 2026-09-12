"""Calendar-year chronological folds and market-only scoring.

Fold key is the **calendar year** of ``match_date`` (YYYY-MM-DD or
date/datetime). English football seasons span two calendar years (August–May),
so this is not a Jul–Jun ``league_season``. A later phase can add a
season-start-year helper; this module does not invent one.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Any, Callable

from src.calc.probability_metrics import (
    mean_accuracy,
    mean_log_loss,
    mean_rps,
    multiclass_brier,
    per_class_ece,
    per_class_log_loss,
    top_label_ece,
)

MARKET_PROB_KEYS = (
    "p_home_market_norm",
    "p_draw_market_norm",
    "p_away_market_norm",
)
VALID_LABELS = frozenset({"1", "X", "2"})


@dataclass
class Fold:
    name: str
    train_years: list[int]
    validate_year: int
    train_rows: list[dict]
    validate_rows: list[dict]


@dataclass
class Metrics:
    n: int
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    ece: float
    log_loss_home: float
    log_loss_draw: float
    log_loss_away: float
    ece_home: float
    ece_draw: float
    ece_away: float


@dataclass
class FoldScores:
    per_fold: list[tuple[Fold, Metrics]]
    mean: Metrics


def calendar_year_of_match_date(match_date: Any) -> int:
    """Calendar year of ``match_date``. Missing or unparseable → ``ValueError``."""
    if match_date is None or match_date == "":
        raise ValueError("missing match_date")
    if isinstance(match_date, (date, datetime)):
        return int(match_date.year)
    text = str(match_date).strip()
    if not text:
        raise ValueError("missing match_date")
    date_part = text[:10]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d").year
    except ValueError as exc:
        raise ValueError(f"unparseable match_date: {match_date!r}") from exc


def _date_sort_key(row: dict[str, Any]) -> str:
    match_date = row.get("match_date")
    if isinstance(match_date, (date, datetime)):
        return match_date.isoformat()[:10]
    if match_date is None:
        return ""
    return str(match_date)[:10]


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (_date_sort_key(row), row.get("match_id", 0)),
    )


def _rows_by_year(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        year = calendar_year_of_match_date(row.get("match_date"))
        grouped.setdefault(year, []).append(row)
    return grouped


def expanding_year_folds(rows: list[dict[str, Any]]) -> list[Fold]:
    """Expanding folds: train ``year < Y``, validate ``year == Y``.

    The earliest year is never a validate fold. Years with no prior train
    rows are skipped. Validate-empty folds are omitted.
    """
    if not rows:
        raise ValueError("rows is empty")
    by_year = _rows_by_year(rows)
    years = sorted(by_year)
    folds: list[Fold] = []
    for validate_year in years:
        train_years = [year for year in years if year < validate_year]
        if not train_years:
            continue
        validate_rows = _sorted_rows(by_year[validate_year])
        if not validate_rows:
            continue
        train_rows = _sorted_rows(
            [row for year in train_years for row in by_year[year]]
        )
        if not train_rows:
            continue
        folds.append(
            Fold(
                name=f"expanding-{validate_year}",
                train_years=train_years,
                validate_year=validate_year,
                train_rows=train_rows,
                validate_rows=validate_rows,
            )
        )
    return folds


def rolling_year_folds(
    rows: list[dict[str, Any]],
    train_window_years: int = 2,
) -> list[Fold]:
    """Rolling folds: train ``[Y - window, Y)``, validate ``Y``.

    Default window is 2 calendar years. Skip if train or validate would be
    empty.
    """
    if train_window_years < 1:
        raise ValueError("train_window_years must be >= 1")
    if not rows:
        raise ValueError("rows is empty")
    by_year = _rows_by_year(rows)
    years = sorted(by_year)
    folds: list[Fold] = []
    for validate_year in years:
        window_start = validate_year - train_window_years
        train_years = [
            year for year in years if window_start <= year < validate_year
        ]
        if not train_years:
            continue
        train_rows = _sorted_rows(
            [row for year in train_years for row in by_year[year]]
        )
        validate_rows = _sorted_rows(by_year[validate_year])
        if not train_rows or not validate_rows:
            continue
        folds.append(
            Fold(
                name=f"rolling-{validate_year}",
                train_years=train_years,
                validate_year=validate_year,
                train_rows=train_rows,
                validate_rows=validate_rows,
            )
        )
    return folds


def _market_label_and_probs(
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


def score_market_only(rows: list[dict[str, Any]]) -> Metrics:
    """Score market-normalized 1X2 columns only (no features, no model)."""
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        parsed = _market_label_and_probs(row)
        if parsed is None:
            continue
        label, probabilities = parsed
        labels.append(label)
        probability_rows.append(probabilities)
    if not labels:
        raise ValueError("no scorable market-only rows")
    log_loss_home, log_loss_draw, log_loss_away = per_class_log_loss(
        labels, probability_rows
    )
    if (
        log_loss_home is None
        or log_loss_draw is None
        or log_loss_away is None
    ):
        raise ValueError("no scorable market-only rows")
    return Metrics(
        n=len(labels),
        log_loss=mean_log_loss(labels, probability_rows),
        brier=multiclass_brier(labels, probability_rows),
        rps=mean_rps(labels, probability_rows),
        accuracy=mean_accuracy(labels, probability_rows),
        ece=top_label_ece(labels, probability_rows),
        log_loss_home=log_loss_home,
        log_loss_draw=log_loss_draw,
        log_loss_away=log_loss_away,
        ece_home=per_class_ece(labels, probability_rows, "1"),
        ece_draw=per_class_ece(labels, probability_rows, "X"),
        ece_away=per_class_ece(labels, probability_rows, "2"),
    )


def _unweighted_mean_metrics(metrics_list: list[Metrics]) -> Metrics:
    fold_count = len(metrics_list)
    if fold_count == 0:
        raise ValueError("no folds to average")
    averages: dict[str, float] = {}
    for field in fields(Metrics):
        averages[field.name] = (
            sum(getattr(item, field.name) for item in metrics_list) / fold_count
        )
    return Metrics(**averages)


def score_folds(
    folds: list[Fold],
    scorer: Callable[[list[dict[str, Any]]], Metrics],
) -> FoldScores:
    """Run ``scorer`` on each fold's validate rows. Mean is unweighted."""
    if not folds:
        raise ValueError("folds is empty")
    per_fold: list[tuple[Fold, Metrics]] = []
    for fold in folds:
        per_fold.append((fold, scorer(fold.validate_rows)))
    return FoldScores(
        per_fold=per_fold,
        mean=_unweighted_mean_metrics([metrics for _fold, metrics in per_fold]),
    )
