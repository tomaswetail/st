"""Scoring helpers for residual ML backtest evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from calc.probability_metrics import multiclass_log_loss
from calc.residual_ml.baseline import apply_market_only_baseline, shrink_toward_market
from config.eval_protocol import DRAW_WINDOW_MAX, DRAW_WINDOW_MIN

MARKET_KEYS = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
DEFAULT_SHRINK_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LABEL_TO_INDEX = {"1": 0, "X": 1, "2": 2}
INDEX_TO_LABEL = ("1", "X", "2")
BASELINE_KEY_GROUPS = {
    "market": MARKET_KEYS,
    "dc": ("p_home_dc_norm", "p_draw_dc_norm", "p_away_dc_norm"),
    "blend": ("p_home_blend", "p_draw_blend", "p_away_blend"),
}


def _draw_number(row: dict[str, Any]) -> int | None:
    value = row.get("draw_number")
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def draw_quarter_ranges(
    *,
    draw_min: int = DRAW_WINDOW_MIN,
    draw_max: int = DRAW_WINDOW_MAX,
) -> list[tuple[str, int, int]]:
    """Return inclusive draw_number ranges for four equal draw quarters."""
    span = draw_max - draw_min + 1
    quarter_size = span // 4
    ranges: list[tuple[str, int, int]] = []
    for quarter_index in range(4):
        start = draw_min + quarter_index * quarter_size
        end = (
            draw_max
            if quarter_index == 3
            else draw_min + (quarter_index + 1) * quarter_size - 1
        )
        ranges.append((f"Q{quarter_index + 1}", start, end))
    return ranges


def slice_rows_by_year(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by calendar year from ``match_date`` (YYYY-MM-DD)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        match_date = str(row.get("match_date", ""))
        year = match_date[:4] if len(match_date) >= 4 else "unknown"
        grouped.setdefault(year, []).append(row)
    return dict(sorted(grouped.items()))


def slice_rows_by_draw_quarter(
    rows: list[dict[str, Any]],
    *,
    draw_min: int = DRAW_WINDOW_MIN,
    draw_max: int = DRAW_WINDOW_MAX,
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by draw quarter within the configured draw window."""
    grouped = {label: [] for label, _, _ in draw_quarter_ranges(draw_min=draw_min, draw_max=draw_max)}
    for row in rows:
        draw_number = _draw_number(row)
        if draw_number is None:
            continue
        for label, quarter_min, quarter_max in draw_quarter_ranges(
            draw_min=draw_min,
            draw_max=draw_max,
        ):
            if quarter_min <= draw_number <= quarter_max:
                grouped[label].append(row)
                break
    return grouped


def slice_rows_by_league(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]] | None:
    """Group rows by ``league_external_id`` when that column is present."""
    if not rows or "league_external_id" not in rows[0]:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        league_id = row.get("league_external_id")
        if league_id in (None, ""):
            continue
        grouped.setdefault(str(league_id), []).append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def top_pick_accuracy(
    rows: list[dict[str, Any]],
    prob_keys: tuple[str, str, str],
) -> tuple[float | None, int]:
    """Fraction of rows where argmax(probs) equals the observed label."""
    correct = 0
    scored = 0
    for row in rows:
        vector = _prob_vector(row, prob_keys)
        if vector is None:
            continue
        predicted_label = INDEX_TO_LABEL[max(range(3), key=lambda index: vector[index])]
        if predicted_label == row["label"]:
            correct += 1
        scored += 1
    if scored == 0:
        return None, 0
    return correct / scored, scored


def _prob_vector(row: dict[str, Any], keys: tuple[str, str, str]) -> list[float] | None:
    if any(row.get(key) in (None, "") for key in keys):
        return None
    return [float(row[key]) for key in keys]


def _market_probs(row: dict[str, Any]) -> dict[str, float] | None:
    vector = _prob_vector(row, MARKET_KEYS)
    if vector is None:
        return None
    return {"1": vector[0], "X": vector[1], "2": vector[2]}


def score_baseline_log_losses(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[float | None, int]]:
    """Score market, DC, and blend baselines on rows."""
    y_true = [LABEL_TO_INDEX[row["label"]] for row in rows]
    results: dict[str, tuple[float | None, int]] = {}
    for name, keys in BASELINE_KEY_GROUPS.items():
        vectors = [_prob_vector(row, keys) for row in rows]
        valid_indices = [index for index, vec in enumerate(vectors) if vec is not None]
        if not valid_indices:
            results[name] = (None, 0)
            continue
        loss = multiclass_log_loss(
            [y_true[index] for index in valid_indices],
            [vectors[index] for index in valid_indices],
        )
        results[name] = (loss, len(valid_indices))
    return results


@dataclass
class BacktestScoringResult:
    baselines: dict[str, tuple[float | None, int]]
    blend_after_market_only: tuple[float | None, int] | None
    market_on_ml_rows: float | None
    ml_row_count: int
    shrink_results: list[tuple[float, float]]
    best_alpha: float | None
    best_loss: float | None


def run_backtest_scoring(
    rows: list[dict[str, Any]],
    trainer: Any,
    *,
    use_market_only_baseline: bool = False,
    shrink_alphas: Sequence[float] | None = None,
    final_shrink: float | None = None,
) -> BacktestScoringResult:
    """Score baselines and ML predictions with optional market shrink."""
    if use_market_only_baseline:
        apply_market_only_baseline(rows)

    baselines = score_baseline_log_losses(rows)
    blend_after_market_only = None
    if use_market_only_baseline:
        blend_loss, blend_count = baselines["blend"]
        blend_after_market_only = (blend_loss, blend_count)

    raw_ml: list[dict[str, float]] = []
    markets: list[dict[str, float]] = []
    ml_labels: list[int] = []
    for row in rows:
        probs = trainer.predict_match_proba(row)
        market = _market_probs(row)
        if probs is None or market is None:
            continue
        raw_ml.append(probs)
        markets.append(market)
        ml_labels.append(LABEL_TO_INDEX[row["label"]])

    if not raw_ml:
        return BacktestScoringResult(
            baselines=baselines,
            blend_after_market_only=blend_after_market_only,
            market_on_ml_rows=None,
            ml_row_count=0,
            shrink_results=[],
            best_alpha=None,
            best_loss=None,
        )

    market_on_ml_rows = multiclass_log_loss(
        ml_labels,
        [[market["1"], market["X"], market["2"]] for market in markets],
    )

    if final_shrink is None:
        alphas = list(shrink_alphas or DEFAULT_SHRINK_ALPHAS)
    else:
        alphas = [0.0]
        if abs(final_shrink) > 1e-12:
            alphas.append(float(final_shrink))

    shrink_results: list[tuple[float, float]] = []
    best_alpha = None
    best_loss = None
    for alpha in alphas:
        vectors = []
        for ml_probs, market in zip(raw_ml, markets):
            if alpha <= 0:
                shrunk = ml_probs
            else:
                shrunk = shrink_toward_market(ml_probs, market, alpha=alpha)
            vectors.append([shrunk["1"], shrunk["X"], shrunk["2"]])
        loss = multiclass_log_loss(ml_labels, vectors)
        shrink_results.append((alpha, loss))
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    return BacktestScoringResult(
        baselines=baselines,
        blend_after_market_only=blend_after_market_only,
        market_on_ml_rows=market_on_ml_rows,
        ml_row_count=len(ml_labels),
        shrink_results=shrink_results,
        best_alpha=best_alpha,
        best_loss=best_loss,
    )


@dataclass
class SliceMetrics:
    name: str
    row_count: int
    baselines: dict[str, tuple[float | None, int]] = field(default_factory=dict)
    market_on_ml_rows: float | None = None
    ml_log_loss: float | None = None
    best_shrink_alpha: float | None = None
    best_shrink_log_loss: float | None = None
    top_pick: dict[str, tuple[float | None, int]] = field(default_factory=dict)


def summarize_slice_metrics(
    slice_name: str,
    rows: list[dict[str, Any]],
    scoring: BacktestScoringResult | None = None,
) -> SliceMetrics:
    """Build pooled metrics for one row slice."""
    baselines = score_baseline_log_losses(rows)
    top_pick = {
        name: top_pick_accuracy(rows, keys)
        for name, keys in BASELINE_KEY_GROUPS.items()
    }
    metrics = SliceMetrics(
        name=slice_name,
        row_count=len(rows),
        baselines=baselines,
        top_pick=top_pick,
    )
    if scoring is None:
        return metrics

    metrics.market_on_ml_rows = scoring.market_on_ml_rows
    metrics.ml_log_loss = next(
        (loss for alpha, loss in scoring.shrink_results if alpha <= 0),
        None,
    )
    metrics.best_shrink_alpha = scoring.best_alpha
    metrics.best_shrink_log_loss = scoring.best_loss
    return metrics


def build_multi_slice_report(
    rows: list[dict[str, Any]],
    scoring: BacktestScoringResult | None = None,
) -> dict[str, SliceMetrics]:
    """Pooled metrics plus per-year and per-draw-quarter slices."""
    report = {
        "pooled": summarize_slice_metrics("pooled", rows, scoring),
    }
    for year, year_rows in slice_rows_by_year(rows).items():
        report[f"year:{year}"] = summarize_slice_metrics(
            f"year:{year}",
            year_rows,
        )
    for quarter, quarter_rows in slice_rows_by_draw_quarter(rows).items():
        report[f"draw_quarter:{quarter}"] = summarize_slice_metrics(
            f"draw_quarter:{quarter}",
            quarter_rows,
        )
    league_slices = slice_rows_by_league(rows)
    if league_slices is not None:
        for league_id, league_rows in league_slices.items():
            report[f"league:{league_id}"] = summarize_slice_metrics(
                f"league:{league_id}",
                league_rows,
            )
    return report
