"""Scoring helpers for residual ML backtest evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from calc.probability_metrics import multiclass_log_loss
from calc.residual_ml.baseline import apply_market_only_baseline, shrink_toward_market

MARKET_KEYS = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")
DEFAULT_SHRINK_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LABEL_TO_INDEX = {"1": 0, "X": 1, "2": 2}
BASELINE_KEY_GROUPS = {
    "market": MARKET_KEYS,
    "dc": ("p_home_dc_norm", "p_draw_dc_norm", "p_away_dc_norm"),
    "blend": ("p_home_blend", "p_draw_blend", "p_away_blend"),
}


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
