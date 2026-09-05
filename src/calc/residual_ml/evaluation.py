"""Scoring helpers for residual ML backtest evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml.baseline import apply_market_only_baseline, shrink_toward_market
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


def _parse_has_availability(row: dict[str, Any]) -> int | None:
    raw = row.get("has_availability")
    if raw in (None, ""):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _parse_float(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def slice_rows_by_availability(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]] | None:
    """Group rows by has_availability (0/1) when column is present."""
    if not rows or "has_availability" not in rows[0]:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {
        "availability:0": [],
        "availability:1": [],
    }
    for row in rows:
        flag = _parse_has_availability(row)
        if flag is None:
            continue
        key = f"availability:{flag}"
        if key in grouped:
            grouped[key].append(row)
    return grouped


def slice_rows_by_injury_heavy(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 0.0,
) -> dict[str, list[dict[str, Any]]] | None:
    """Rows with has_availability=1 and |missing_value_difference| > threshold."""
    if not rows or "has_availability" not in rows[0]:
        return None
    heavy: list[dict[str, Any]] = []
    light: list[dict[str, Any]] = []
    for row in rows:
        if _parse_has_availability(row) != 1:
            continue
        diff = _parse_float(row, "missing_value_difference")
        if diff is None:
            light.append(row)
            continue
        if abs(diff) > threshold:
            heavy.append(row)
        else:
            light.append(row)
    return {
        f"injury_heavy:>{threshold}": heavy,
        f"injury_heavy:<={threshold}": light,
    }


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
    production_shrink_alpha: float | None = None
    production_shrink_log_loss: float | None = None
    top_pick: dict[str, tuple[float | None, int]] = field(default_factory=dict)


def _production_shrink_loss(
    scoring: BacktestScoringResult,
    production_shrink_alpha: float | None,
) -> float | None:
    if production_shrink_alpha is None:
        return None
    for alpha, loss in scoring.shrink_results:
        if abs(alpha - production_shrink_alpha) < 1e-12:
            return loss
    return None


def summarize_slice_metrics(
    slice_name: str,
    rows: list[dict[str, Any]],
    scoring: BacktestScoringResult | None = None,
    *,
    production_shrink_alpha: float | None = None,
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
    if production_shrink_alpha is not None:
        metrics.production_shrink_alpha = production_shrink_alpha
        metrics.production_shrink_log_loss = _production_shrink_loss(
            scoring, production_shrink_alpha
        )
    return metrics


def _score_slice_with_trainer(
    slice_rows: list[dict[str, Any]],
    trainer: Any,
) -> BacktestScoringResult:
    return run_backtest_scoring(
        slice_rows,
        trainer,
        use_market_only_baseline=False,
    )


def build_multi_slice_report(
    rows: list[dict[str, Any]],
    scoring: BacktestScoringResult | None = None,
    trainer: Any | None = None,
    *,
    production_shrink_alpha: float | None = None,
    use_market_only_baseline: bool = False,
) -> dict[str, SliceMetrics]:
    """Pooled metrics plus year/quarter/league/availability/injury slices."""
    if use_market_only_baseline:
        apply_market_only_baseline(rows)

    def metrics_for(
        slice_name: str,
        slice_rows: list[dict[str, Any]],
        *,
        fallback_scoring: BacktestScoringResult | None = None,
    ) -> SliceMetrics:
        slice_scoring = fallback_scoring
        if trainer is not None and slice_rows:
            slice_scoring = _score_slice_with_trainer(slice_rows, trainer)
        return summarize_slice_metrics(
            slice_name,
            slice_rows,
            slice_scoring,
            production_shrink_alpha=production_shrink_alpha,
        )

    report = {
        "pooled": metrics_for("pooled", rows, fallback_scoring=scoring),
    }
    for year, year_rows in slice_rows_by_year(rows).items():
        report[f"year:{year}"] = metrics_for(f"year:{year}", year_rows)
    for quarter, quarter_rows in slice_rows_by_draw_quarter(rows).items():
        report[f"draw_quarter:{quarter}"] = metrics_for(
            f"draw_quarter:{quarter}",
            quarter_rows,
        )
    league_slices = slice_rows_by_league(rows)
    if league_slices is not None:
        for league_id, league_rows in league_slices.items():
            report[f"league:{league_id}"] = metrics_for(
                f"league:{league_id}",
                league_rows,
            )
    availability_slices = slice_rows_by_availability(rows)
    if availability_slices is not None:
        for slice_name, slice_rows in availability_slices.items():
            report[slice_name] = metrics_for(slice_name, slice_rows)
    injury_slices = slice_rows_by_injury_heavy(rows)
    if injury_slices is not None:
        for slice_name, slice_rows in injury_slices.items():
            report[slice_name] = metrics_for(slice_name, slice_rows)
    return report


def _clip_unit(probability: float, *, epsilon: float = 1e-15) -> float:
    return min(1.0 - epsilon, max(epsilon, probability))


def _binary_log_loss(y_true: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if not y_true:
        return None
    total = 0.0
    for label, probability in zip(y_true, probabilities):
        clipped = _clip_unit(float(probability))
        if label:
            total += -math.log(clipped)
        else:
            total += -math.log(1.0 - clipped)
    return total / len(y_true)


def _binary_brier(y_true: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if not y_true:
        return None
    total = 0.0
    for label, probability in zip(y_true, probabilities):
        total += (float(probability) - float(label)) ** 2
    return total / len(y_true)


def score_outcome_metrics(
    rows: list[dict[str, Any]],
    prob_keys: tuple[str, str, str],
) -> dict[str, Any]:
    """Binary one-vs-rest metrics for home/draw/away predicted probabilities.

    - draw_log_loss / draw_brier: y=(label==X), p=p_draw
    - home_log_loss: y=(label==1), p=p_home  (1 vs rest)
    - away_log_loss: y=(label==2), p=p_away  (2 vs rest)
    """
    y_home: list[int] = []
    y_draw: list[int] = []
    y_away: list[int] = []
    p_home: list[float] = []
    p_draw: list[float] = []
    p_away: list[float] = []
    scored = 0
    y_true_mc: list[int] = []
    y_prob_mc: list[list[float]] = []
    for row in rows:
        vector = _prob_vector(row, prob_keys)
        if vector is None:
            continue
        label = str(row.get("label", "")).strip().upper()
        if label not in LABEL_TO_INDEX:
            continue
        y_home.append(1 if label == "1" else 0)
        y_draw.append(1 if label == "X" else 0)
        y_away.append(1 if label == "2" else 0)
        p_home.append(vector[0])
        p_draw.append(vector[1])
        p_away.append(vector[2])
        y_true_mc.append(LABEL_TO_INDEX[label])
        y_prob_mc.append(vector)
        scored += 1

    return {
        "row_count": scored,
        "metric_definition": (
            "binary one-vs-rest: draw uses y=(label==X); "
            "home uses y=(label==1); away uses y=(label==2)"
        ),
        "draw_log_loss": _binary_log_loss(y_draw, p_draw),
        "draw_brier": _binary_brier(y_draw, p_draw),
        "home_log_loss": _binary_log_loss(y_home, p_home),
        "away_log_loss": _binary_log_loss(y_away, p_away),
        "pooled_multiclass_log_loss": (
            multiclass_log_loss(y_true_mc, y_prob_mc) if y_true_mc else None
        ),
    }


def draw_calibration_deciles(
    rows: list[dict[str, Any]],
    *,
    draw_key: str = "p_draw_blend",
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Predicted p_draw deciles vs empirical draw rate (optional calibration)."""
    scored: list[tuple[float, int]] = []
    for row in rows:
        raw = row.get(draw_key)
        if raw in (None, ""):
            continue
        try:
            probability = float(raw)
        except (TypeError, ValueError):
            continue
        is_draw = 1 if str(row.get("label", "")).strip().upper() == "X" else 0
        scored.append((probability, is_draw))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0])
    bin_size = max(1, len(scored) // n_bins)
    deciles: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        start = bin_index * bin_size
        end = len(scored) if bin_index == n_bins - 1 else (bin_index + 1) * bin_size
        chunk = scored[start:end]
        if not chunk:
            continue
        mean_pred = sum(item[0] for item in chunk) / len(chunk)
        actual_rate = sum(item[1] for item in chunk) / len(chunk)
        deciles.append(
            {
                "decile": bin_index + 1,
                "n": len(chunk),
                "mean_predicted_p_draw": mean_pred,
                "actual_draw_rate": actual_rate,
            }
        )
    return deciles


def _pre_draw_blend(row: dict[str, Any]) -> dict[str, float] | None:
    """Blend before draw adjust: prefer audit columns, else stored blend."""
    pre_keys = (
        "p_home_blend_pre_draw",
        "p_draw_blend_pre_draw",
        "p_away_blend_pre_draw",
    )
    if all(row.get(key) not in (None, "") for key in pre_keys):
        return {
            "1": float(row["p_home_blend_pre_draw"]),
            "X": float(row["p_draw_blend_pre_draw"]),
            "2": float(row["p_away_blend_pre_draw"]),
        }
    blend_keys = ("p_home_blend", "p_draw_blend", "p_away_blend")
    vector = _prob_vector(row, blend_keys)
    if vector is None:
        return None
    return {"1": vector[0], "X": vector[1], "2": vector[2]}


def apply_draw_adjustment_to_rows(
    rows: list[dict[str, Any]],
    *,
    draw_config: Any | None = None,
) -> list[dict[str, Any]]:
    """Return copies with p_*_blend set to draw-adjusted pre-draw blend."""
    from src.calc.draw_adjustment import apply_draw_adjustment, load_draw_adjustment_config

    config = draw_config if draw_config is not None else load_draw_adjustment_config()

    adjusted_rows: list[dict[str, Any]] = []
    for row in rows:
        blend = _pre_draw_blend(row)
        if blend is None:
            continue
        adjusted = apply_draw_adjustment(blend, row, config)
        if adjusted is None:
            continue
        copy = dict(row)
        copy["p_home_blend"] = adjusted["1"]
        copy["p_draw_blend"] = adjusted["X"]
        copy["p_away_blend"] = adjusted["2"]
        adjusted_rows.append(copy)
    return adjusted_rows


@dataclass
class AblationRowMetrics:
    name: str
    description: str
    row_count: int
    pooled_log_loss: float | None
    draw_log_loss: float | None
    draw_brier: float | None
    home_log_loss: float | None
    away_log_loss: float | None
    shrink_alpha: float | None = None


def run_phase3_ablation(
    rows: list[dict[str, Any]],
    trainer: Any | None = None,
    *,
    shrink_alpha: float = 0.3,
    draw_config: Any | None = None,
) -> dict[str, Any]:
    """Ablation A–D on existing CSV (no rebuild).

    A: stored blend (pre-rebuild may already be fixed 70/30 without draw adj)
    B: blend + draw adjustment (offline apply)
    C: B + HGB residual (requires trainer)
    D: C + shrink toward market
    """
    from src.calc.draw_adjustment import load_draw_adjustment_config

    config = draw_config if draw_config is not None else load_draw_adjustment_config()
    blend_keys = ("p_home_blend", "p_draw_blend", "p_away_blend")

    def _row_metrics(
        name: str,
        description: str,
        scored_rows: list[dict[str, Any]],
        keys: tuple[str, str, str] = blend_keys,
        *,
        shrink_alpha_value: float | None = None,
    ) -> AblationRowMetrics:
        outcome = score_outcome_metrics(scored_rows, keys)
        return AblationRowMetrics(
            name=name,
            description=description,
            row_count=int(outcome["row_count"]),
            pooled_log_loss=outcome["pooled_multiclass_log_loss"],
            draw_log_loss=outcome["draw_log_loss"],
            draw_brier=outcome["draw_brier"],
            home_log_loss=outcome["home_log_loss"],
            away_log_loss=outcome["away_log_loss"],
            shrink_alpha=shrink_alpha_value,
        )

    row_a = _row_metrics(
        "A",
        "Blend as stored in dataset (fixed or conditional; may lack draw adj)",
        rows,
    )

    rows_b = apply_draw_adjustment_to_rows(rows, draw_config=config)
    row_b = _row_metrics(
        "B",
        "Blend + explicit draw adjustment (offline)",
        rows_b,
    )

    result: dict[str, Any] = {
        "metric_definition": (
            "draw/home/away LL are binary one-vs-rest; "
            "pooled_log_loss is multiclass 1X2"
        ),
        "calibration_deciles_A": draw_calibration_deciles(rows),
        "calibration_deciles_B": draw_calibration_deciles(rows_b),
    }
    ablation_rows: dict[str, AblationRowMetrics] = {"A": row_a, "B": row_b}

    if trainer is not None and rows_b:
        ml_rows: list[dict[str, Any]] = []
        for row in rows_b:
            probs = trainer.predict_match_proba(row)
            market = _market_probs(row)
            if probs is None or market is None:
                continue
            copy = dict(row)
            copy["p_home_ml"] = probs["1"]
            copy["p_draw_ml"] = probs["X"]
            copy["p_away_ml"] = probs["2"]
            shrunk = shrink_toward_market(probs, market, alpha=shrink_alpha)
            copy["p_home_ml_shrink"] = shrunk["1"]
            copy["p_draw_ml_shrink"] = shrunk["X"]
            copy["p_away_ml_shrink"] = shrunk["2"]
            ml_rows.append(copy)

        ml_keys = ("p_home_ml", "p_draw_ml", "p_away_ml")
        shrink_keys = ("p_home_ml_shrink", "p_draw_ml_shrink", "p_away_ml_shrink")
        ablation_rows["C"] = _row_metrics(
            "C",
            "Blend + draw adj + HGB (predict on draw-adjusted blend columns)",
            ml_rows,
            ml_keys,
        )
        ablation_rows["D"] = _row_metrics(
            "D",
            f"C + shrink toward market (alpha={shrink_alpha})",
            ml_rows,
            shrink_keys,
            shrink_alpha_value=shrink_alpha,
        )
    else:
        ablation_rows["C"] = AblationRowMetrics(
            name="C",
            description="Blend + draw adj + HGB (model unavailable)",
            row_count=0,
            pooled_log_loss=None,
            draw_log_loss=None,
            draw_brier=None,
            home_log_loss=None,
            away_log_loss=None,
        )
        ablation_rows["D"] = AblationRowMetrics(
            name="D",
            description="C + shrink (model unavailable)",
            row_count=0,
            pooled_log_loss=None,
            draw_log_loss=None,
            draw_brier=None,
            home_log_loss=None,
            away_log_loss=None,
            shrink_alpha=shrink_alpha,
        )

    result["rows"] = {name: asdict(metrics) for name, metrics in ablation_rows.items()}
    return result
