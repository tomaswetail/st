"""Opening / ST residual gate: expanding-year fits, shrink α, large-move threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.calc.probability_metrics import mean_log_loss
from src.calc.residual_ml.baseline import apply_large_move_or_identity
from src.calc.residual_ml.calibration import split_validate_for_calibration
from src.calc.residual_ml.dual_price import (
    has_universe_triple,
    remap_universe_row,
)
from src.calc.residual_ml.folds import calendar_year_of_match_date, expanding_year_folds
from src.calc.residual_ml.models import get_model
from src.calc.residual_ml.trainer import market_from_row

Universe = Literal["opening", "st"]

SEED = 42
ALPHAS = (0.7, 0.8, 0.9)
THRESHOLDS = (0.02, 0.04, 0.06, 0.08)
ST_MIN_VAL_ROWS = 80
ST_MIN_ELIGIBLE_YEARS = 3
YEAR_MARGIN = 0.01

HGB_GRID: tuple[dict[str, Any], ...] = (
    {"max_iter": 200, "max_depth": 4, "learning_rate": 0.05},
    {"max_iter": 200, "max_depth": 6, "learning_rate": 0.05},
    {"max_iter": 300, "max_depth": 4, "learning_rate": 0.05},
    {"max_iter": 300, "max_depth": 6, "learning_rate": 0.05},
)
CATBOOST_GRID: tuple[dict[str, Any], ...] = (
    {"iterations": 300, "max_depth": 4, "learning_rate": 0.05},
    {"iterations": 300, "max_depth": 6, "learning_rate": 0.05},
    {"iterations": 500, "max_depth": 4, "learning_rate": 0.05},
    {"iterations": 500, "max_depth": 6, "learning_rate": 0.05},
)


def residual_configs() -> list[tuple[str, dict[str, Any]]]:
    configs: list[tuple[str, dict[str, Any]]] = []
    for hyper in HGB_GRID:
        configs.append(("hist_gradient", dict(hyper)))
    for hyper in CATBOOST_GRID:
        configs.append(("catboost", dict(hyper)))
    return configs


def prepare_universe_rows(
    rows: list[dict[str, Any]], universe: Universe
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if not has_universe_triple(row, universe):
            continue
        remapped = remap_universe_row(row, universe)
        if market_from_row(remapped) is None:
            continue
        prepared.append(remapped)
    return prepared


def closing_market_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    market = {
        "1": row.get("p_home_market_norm_closing"),
        "X": row.get("p_draw_market_norm_closing"),
        "2": row.get("p_away_market_norm_closing"),
    }
    if any(value is None for value in market.values()):
        return None
    return {key: float(value) for key, value in market.items()}


def predict_adjusted(
    model: Any,
    row: dict[str, Any],
    market: dict[str, float],
    *,
    alpha: float,
    threshold: float,
) -> dict[str, float]:
    ml_probabilities = model.predict_proba(row, market)
    return apply_large_move_or_identity(
        ml_probabilities, market, alpha=alpha, threshold=threshold
    )


def log_loss_for_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    threshold: float,
    market_fn=market_from_row,
) -> float | None:
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        market = market_fn(row)
        if market is None:
            continue
        final = predict_adjusted(
            model, row, market, alpha=alpha, threshold=threshold
        )
        labels.append(str(row["label"]))
        probability_rows.append((final["1"], final["X"], final["2"]))
    if not labels:
        return None
    return mean_log_loss(labels, probability_rows)


def baseline_log_loss(
    rows: list[dict[str, Any]], *, market_fn=market_from_row
) -> float | None:
    labels: list[str] = []
    probability_rows: list[tuple[float, float, float]] = []
    for row in rows:
        market = market_fn(row)
        if market is None:
            continue
        labels.append(str(row["label"]))
        probability_rows.append((market["1"], market["X"], market["2"]))
    if not labels:
        return None
    return mean_log_loss(labels, probability_rows)


def search_alpha_threshold(
    model: Any,
    search_rows: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Return (alpha, threshold, search_log_loss) minimizing outcome LL."""
    best: tuple[float, float, float] | None = None
    for alpha in ALPHAS:
        for threshold in THRESHOLDS:
            loss = log_loss_for_rows(
                model, search_rows, alpha=alpha, threshold=threshold
            )
            if loss is None:
                continue
            if best is None or loss < best[2]:
                best = (alpha, threshold, loss)
    if best is None:
        raise ValueError("alpha/threshold search has no scorable rows")
    return best


def last_train_year_rows(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_year = max(calendar_year_of_match_date(row.get("match_date")) for row in train_rows)
    return [
        row
        for row in train_rows
        if calendar_year_of_match_date(row.get("match_date")) == last_year
    ]


def decide_gate(
    year_rows: list[dict[str, Any]],
    *,
    universe: Universe,
) -> tuple[str, str]:
    """Return (decision, reason). Equal mean → REJECT."""
    eligible = [row for row in year_rows if row.get("eligible", True)]
    if universe == "st" and len(eligible) < ST_MIN_ELIGIBLE_YEARS:
        return "REJECT", "underpowered"
    if not eligible:
        return "REJECT", "no eligible years"
    model_mean = sum(row["model_ll"] for row in eligible) / len(eligible)
    baseline_mean = sum(row["baseline_ll"] for row in eligible) / len(eligible)
    if model_mean >= baseline_mean:
        return "REJECT", "mean not strictly better"
    for row in eligible:
        if row["model_ll"] >= row["baseline_ll"] + YEAR_MARGIN:
            return "REJECT", f"year {row['year']} >= baseline + {YEAR_MARGIN}"
    return "ACCEPT", "strict mean improvement and no year >= baseline+0.01"


@dataclass
class FoldYearResult:
    year: int
    n_train: int
    n_val: int
    n_score: int
    eligible: bool
    alpha: float | None
    threshold: float | None
    model_ll: float | None
    baseline_ll: float | None
    closing_ll: float | None = None
    note: str = ""


def evaluate_config(
    backend: str,
    hyperparams: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    universe: Universe,
) -> dict[str, Any]:
    folds = expanding_year_folds(rows)
    year_results: list[dict[str, Any]] = []
    for fold in folds:
        n_val = len(fold.validate_rows)
        eligible = True
        note = ""
        if universe == "st" and n_val < ST_MIN_VAL_ROWS:
            eligible = False
            note = f"n_val={n_val} < {ST_MIN_VAL_ROWS}"
            print(
                f"[{universe}] {backend} {hyperparams} year={fold.validate_year} "
                f"{note} (excluded from mean / +0.01 rule)",
                flush=True,
            )
            year_results.append(
                asdict(
                    FoldYearResult(
                        year=fold.validate_year,
                        n_train=len(fold.train_rows),
                        n_val=n_val,
                        n_score=n_val,
                        eligible=False,
                        alpha=None,
                        threshold=None,
                        model_ll=None,
                        baseline_ll=None,
                        note=note,
                    )
                )
            )
            continue

        model = get_model(backend, "residual", random_state=SEED, **hyperparams)
        print(
            f"[{universe}] fitting {backend} {hyperparams} "
            f"train_years={fold.train_years} val={fold.validate_year} "
            f"n_train={len(fold.train_rows)} n_val={n_val}",
            flush=True,
        )
        model.fit(fold.train_rows)

        if universe == "opening":
            calib_rows, score_rows = split_validate_for_calibration(fold.validate_rows)
            search_rows = calib_rows
            eval_rows = score_rows
        else:
            search_rows = last_train_year_rows(fold.train_rows)
            eval_rows = fold.validate_rows

        alpha, threshold, _search_ll = search_alpha_threshold(model, search_rows)
        model_ll = log_loss_for_rows(
            model, eval_rows, alpha=alpha, threshold=threshold
        )
        baseline_ll = baseline_log_loss(eval_rows)
        closing_ll = None
        if universe == "opening":
            closing_ll = baseline_log_loss(
                eval_rows, market_fn=closing_market_from_row
            )

        year_results.append(
            asdict(
                FoldYearResult(
                    year=fold.validate_year,
                    n_train=len(fold.train_rows),
                    n_val=n_val,
                    n_score=len(eval_rows),
                    eligible=eligible,
                    alpha=alpha,
                    threshold=threshold,
                    model_ll=model_ll,
                    baseline_ll=baseline_ll,
                    closing_ll=closing_ll,
                    note=note,
                )
            )
        )
        print(
            f"[{universe}] year={fold.validate_year} n_score={len(eval_rows)} "
            f"model_ll={model_ll} baseline_ll={baseline_ll} "
            f"alpha={alpha} threshold={threshold} closing_ll={closing_ll}",
            flush=True,
        )

    eligible_rows = [
        row for row in year_results if row["eligible"] and row["model_ll"] is not None
    ]
    if eligible_rows:
        mean_model = sum(row["model_ll"] for row in eligible_rows) / len(eligible_rows)
        mean_baseline = sum(row["baseline_ll"] for row in eligible_rows) / len(
            eligible_rows
        )
    else:
        mean_model = None
        mean_baseline = None
    decision, reason = decide_gate(year_results, universe=universe)
    return {
        "backend": backend,
        "hyperparams": hyperparams,
        "years": year_results,
        "mean_model_ll": mean_model,
        "mean_baseline_ll": mean_baseline,
        "decision": decision,
        "reason": reason,
        "n_eligible_years": len(eligible_rows),
    }


def select_best_config(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [
        result
        for result in results
        if result.get("mean_model_ll") is not None and result["n_eligible_years"] > 0
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda result: (
            result["mean_model_ll"],
            result["backend"],
            str(result["hyperparams"]),
        ),
    )


def chosen_alpha_threshold(best: dict[str, Any]) -> dict[str, Any]:
    eligible = [
        row
        for row in best["years"]
        if row["eligible"] and row["alpha"] is not None
    ]
    latest = eligible[-1] if eligible else None
    return {
        "latest_eligible_year": None if latest is None else latest["year"],
        "alpha": None if latest is None else latest["alpha"],
        "threshold": None if latest is None else latest["threshold"],
        "per_year": [
            {
                "year": row["year"],
                "alpha": row["alpha"],
                "threshold": row["threshold"],
                "eligible": row["eligible"],
            }
            for row in best["years"]
        ],
    }


def run_universe_gate(
    rows: list[dict[str, Any]],
    *,
    universe: Universe,
) -> dict[str, Any]:
    prepared = prepare_universe_rows(rows, universe)
    print(
        f"[{universe}] prepared {len(prepared)} / {len(rows)} rows",
        flush=True,
    )
    config_results: list[dict[str, Any]] = []
    for backend, hyperparams in residual_configs():
        result = evaluate_config(backend, hyperparams, prepared, universe=universe)
        config_results.append(result)
        print(
            f"[{universe}] {backend} {hyperparams} mean_model={result['mean_model_ll']} "
            f"mean_baseline={result['mean_baseline_ll']} {result['decision']}",
            flush=True,
        )
    best = select_best_config(config_results)
    if best is None:
        return {
            "universe": universe,
            "n_rows": len(prepared),
            "decision": "REJECT",
            "reason": "no scorable configs",
            "chosen": None,
            "configs": config_results,
        }
    decision, reason = decide_gate(best["years"], universe=universe)
    return {
        "universe": universe,
        "n_rows": len(prepared),
        "decision": decision,
        "reason": reason,
        "chosen": {
            "backend": best["backend"],
            "hyperparams": best["hyperparams"],
            **chosen_alpha_threshold(best),
            "mean_model_ll": best["mean_model_ll"],
            "mean_baseline_ll": best["mean_baseline_ll"],
            "years": best["years"],
        },
        "configs": config_results,
    }
