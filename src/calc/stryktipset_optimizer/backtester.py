"""Chronological OOS historical parameter tuner.

Supports:
- VALUE mode: primary ranking by mean portfolio leverage Σ log(Pm/Pp)
  (prior β-leverage backtests apply here only).
- PREDICTION mode: primary ranking by mean coverage objective
  (P_full / P12+ / P11+ / E[best_correct]); does NOT use leverage primary.

Orders coupons by min(start_time) else draw_number — never shuffles time.

Leakage UNKNOWN: odds/public shares lack timestamps; regCloseTime not on
STRoundModel. Do not claim closing-line safety from this backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from src.calc.stryktipset_optimizer.data import (
    LEAKAGE_LIMITATIONS,
    PreparedCoupon,
)
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import (
    OptimizerMode,
    OptimizerParams,
    PredictionObjective,
)
from src.calc.stryktipset_optimizer.simulator import simulate_pool
from src.utils.common import Outcome

BacktestPrimaryMetric = Literal[
    "portfolio_leverage",
    "p_full",
    "p_12_or_better",
    "p_11_or_better",
    "expected_best_correct",
]


@dataclass(frozen=True)
class ParamConfig:
    """One grid point for historical tuning."""

    mode: OptimizerMode = "VALUE"
    objective: PredictionObjective = "MAX_P13"
    beta: float = 1.0
    lambda_diversity: float = 0.5
    banker_value_weight: float = 1.0
    candidate_count: int = 500
    prediction_candidate_count: int = 2000
    row_count: int = 1
    reduced_system: bool = False


@dataclass(frozen=True)
class RoundEval:
    draw_number: int | None
    sort_key: tuple
    # VALUE primary (comparable across β):
    portfolio_leverage: float
    top_leverage_score: float
    # PREDICTION primary / diagnostics:
    coverage_p_full: float
    coverage_p_12_or_better: float
    coverage_p_11_or_better: float
    coverage_expected_best: float
    # Diagnostics — construction row_score is NOT comparable across β:
    row_score_top: float
    realized_row_score: float | None
    mean_correct: float
    tier13_rate: float


@dataclass(frozen=True)
class BacktestConfigResult:
    params: ParamConfig
    rounds_evaluated: int
    rounds_skipped: int
    mean_portfolio_leverage: float
    mean_coverage_p_full: float
    mean_coverage_p_12_or_better: float
    mean_coverage_p_11_or_better: float
    mean_coverage_expected_best: float
    # Diagnostics only:
    mean_top_row_score: float
    mean_realized_row_score: float | None
    mean_correct: float
    mean_tier13_rate: float
    round_evals: tuple[RoundEval, ...]
    limitations: str


@dataclass(frozen=True)
class BacktestResult:
    results: tuple[BacktestConfigResult, ...]
    mode: OptimizerMode
    primary_metric: BacktestPrimaryMetric
    best_by_primary: ParamConfig | None
    # Backward-compatible alias for VALUE leverage ranking:
    best_by_mean_portfolio_leverage: ParamConfig | None
    limitations: str = LEAKAGE_LIMITATIONS


def coupon_sort_key(coupon: PreparedCoupon) -> tuple:
    """Chronological key: min(start_time) else draw_number. No shuffle."""
    start_times = [
        match.start_time
        for match in coupon.matches
        if match.start_time is not None
    ]
    draw = coupon.draw_number if coupon.draw_number is not None else 0
    if start_times:
        return (0, min(start_times), draw)
    return (1, datetime.min, draw)


def order_coupons_chronologically(
    coupons: Sequence[PreparedCoupon],
) -> list[PreparedCoupon]:
    return sorted(coupons, key=coupon_sort_key)


def leverage_score_for_outcomes(
    coupon: PreparedCoupon,
    outcomes: Sequence[Outcome],
) -> float:
    """Σ log(Pm/Pp) — row_score at β=1; comparable across construction β."""
    if len(outcomes) != len(coupon.matches):
        raise ValueError("outcomes length must match coupon matches")
    total = 0.0
    for match, outcome in zip(coupon.matches, outcomes):
        pm = float(match.market_probs[outcome])
        pp = float(match.public_probs[outcome])
        total += math.log(pm / pp)
    return total


def realized_row_score_for_outcomes(
    coupon: PreparedCoupon,
    outcomes: Sequence[Outcome],
    beta: float,
) -> float:
    """log Pm(actual) - beta * log Pp(actual) for a full outcome vector."""
    if len(outcomes) != len(coupon.matches):
        raise ValueError("outcomes length must match coupon matches")
    total = 0.0
    for match, outcome in zip(coupon.matches, outcomes):
        pm = float(match.market_probs[outcome])
        pp = float(match.public_probs[outcome])
        total += math.log(pm) - beta * math.log(pp)
    return total


def _primary_for_mode(
    mode: OptimizerMode,
    objective: PredictionObjective,
) -> BacktestPrimaryMetric:
    if mode == "VALUE":
        return "portfolio_leverage"
    if objective == "MAX_P13":
        return "p_full"
    if objective == "MAX_P12_OR_BETTER":
        return "p_12_or_better"
    if objective == "MAX_P11_OR_BETTER":
        return "p_11_or_better"
    if objective == "MAX_EXPECTED_CORRECT":
        return "expected_best_correct"
    return "p_full"


def _primary_value(
    item: BacktestConfigResult,
    metric: BacktestPrimaryMetric,
) -> float:
    if metric == "portfolio_leverage":
        return item.mean_portfolio_leverage
    if metric == "p_full":
        return item.mean_coverage_p_full
    if metric == "p_12_or_better":
        return item.mean_coverage_p_12_or_better
    if metric == "p_11_or_better":
        return item.mean_coverage_p_11_or_better
    if metric == "expected_best_correct":
        return item.mean_coverage_expected_best
    raise ValueError(f"unknown primary metric: {metric}")


def evaluate_coupon_against_truth(
    coupon: PreparedCoupon,
    params: ParamConfig,
    *,
    public_epsilon: float = 1e-6,
    seed: int = 0,
    n_simulations: int = 200,
) -> RoundEval | None:
    """Optimize one coupon; record mode-appropriate metrics + diagnostics."""
    if len(coupon.matches) == 0:
        return None

    optimizer = CouponOptimizer(
        OptimizerParams(
            mode=params.mode,
            objective=params.objective,
            beta=params.beta,
            lambda_diversity=params.lambda_diversity,
            banker_value_weight=params.banker_value_weight,
            candidate_count=params.candidate_count,
            prediction_candidate_count=params.prediction_candidate_count,
            public_epsilon=public_epsilon,
            coupon_size=len(coupon.matches),
            seed=seed,
            reduced_system=params.reduced_system,
        )
    )
    result = optimizer.optimize_prepared(coupon, row_count=params.row_count)
    top_score = result.rows[0].row_score if result.rows else float("nan")
    our_rows = [tuple(row.outcomes) for row in result.rows]

    leverages = [
        leverage_score_for_outcomes(coupon, row) for row in our_rows
    ]
    portfolio_leverage = sum(leverages) / len(leverages)
    top_leverage = max(leverages)

    coverage = result.coverage
    if coverage is None:
        raise RuntimeError("optimizer result missing coverage block")

    realized: float | None = None
    if all(match.result is not None for match in coupon.matches):
        truth: list[Outcome] = [match.result for match in coupon.matches]  # type: ignore[misc]
        realized = realized_row_score_for_outcomes(coupon, truth, params.beta)
        best_correct = max(
            sum(1 for a, b in zip(row, truth) if a == b) for row in our_rows
        )
        mean_correct = float(best_correct)
        tier13 = 1.0 if best_correct >= len(coupon.matches) else 0.0
    else:
        market_probs = [m.market_probs for m in coupon.matches]
        public_probs = [m.public_probs for m in coupon.matches]
        sim = simulate_pool(
            market_probs,
            public_probs,
            our_rows,
            n_simulations=n_simulations,
            seed=seed,
            public_row_count=50,
        )
        mean_correct = sim.mean_correct_ours
        tier13 = sim.our_tier_rates["13"]

    return RoundEval(
        draw_number=coupon.draw_number,
        sort_key=coupon_sort_key(coupon),
        portfolio_leverage=portfolio_leverage,
        top_leverage_score=top_leverage,
        coverage_p_full=coverage.p_full,
        coverage_p_12_or_better=coverage.p_12_or_better,
        coverage_p_11_or_better=coverage.p_11_or_better,
        coverage_expected_best=coverage.expected_best_correct,
        row_score_top=top_score,
        realized_row_score=realized,
        mean_correct=mean_correct,
        tier13_rate=tier13,
    )


def run_backtest(
    coupons: Sequence[PreparedCoupon],
    param_grid: Sequence[ParamConfig],
    *,
    public_epsilon: float = 1e-6,
    seed: int = 0,
    n_simulations: int = 200,
    mode: OptimizerMode | None = None,
) -> BacktestResult:
    """Chronological OOS: evaluate each param config on ordered coupons.

    If ``mode`` is set, all grid points must match that mode (or grid mode
    fields are overridden). Primary ranking:
    - VALUE → mean portfolio leverage
    - PREDICTION → coverage metric matching objective
    Does not shuffle time. Does not silently score PREDICTION with leverage.
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    resolved_mode: OptimizerMode = mode or param_grid[0].mode
    for config in param_grid:
        cfg_mode = mode or config.mode
        if cfg_mode != resolved_mode:
            raise ValueError(
                "mixed modes in one backtest run are not allowed; "
                f"got {resolved_mode!r} and {cfg_mode!r}"
            )

    # Use first config's objective to pick PREDICTION primary when uniform;
    # if objectives differ within PREDICTION, still rank each by its own
    # objective via best_by_primary using the first's metric only when all
    # share it — otherwise require a single objective.
    objectives = {config.objective for config in param_grid}
    if resolved_mode == "PREDICTION" and len(objectives) > 1:
        raise ValueError(
            "PREDICTION backtest grid must share one objective "
            f"(got {sorted(objectives)})"
        )
    objective = param_grid[0].objective
    primary = _primary_for_mode(resolved_mode, objective)

    ordered = order_coupons_chronologically(coupons)
    config_results: list[BacktestConfigResult] = []

    for config in param_grid:
        effective = ParamConfig(
            mode=resolved_mode,
            objective=config.objective,
            beta=config.beta,
            lambda_diversity=config.lambda_diversity,
            banker_value_weight=config.banker_value_weight,
            candidate_count=config.candidate_count,
            prediction_candidate_count=config.prediction_candidate_count,
            row_count=config.row_count,
            reduced_system=config.reduced_system,
        )
        round_evals: list[RoundEval] = []
        skipped = 0
        for coupon in ordered:
            try:
                evaluation = evaluate_coupon_against_truth(
                    coupon,
                    effective,
                    public_epsilon=public_epsilon,
                    seed=seed,
                    n_simulations=n_simulations,
                )
            except Exception:
                skipped += 1
                continue
            if evaluation is None:
                skipped += 1
                continue
            round_evals.append(evaluation)

        n = len(round_evals)
        if n == 0:
            config_results.append(
                BacktestConfigResult(
                    params=effective,
                    rounds_evaluated=0,
                    rounds_skipped=skipped + len(ordered),
                    mean_portfolio_leverage=float("nan"),
                    mean_coverage_p_full=float("nan"),
                    mean_coverage_p_12_or_better=float("nan"),
                    mean_coverage_p_11_or_better=float("nan"),
                    mean_coverage_expected_best=float("nan"),
                    mean_top_row_score=float("nan"),
                    mean_realized_row_score=None,
                    mean_correct=float("nan"),
                    mean_tier13_rate=float("nan"),
                    round_evals=(),
                    limitations=LEAKAGE_LIMITATIONS,
                )
            )
            continue

        realized_vals = [
            r.realized_row_score
            for r in round_evals
            if r.realized_row_score is not None
        ]
        mean_realized = (
            sum(realized_vals) / len(realized_vals) if realized_vals else None
        )

        config_results.append(
            BacktestConfigResult(
                params=effective,
                rounds_evaluated=n,
                rounds_skipped=skipped,
                mean_portfolio_leverage=(
                    sum(r.portfolio_leverage for r in round_evals) / n
                ),
                mean_coverage_p_full=(
                    sum(r.coverage_p_full for r in round_evals) / n
                ),
                mean_coverage_p_12_or_better=(
                    sum(r.coverage_p_12_or_better for r in round_evals) / n
                ),
                mean_coverage_p_11_or_better=(
                    sum(r.coverage_p_11_or_better for r in round_evals) / n
                ),
                mean_coverage_expected_best=(
                    sum(r.coverage_expected_best for r in round_evals) / n
                ),
                mean_top_row_score=sum(r.row_score_top for r in round_evals) / n,
                mean_realized_row_score=mean_realized,
                mean_correct=sum(r.mean_correct for r in round_evals) / n,
                mean_tier13_rate=sum(r.tier13_rate for r in round_evals) / n,
                round_evals=tuple(round_evals),
                limitations=LEAKAGE_LIMITATIONS,
            )
        )

    scored = [
        item
        for item in config_results
        if item.rounds_evaluated > 0
        and _primary_value(item, primary) == _primary_value(item, primary)
    ]
    best = None
    if scored:
        best = max(scored, key=lambda item: _primary_value(item, primary)).params

    leverage_best = None
    if resolved_mode == "VALUE":
        leverage_best = best
    else:
        # Do not pretend leverage is the PREDICTION primary.
        leverage_best = None

    limitations = LEAKAGE_LIMITATIONS
    if resolved_mode == "PREDICTION":
        limitations = (
            f"{LEAKAGE_LIMITATIONS} | PREDICTION backtest primary="
            f"{primary}; prior β-leverage OOS reports are VALUE-only "
            "and do not validate PREDICTION."
        )

    return BacktestResult(
        results=tuple(config_results),
        mode=resolved_mode,
        primary_metric=primary,
        best_by_primary=best,
        best_by_mean_portfolio_leverage=leverage_best,
        limitations=limitations,
    )
