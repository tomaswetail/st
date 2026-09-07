"""Chronological OOS historical parameter tuner.

Orders coupons by min(start_time) else draw_number — never shuffles time.
Aggregates relative metrics per parameter config. Skips incomplete rounds.

Primary ranking uses mean portfolio leverage
``Σ log(Pm/Pp)`` (≡ row_score at β=1), which is comparable across construction
β / λ. Construction ``row_score`` / ``mean_top_row_score`` inflate with β and
are diagnostics only. mean_correct / tier rates are also diagnostics only.

Leakage UNKNOWN: odds/public shares lack timestamps; regCloseTime not on
STRoundModel. Do not claim closing-line safety from this backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from src.calc.stryktipset_optimizer.data import (
    LEAKAGE_LIMITATIONS,
    PreparedCoupon,
)
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.calc.stryktipset_optimizer.simulator import simulate_pool
from src.utils.common import Outcome


@dataclass(frozen=True)
class ParamConfig:
    """One grid point for historical tuning."""

    beta: float = 1.0
    lambda_diversity: float = 0.5
    banker_value_weight: float = 1.0
    candidate_count: int = 500
    row_count: int = 1


@dataclass(frozen=True)
class RoundEval:
    draw_number: int | None
    sort_key: tuple
    # Cross-β comparable primary metric (mean leverage of selected rows):
    portfolio_leverage: float
    top_leverage_score: float
    # Diagnostics — construction row_score is NOT comparable across β:
    row_score_top: float
    realized_row_score: float | None  # actual outcome under construction beta
    mean_correct: float
    tier13_rate: float


@dataclass(frozen=True)
class BacktestConfigResult:
    params: ParamConfig
    rounds_evaluated: int
    rounds_skipped: int
    mean_portfolio_leverage: float
    # Diagnostics only:
    mean_top_row_score: float  # not comparable across β
    mean_realized_row_score: float | None
    mean_correct: float
    mean_tier13_rate: float
    round_evals: tuple[RoundEval, ...]
    limitations: str


@dataclass(frozen=True)
class BacktestResult:
    results: tuple[BacktestConfigResult, ...]
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


def evaluate_coupon_against_truth(
    coupon: PreparedCoupon,
    params: ParamConfig,
    *,
    public_epsilon: float = 1e-6,
    seed: int = 0,
    n_simulations: int = 200,
) -> RoundEval | None:
    """Optimize one coupon; record cross-β leverage + descriptive diagnostics."""
    if len(coupon.matches) == 0:
        return None

    optimizer = CouponOptimizer(
        OptimizerParams(
            beta=params.beta,
            lambda_diversity=params.lambda_diversity,
            banker_value_weight=params.banker_value_weight,
            candidate_count=params.candidate_count,
            public_epsilon=public_epsilon,
            coupon_size=len(coupon.matches),
            seed=seed,
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
            public_row_count=50,  # keep unsettled MC light in backtest loops
        )
        mean_correct = sim.mean_correct_ours
        tier13 = sim.our_tier_rates["13"]

    return RoundEval(
        draw_number=coupon.draw_number,
        sort_key=coupon_sort_key(coupon),
        portfolio_leverage=portfolio_leverage,
        top_leverage_score=top_leverage,
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
) -> BacktestResult:
    """Chronological OOS: evaluate each param config on ordered coupons.

    Primary ranking: ``best_by_mean_portfolio_leverage`` (Σ log(Pm/Pp) of
    selected rows — comparable across β). Construction row_score and
    mean_correct are diagnostics only. Does not shuffle time.
    """
    ordered = order_coupons_chronologically(coupons)
    config_results: list[BacktestConfigResult] = []

    for config in param_grid:
        round_evals: list[RoundEval] = []
        skipped = 0
        for coupon in ordered:
            try:
                evaluation = evaluate_coupon_against_truth(
                    coupon,
                    config,
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
                    params=config,
                    rounds_evaluated=0,
                    rounds_skipped=skipped + len(ordered),
                    mean_portfolio_leverage=float("nan"),
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
                params=config,
                rounds_evaluated=n,
                rounds_skipped=skipped,
                mean_portfolio_leverage=(
                    sum(r.portfolio_leverage for r in round_evals) / n
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
        and item.mean_portfolio_leverage == item.mean_portfolio_leverage
    ]
    best = None
    if scored:
        best = max(scored, key=lambda item: item.mean_portfolio_leverage).params

    return BacktestResult(
        results=tuple(config_results),
        best_by_mean_portfolio_leverage=best,
        limitations=LEAKAGE_LIMITATIONS,
    )
