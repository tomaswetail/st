"""MAX_P13 scoring helpers for fixture packs and real coupons."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from src.calc.probability_metrics import clip_and_normalize_probs
from src.calc.residual_ml.packs import pack_consecutive_pairs
from src.calc.stryktipset_optimizer.coverage import (
    CoverageMetrics,
    row_joint_probability,
)
from src.calc.stryktipset_optimizer.data import PreparedCoupon, PreparedMatch
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.objects.schema.data_classes.stryktipset_optimizer import (
    StryktipsetOptimizationResult,
)
from src.utils.common import OUTCOMES, Outcome

EQUAL_PUBLIC_SHARES: dict[Outcome, float] = {"1": 1.0 / 3.0, "X": 1.0 / 3.0, "2": 1.0 / 3.0}
VALID_LABELS = frozenset({"1", "X", "2"})


@dataclass(frozen=True)
class MaxP13Score:
    """One MAX_P13 portfolio evaluation."""

    predicted_p13: float
    covered13: bool
    best_correct: int
    realized_rank: int
    n_matches: int
    row_count: int


def realized_outcome_tuple(coupon: PreparedCoupon) -> tuple[Outcome, ...]:
    """Return the realized 1X2 tuple; missing results raise."""
    outcomes: list[Outcome] = []
    for match in coupon.matches:
        if match.result not in OUTCOMES:
            raise ValueError(f"match {match.match_index} has no realized result")
        outcomes.append(match.result)
    return tuple(outcomes)


def selected_outcome_tuples(
    result: StryktipsetOptimizationResult,
) -> list[tuple[Outcome, ...]]:
    return [tuple(row.outcomes) for row in result.rows]


def best_correct_for_truth(
    selected: Sequence[Sequence[Outcome]],
    truth: Sequence[Outcome],
) -> int:
    if not selected:
        return 0
    return max(
        sum(predicted == actual for predicted, actual in zip(row, truth))
        for row in selected
    )


def realized_rank_in_portfolio(
    selected: Sequence[Sequence[Outcome]],
    truth: Sequence[Outcome],
    *,
    row_count: int,
) -> int:
    """1-based index of ``truth`` in optimizer order, else ``row_count + 1``."""
    truth_tuple = tuple(truth)
    for index, row in enumerate(selected, start=1):
        if tuple(row) == truth_tuple:
            return index
    return row_count + 1


def score_max_p13_result(
    result: StryktipsetOptimizationResult,
    truth: Sequence[Outcome],
    *,
    row_count: int,
) -> MaxP13Score:
    if result.coverage is None:
        raise ValueError("optimizer result has no coverage")
    selected = selected_outcome_tuples(result)
    truth_tuple = tuple(truth)
    covered = truth_tuple in selected
    best_correct = best_correct_for_truth(selected, truth_tuple)
    return MaxP13Score(
        predicted_p13=float(result.coverage.p_full),
        covered13=covered,
        best_correct=best_correct,
        realized_rank=realized_rank_in_portfolio(
            selected, truth_tuple, row_count=row_count
        ),
        n_matches=len(truth_tuple),
        row_count=row_count,
    )


def max_p13_selected_coverage(
    market_probs: Sequence[dict[str, float]],
    rows: Sequence[Sequence[Outcome]],
) -> CoverageMetrics:
    """Exact ``p_full`` for MAX_P13: sum of joint P of selected (disjoint) rows.

    Other coverage slots are not used by Phase 7 (covered13 is membership).
    This is the same ``p_full`` as full 3^N enumeration for unique rows.
    """
    n_matches = len(market_probs)
    p_full = sum(row_joint_probability(market_probs, row) for row in rows)
    hist = [0.0] * (n_matches + 1)
    hist[n_matches] = p_full
    hist[0] = max(0.0, 1.0 - p_full)
    return CoverageMetrics(
        n_matches=n_matches,
        p_best_correct=tuple(hist),
        p_full=p_full,
        p_12_or_better=p_full,
        p_11_or_better=p_full,
        p_10_or_better=p_full,
        expected_best_correct=p_full * n_matches,
    )


def optimize_max_p13(
    coupon: PreparedCoupon,
    *,
    row_count: int,
) -> StryktipsetOptimizationResult:
    """Call ``CouponOptimizer.optimize_prepared`` (PREDICTION / MAX_P13).

    The heap is unchanged. Coverage ``p_full`` uses the equivalent sum of
    selected joint probabilities so 13-match packs do not enumerate 3^13.
    """
    import src.calc.stryktipset_optimizer.optimize as optimize_mod

    optimizer = CouponOptimizer(
        OptimizerParams(mode="PREDICTION", objective="MAX_P13")
    )
    original_coverage = optimize_mod.compute_coverage
    optimize_mod.compute_coverage = max_p13_selected_coverage
    try:
        return optimizer.optimize_prepared(coupon, row_count=row_count)
    finally:
        optimize_mod.compute_coverage = original_coverage


def score_coupon_max_p13(
    coupon: PreparedCoupon,
    *,
    row_count: int,
) -> MaxP13Score:
    result = optimize_max_p13(coupon, row_count=row_count)
    return score_max_p13_result(
        result,
        realized_outcome_tuple(coupon),
        row_count=row_count,
    )


def coupon_from_probability_rows(
    probability_rows: Sequence[tuple[float, float, float]],
    labels: Sequence[str],
) -> PreparedCoupon:
    """Build a coupon whose ``market_probs`` are the given 1X2 vectors."""
    if len(probability_rows) != len(labels):
        raise ValueError("probability_rows and labels length mismatch")
    matches: list[PreparedMatch] = []
    for index, ((p_home, p_draw, p_away), label) in enumerate(
        zip(probability_rows, labels)
    ):
        normalized = clip_and_normalize_probs(p_home, p_draw, p_away)
        result: Outcome | None
        cleaned = str(label).strip().upper()
        result = cleaned if cleaned in VALID_LABELS else None  # type: ignore[assignment]
        matches.append(
            PreparedMatch(
                match_index=index,
                market_probs={
                    "1": normalized[0],
                    "X": normalized[1],
                    "2": normalized[2],
                },
                public_probs=dict(EQUAL_PUBLIC_SHARES),
                label=None,
                external_id=None,
                start_time=None,
                result=result,
            )
        )
    return PreparedCoupon(draw_number=None, matches=tuple(matches))


def _score_pack_payload(
    payload: tuple[
        list[tuple[float, float, float]],
        list[str],
        int,
    ],
) -> tuple[float, float] | None:
    pack_probs, pack_labels, row_count = payload
    if any(label not in VALID_LABELS for label in pack_labels):
        return None
    coupon = coupon_from_probability_rows(pack_probs, pack_labels)
    score = score_coupon_max_p13(coupon, row_count=row_count)
    return score.predicted_p13, 1.0 if score.covered13 else 0.0


def _score_coupon_payload(
    payload: tuple[PreparedCoupon, int],
) -> MaxP13Score:
    coupon, row_count = payload
    return score_coupon_max_p13(coupon, row_count=row_count)


def _worker_count(requested: int | None) -> int:
    available = os.cpu_count() or 1
    if requested is None:
        return max(1, available)
    return max(1, min(int(requested), available))


def score_p13_on_packs(
    rows: list[dict[str, Any]],
    probability_rows: Sequence[tuple[float, float, float]],
    *,
    row_count: int,
    coupon_size: int = 13,
    workers: int | None = None,
) -> tuple[float | None, float | None, int]:
    """Mean predicted P13 and covered13 over consecutive packs.

    Returns ``(mean_p13, mean_covered13, n_packs)``. No packs → ``(None, None, 0)``.
    """
    if len(rows) != len(probability_rows):
        raise ValueError("rows and probability_rows length mismatch")
    packs = pack_consecutive_pairs(
        rows, probability_rows, coupon_size=coupon_size
    )
    if not packs:
        return None, None, 0

    payloads = [
        (
            pack_probs,
            [str(row.get("label", "")).strip().upper() for row in pack_rows],
            row_count,
        )
        for pack_rows, pack_probs in packs
    ]
    worker_count = _worker_count(workers)
    if worker_count == 1 or len(payloads) == 1:
        scored = [_score_pack_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            scored = list(pool.map(_score_pack_payload, payloads))

    predicted: list[float] = []
    covered_flags: list[float] = []
    for item in scored:
        if item is None:
            continue
        predicted.append(item[0])
        covered_flags.append(item[1])
    if not predicted:
        return None, None, 0
    n_packs = len(predicted)
    return (
        sum(predicted) / n_packs,
        sum(covered_flags) / n_packs,
        n_packs,
    )


def score_coupons_max_p13(
    coupons: Sequence[PreparedCoupon],
    *,
    row_count: int,
    workers: int | None = None,
) -> list[MaxP13Score]:
    """Score many coupons with optional process parallelism."""
    if not coupons:
        return []
    payloads = [(coupon, row_count) for coupon in coupons]
    worker_count = _worker_count(workers)
    if worker_count == 1 or len(payloads) == 1:
        return [_score_coupon_payload(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(_score_coupon_payload, payloads))
