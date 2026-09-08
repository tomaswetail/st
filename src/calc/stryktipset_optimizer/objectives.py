"""Approximate PREDICTION objectives beyond exact MAX_P13.

MAX_P12_OR_BETTER / MAX_P11_OR_BETTER / MAX_EXPECTED_CORRECT:
exact set selection is combinatorial-hard. This module uses greedy
marginal-gain from the empty set over a candidate pool of top-C market
rows (from prediction.top_market_rows). Each addition is scored by exact
coverage metrics on the growing set.

Results MUST be labeled approximate (exact_selection=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.calc.stryktipset_optimizer.coverage import (
    CoverageMetrics,
    compute_coverage,
)
from src.calc.stryktipset_optimizer.prediction import MarketRow, top_market_rows
from src.utils.common import Outcome

PredictionObjective = Literal[
    "MAX_P13",
    "MAX_P12_OR_BETTER",
    "MAX_P11_OR_BETTER",
    "MAX_EXPECTED_CORRECT",
]

APPROX_OBJECTIVES: frozenset[str] = frozenset(
    {
        "MAX_P12_OR_BETTER",
        "MAX_P11_OR_BETTER",
        "MAX_EXPECTED_CORRECT",
    }
)


@dataclass(frozen=True)
class ApproxPortfolio:
    """Greedy approximate portfolio for non-P13 PREDICTION objectives."""

    rows: tuple[MarketRow, ...]
    coverage: CoverageMetrics
    objective: PredictionObjective
    candidate_pool_size: int
    exact_selection: bool = False
    limitations: str = (
        "Approximate selection: greedy marginal-gain over top-C market "
        "rows; not guaranteed optimal for MAX_P12+/MAX_EXPECTED_CORRECT."
    )


def _objective_value(
    coverage: CoverageMetrics,
    objective: PredictionObjective,
) -> float:
    if objective == "MAX_P13":
        return coverage.p_full
    if objective == "MAX_P12_OR_BETTER":
        return coverage.p_12_or_better
    if objective == "MAX_P11_OR_BETTER":
        return coverage.p_11_or_better
    if objective == "MAX_EXPECTED_CORRECT":
        return coverage.expected_best_correct
    raise ValueError(f"unsupported objective: {objective}")


def greedy_approximate_portfolio(
    market_probs: Sequence[dict[Outcome, float]],
    *,
    objective: PredictionObjective,
    row_count: int,
    candidate_count: int = 2000,
) -> ApproxPortfolio:
    """Greedy: repeatedly add the unevaluated candidate with best Δ objective.

    Candidate pool = top ``candidate_count`` rows by joint market P.
    Coverage is exact for each evaluated set; selection is approximate.
    """
    if objective == "MAX_P13":
        raise ValueError(
            "MAX_P13 must use exact top_market_rows, not greedy approx"
        )
    if objective not in APPROX_OBJECTIVES:
        raise ValueError(f"unsupported approx objective: {objective}")
    if row_count < 1:
        raise ValueError(f"row_count must be >= 1, got {row_count}")
    if candidate_count < 1:
        raise ValueError(f"candidate_count must be >= 1, got {candidate_count}")

    pool = top_market_rows(market_probs, row_count=candidate_count)
    selected: list[MarketRow] = []
    remaining = list(pool)
    current_value = float("-inf")

    for _ in range(min(row_count, len(remaining))):
        best_index = -1
        best_value = float("-inf")
        best_coverage: CoverageMetrics | None = None

        for index, candidate in enumerate(remaining):
            trial_rows = [row.outcomes for row in selected] + [
                candidate.outcomes
            ]
            coverage = compute_coverage(market_probs, trial_rows)
            value = _objective_value(coverage, objective)
            # Prefer higher value; ties keep earlier (higher P(row)) candidate.
            if value > best_value + 1e-15 or (
                abs(value - best_value) <= 1e-15 and best_index < 0
            ):
                best_value = value
                best_index = index
                best_coverage = coverage
            elif abs(value - best_value) <= 1e-15 and index < best_index:
                best_index = index
                best_coverage = coverage

        if best_index < 0 or best_coverage is None:
            break
        # Stop early if no improvement (except first pick).
        if selected and best_value < current_value - 1e-15:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        current_value = best_value

    if not selected:
        raise ValueError("greedy approx produced empty portfolio")

    final_coverage = compute_coverage(
        market_probs, [row.outcomes for row in selected]
    )
    return ApproxPortfolio(
        rows=tuple(selected),
        coverage=final_coverage,
        objective=objective,
        candidate_pool_size=len(pool),
        exact_selection=False,
    )
