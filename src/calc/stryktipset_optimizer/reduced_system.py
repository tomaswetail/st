"""Doubles/triples reduced-system builder (PREDICTION).

Traditional systems: each match has 1/2/3 signs; system size = product of
branch factors. This builder starts from all-favorite singles and iteratively
upgrades single→double or double→triple by largest marginal gain in the
active objective, subject to product ≤ row budget.

Reduced systems are a constrained subset of arbitrary R-row portfolios;
unconstrained MAX_P13 top-R may achieve higher P(full-correct).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal, Sequence

from src.calc.stryktipset_optimizer.candidates import market_favorite
from src.calc.stryktipset_optimizer.coverage import (
    CoverageMetrics,
    compute_coverage,
    row_joint_probability,
)
from src.calc.stryktipset_optimizer.objectives import PredictionObjective
from src.calc.stryktipset_optimizer.prediction import MarketRow
from src.utils.common import OUTCOMES, Outcome

SignCount = Literal[1, 2, 3]


def _outcomes_ranked_by_pm(
    market_probs: dict[Outcome, float],
) -> tuple[Outcome, ...]:
    """Outcomes sorted by Pm desc; ties broken by OUTCOMES order."""
    return tuple(
        sorted(
            OUTCOMES,
            key=lambda outcome: (
                -float(market_probs[outcome]),
                OUTCOMES.index(outcome),
            ),
        )
    )


def _signs_for_match(
    ranked: Sequence[Outcome],
    sign_count: int,
) -> tuple[Outcome, ...]:
    if sign_count not in (1, 2, 3):
        raise ValueError(f"sign_count must be 1/2/3, got {sign_count}")
    return tuple(ranked[:sign_count])


def expand_system_rows(
    match_signs: Sequence[Sequence[Outcome]],
) -> list[tuple[Outcome, ...]]:
    """Cartesian product of per-match sign lists."""
    if not match_signs:
        raise ValueError("match_signs must be non-empty")
    return [tuple(row) for row in itertools.product(*match_signs)]


def system_product(sign_counts: Sequence[int]) -> int:
    product = 1
    for count in sign_counts:
        product *= int(count)
    return product


def _market_rows_from_outcomes(
    market_probs: Sequence[dict[Outcome, float]],
    rows: Sequence[Sequence[Outcome]],
) -> list[MarketRow]:
    results: list[MarketRow] = []
    for outcomes in rows:
        log_sum = 0.0
        favorite_count = 0
        home_count = 0
        draw_count = 0
        away_count = 0
        for match_index, outcome in enumerate(outcomes):
            log_sum += math.log(float(market_probs[match_index][outcome]))
            if outcome == market_favorite(market_probs[match_index]):
                favorite_count += 1
            if outcome == "1":
                home_count += 1
            elif outcome == "X":
                draw_count += 1
            else:
                away_count += 1
        results.append(
            MarketRow(
                outcomes=tuple(outcomes),
                log_pm_sum=log_sum,
                joint_probability=math.exp(log_sum),
                favorite_count=favorite_count,
                home_count=home_count,
                draw_count=draw_count,
                away_count=away_count,
            )
        )
    return results


def _objective_from_rows(
    market_probs: Sequence[dict[Outcome, float]],
    rows: Sequence[Sequence[Outcome]],
    objective: PredictionObjective,
) -> float:
    if objective == "MAX_P13":
        return sum(
            row_joint_probability(market_probs, row) for row in rows
        )
    coverage = compute_coverage(market_probs, rows)
    if objective == "MAX_P12_OR_BETTER":
        return coverage.p_12_or_better
    if objective == "MAX_P11_OR_BETTER":
        return coverage.p_11_or_better
    if objective == "MAX_EXPECTED_CORRECT":
        return coverage.expected_best_correct
    raise ValueError(f"unsupported objective: {objective}")


@dataclass(frozen=True)
class ReducedSystemResult:
    """Expanded reduced system with coverage and sign pattern."""

    sign_counts: tuple[int, ...]
    sign_pattern: tuple[tuple[Outcome, ...], ...]
    rows: tuple[MarketRow, ...]
    coverage: CoverageMetrics
    objective: PredictionObjective
    row_budget: int
    exact_selection: bool
    limitations: str


def build_reduced_system(
    market_probs: Sequence[dict[Outcome, float]],
    *,
    row_budget: int,
    objective: PredictionObjective = "MAX_P13",
) -> ReducedSystemResult:
    """Marginal-gain doubles/triples allocation under product ≤ row_budget.

    Starts from all-favorite singles. At each step considers every legal
    upgrade (1→2 or 2→3) that keeps product ≤ budget and picks the upgrade
    with largest objective gain. Stops when no upgrade fits.

    For MAX_P13 the gain is Δ Σ P(row) over newly added Cartesian rows
    (equivalent to full-system Δ since retained rows keep their mass).
    """
    if row_budget < 1:
        raise ValueError(f"row_budget must be >= 1, got {row_budget}")
    if not market_probs:
        raise ValueError("need at least one match")

    n_matches = len(market_probs)
    ranked = [_outcomes_ranked_by_pm(pm) for pm in market_probs]
    sign_counts = [1] * n_matches

    def current_signs() -> list[tuple[Outcome, ...]]:
        return [
            _signs_for_match(ranked[i], sign_counts[i])
            for i in range(n_matches)
        ]

    current_rows = expand_system_rows(current_signs())
    current_value = _objective_from_rows(market_probs, current_rows, objective)

    while True:
        best_match: int | None = None
        best_gain = float("-inf")
        best_rows: list[tuple[Outcome, ...]] | None = None
        best_value = current_value

        for match_index in range(n_matches):
            if sign_counts[match_index] >= 3:
                continue
            trial_counts = list(sign_counts)
            trial_counts[match_index] += 1
            if system_product(trial_counts) > row_budget:
                continue
            trial_signs = [
                _signs_for_match(ranked[i], trial_counts[i])
                for i in range(n_matches)
            ]
            trial_rows = expand_system_rows(trial_signs)
            trial_value = _objective_from_rows(
                market_probs, trial_rows, objective
            )
            gain = trial_value - current_value
            # Prefer larger gain; ties: lower match_index (deterministic).
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15 and best_match is None
            ):
                best_gain = gain
                best_match = match_index
                best_rows = trial_rows
                best_value = trial_value
            elif (
                abs(gain - best_gain) <= 1e-15
                and best_match is not None
                and match_index < best_match
            ):
                best_match = match_index
                best_rows = trial_rows
                best_value = trial_value

        if best_match is None or best_rows is None:
            break
        sign_counts[best_match] += 1
        current_rows = best_rows
        current_value = best_value

    sign_pattern = tuple(
        _signs_for_match(ranked[i], sign_counts[i]) for i in range(n_matches)
    )
    market_rows = _market_rows_from_outcomes(market_probs, current_rows)
    # Stable order: joint P desc, then OUTCOMES-lex of the row tuple.
    market_rows.sort(
        key=lambda row: (
            -row.joint_probability,
            row.outcomes,
        )
    )
    coverage = compute_coverage(
        market_probs, [row.outcomes for row in market_rows]
    )
    exact = objective == "MAX_P13"
    limitations = (
        "Reduced doubles/triples system: constrained Cartesian subset of "
        "arbitrary R-row portfolios; unconstrained MAX_P13 top-R may have "
        "higher P(full-correct). "
        + (
            "MAX_P13 system mass is exact for the chosen sign pattern."
            if exact
            else "Non-P13 objective used greedy upgrade steps (approximate)."
        )
    )
    return ReducedSystemResult(
        sign_counts=tuple(sign_counts),
        sign_pattern=sign_pattern,
        rows=tuple(market_rows),
        coverage=coverage,
        objective=objective,
        row_budget=row_budget,
        exact_selection=exact,
        limitations=limitations,
    )


def naive_closest_match_upgrades(
    market_probs: Sequence[dict[Outcome, float]],
    *,
    row_budget: int,
) -> tuple[int, ...]:
    """Naive baseline: always upgrade the match with closest top-2 Pm gap.

    Used in tests to show marginal-gain differs from \"always closest\".
    Closeness = Pm(1st) - Pm(2nd) (smaller ⇒ closer / more uncertain).
    """
    if row_budget < 1:
        raise ValueError(f"row_budget must be >= 1, got {row_budget}")
    n_matches = len(market_probs)
    ranked = [_outcomes_ranked_by_pm(pm) for pm in market_probs]
    sign_counts = [1] * n_matches

    def closeness(match_index: int) -> float:
        pm = market_probs[match_index]
        ordered = ranked[match_index]
        if sign_counts[match_index] == 1:
            return float(pm[ordered[0]]) - float(pm[ordered[1]])
        if sign_counts[match_index] == 2:
            return float(pm[ordered[1]]) - float(pm[ordered[2]])
        return float("inf")

    while True:
        candidates = [
            i
            for i in range(n_matches)
            if sign_counts[i] < 3
            and system_product(
                [sign_counts[j] + (1 if j == i else 0) for j in range(n_matches)]
            )
            <= row_budget
        ]
        if not candidates:
            break
        # Smallest closeness first; ties: lower match index.
        best = min(candidates, key=lambda i: (closeness(i), i))
        sign_counts[best] += 1

    return tuple(sign_counts)
