"""Market-only top-R row search for PREDICTION / MAX_P13.

Objective: maximize Σ_{r∈S} P(r) where P(r)=∏_j Pm_j(r_j).
Under independence this equals P(at least one fully-correct row).

Exact algorithm: DFS/heap over log-sum contributions a_j(i)=log Pm_j(i)
(same pattern as candidates.py VALUE heap). Selected portfolio IS the top-R
heap result — no Hamming diversity, no λ_diversity.

Ties: when scores equal, earlier DFS enumeration order wins (OUTCOMES order
\"1\",\"X\",\"2\" at each match depth; stable via monotonic tie_breaker).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.calc.stryktipset_optimizer.candidates import market_favorite
from src.utils.common import OUTCOMES, Outcome

OUTCOME_INDEX: dict[Outcome, int] = {"1": 0, "X": 1, "2": 2}
INDEX_OUTCOME: tuple[Outcome, ...] = OUTCOMES


@dataclass(frozen=True)
class MarketRow:
    """One market-probability-ranked coupon row."""

    outcomes: tuple[Outcome, ...]
    log_pm_sum: float
    joint_probability: float
    favorite_count: int
    home_count: int
    draw_count: int
    away_count: int


def build_log_pm_table(
    market_probs: Sequence[dict[Outcome, float]],
) -> np.ndarray:
    """Return shape (n_matches, 3) of log Pm_j(i)."""
    n_matches = len(market_probs)
    if n_matches < 1:
        raise ValueError("need at least one match")
    table = np.empty((n_matches, 3), dtype=np.float64)
    for match_index, pm in enumerate(market_probs):
        for outcome in OUTCOMES:
            pm_i = float(pm[outcome])
            if pm_i <= 0:
                raise ValueError(
                    f"match {match_index} outcome {outcome}: "
                    f"Pm must be > 0 (got {pm_i})"
                )
            table[match_index, OUTCOME_INDEX[outcome]] = math.log(pm_i)
    return table


def _row_diagnostics(
    outcomes: tuple[Outcome, ...],
    market_probs: Sequence[dict[Outcome, float]],
    log_pm_sum: float,
) -> MarketRow:
    favorite_count = 0
    home_count = 0
    draw_count = 0
    away_count = 0
    for match_index, outcome in enumerate(outcomes):
        if outcome == market_favorite(market_probs[match_index]):
            favorite_count += 1
        if outcome == "1":
            home_count += 1
        elif outcome == "X":
            draw_count += 1
        else:
            away_count += 1
    return MarketRow(
        outcomes=outcomes,
        log_pm_sum=log_pm_sum,
        joint_probability=math.exp(log_pm_sum),
        favorite_count=favorite_count,
        home_count=home_count,
        draw_count=draw_count,
        away_count=away_count,
    )


def top_market_rows(
    market_probs: Sequence[dict[Outcome, float]],
    *,
    row_count: int,
) -> list[MarketRow]:
    """Exact top-R rows by joint market P via DFS min-heap (no full 3^N list).

    Does not materialize all atoms into a Python list; only the heap of size R
    is retained. Ranking: higher log_pm_sum first; ties broken by earlier
    enumeration order (OUTCOMES lex at each depth).
    """
    if row_count < 1:
        raise ValueError(f"row_count must be >= 1, got {row_count}")
    if not market_probs:
        raise ValueError("need at least one match")

    log_table = build_log_pm_table(market_probs)
    n_matches = int(log_table.shape[0])
    scores = log_table.tolist()
    path = [0] * n_matches
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    tie_breaker = 0

    def consider(score: float) -> None:
        nonlocal tie_breaker
        selections = tuple(path)
        if len(heap) < row_count:
            heapq.heappush(heap, (score, tie_breaker, selections))
            tie_breaker += 1
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, tie_breaker, selections))
            tie_breaker += 1

    def dfs(match_index: int, running: float) -> None:
        if match_index == n_matches:
            consider(running)
            return
        row_scores = scores[match_index]
        for outcome_index in (0, 1, 2):
            path[match_index] = outcome_index
            dfs(match_index + 1, running + row_scores[outcome_index])

    dfs(0, 0.0)

    # Higher score first; among equals, smaller tie_breaker (earlier DFS) wins.
    ranked = sorted(heap, key=lambda item: (-item[0], item[1]))
    results: list[MarketRow] = []
    for score, _tb, selections in ranked:
        outcomes = tuple(INDEX_OUTCOME[index] for index in selections)
        results.append(_row_diagnostics(outcomes, market_probs, score))
    return results


def single_favorite_row(
    market_probs: Sequence[dict[Outcome, float]],
) -> MarketRow:
    """All-favorite singles row (argmax Pm per match; OUTCOMES tie-break)."""
    outcomes = tuple(market_favorite(pm) for pm in market_probs)
    log_sum = 0.0
    for match_index, outcome in enumerate(outcomes):
        log_sum += math.log(float(market_probs[match_index][outcome]))
    return _row_diagnostics(outcomes, market_probs, log_sum)
