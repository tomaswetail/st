"""Exhaustive top-C coupon row search via heap.

For N=13 (3^13 ≈ 1.6e6) a full enumeration is fine. Precompute per-match
score contributions ``a_j(i) = log Pm_j(i) - beta * log Pp_j(i)`` then
``row_score = sum_j a_j(sel_j)``.

Approximate runtime (typical laptop, 13 matches, recursive enumeration):
- candidate_count=500: ~0.2–0.6 s
- candidate_count=50000: ~0.3–0.9 s
(enumeration dominates; heap cost grows slowly with C)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.utils.common import OUTCOMES, Outcome

OUTCOME_INDEX: dict[Outcome, int] = {"1": 0, "X": 1, "2": 2}
INDEX_OUTCOME: tuple[Outcome, ...] = OUTCOMES


def market_favorite(market_probs: dict[Outcome, float]) -> Outcome:
    """argmax_i Pm(i); ties broken by OUTCOMES order (\"1\", \"X\", \"2\")."""
    best_outcome: Outcome = OUTCOMES[0]
    best_prob = float(market_probs[best_outcome])
    for outcome in OUTCOMES[1:]:
        prob = float(market_probs[outcome])
        if prob > best_prob:
            best_outcome = outcome
            best_prob = prob
    return best_outcome


@dataclass(frozen=True)
class CandidateRow:
    """One coupon row among the top-C candidates."""

    outcomes: tuple[Outcome, ...]
    row_score: float
    log_pm_sum: float
    log_pp_sum: float
    favorite_count: int  # selections equal to market argmax Pm
    home_count: int
    draw_count: int
    away_count: int


def build_score_table(
    market_probs: Sequence[dict[Outcome, float]],
    public_probs: Sequence[dict[Outcome, float]],
    beta: float,
) -> np.ndarray:
    """Return shape (n_matches, 3) of per-outcome contributions a_j(i)."""
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if len(market_probs) != len(public_probs):
        raise ValueError("market_probs and public_probs length mismatch")

    n_matches = len(market_probs)
    table = np.empty((n_matches, 3), dtype=np.float64)
    for match_index, (pm, pp) in enumerate(zip(market_probs, public_probs)):
        for outcome in OUTCOMES:
            pm_i = float(pm[outcome])
            pp_i = float(pp[outcome])
            if pm_i <= 0 or pp_i <= 0:
                raise ValueError(
                    f"match {match_index} outcome {outcome}: "
                    f"Pm and Pp must be > 0 (got Pm={pm_i}, Pp={pp_i})"
                )
            table[match_index, OUTCOME_INDEX[outcome]] = (
                math.log(pm_i) - beta * math.log(pp_i)
            )
    return table


def row_score_from_selections(
    score_table: np.ndarray,
    selections: Sequence[int],
) -> float:
    """Sum of a_j(sel_j) for integer outcome indices."""
    if len(selections) != score_table.shape[0]:
        raise ValueError("selections length must equal number of matches")
    total = 0.0
    for match_index, outcome_index in enumerate(selections):
        total += float(score_table[match_index, outcome_index])
    return total


def _row_diagnostics(
    outcomes: tuple[Outcome, ...],
    market_probs: Sequence[dict[Outcome, float]],
    public_probs: Sequence[dict[Outcome, float]],
    row_score: float,
) -> CandidateRow:
    log_pm = 0.0
    log_pp = 0.0
    favorite_count = 0
    home_count = 0
    draw_count = 0
    away_count = 0
    for match_index, outcome in enumerate(outcomes):
        log_pm += math.log(float(market_probs[match_index][outcome]))
        log_pp += math.log(float(public_probs[match_index][outcome]))
        if outcome == market_favorite(market_probs[match_index]):
            favorite_count += 1
        if outcome == "1":
            home_count += 1
        elif outcome == "X":
            draw_count += 1
        else:
            away_count += 1
    return CandidateRow(
        outcomes=outcomes,
        row_score=row_score,
        log_pm_sum=log_pm,
        log_pp_sum=log_pp,
        favorite_count=favorite_count,
        home_count=home_count,
        draw_count=draw_count,
        away_count=away_count,
    )


def top_candidates(
    market_probs: Sequence[dict[Outcome, float]],
    public_probs: Sequence[dict[Outcome, float]],
    *,
    beta: float,
    candidate_count: int,
) -> list[CandidateRow]:
    """Exhaustive search; keep top ``candidate_count`` by row_score.

    Recursive DFS accumulates scores without allocating a tuple per leaf until
    the row enters the top-C heap. Heap is a min-heap of size C.
    """
    if candidate_count < 1:
        raise ValueError(f"candidate_count must be >= 1, got {candidate_count}")
    if len(market_probs) != len(public_probs):
        raise ValueError("market_probs and public_probs length mismatch")
    if not market_probs:
        raise ValueError("need at least one match")

    score_table = build_score_table(market_probs, public_probs, beta)
    n_matches = int(score_table.shape[0])
    # Plain nested lists for fast C-level-ish indexing in tight loop.
    scores = score_table.tolist()
    path = [0] * n_matches
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    tie_breaker = 0

    def consider(score: float) -> None:
        nonlocal tie_breaker
        selections = tuple(path)
        if len(heap) < candidate_count:
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

    ranked = sorted(heap, key=lambda item: (-item[0], item[1]))
    results: list[CandidateRow] = []
    for score, _tb, selections in ranked:
        outcomes = tuple(INDEX_OUTCOME[index] for index in selections)
        results.append(
            _row_diagnostics(outcomes, market_probs, public_probs, score)
        )
    return results
