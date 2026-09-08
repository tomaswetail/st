"""Exact system coverage metrics under independent market measure.

For selected row set S and world ω ∈ {1,X,2}^{N}:

    best_correct(ω) = max_{r∈S} #{j : r_j = ω_j}

Enumerate all 3^N worlds (OK for N≤13 ≈ 1.6e6) and return
P(best_correct=k), cumulatives, and E[best_correct].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.utils.common import OUTCOMES, Outcome

OUTCOME_INDEX: dict[Outcome, int] = {"1": 0, "X": 1, "2": 2}


@dataclass(frozen=True)
class CoverageMetrics:
    """Exact coverage distribution for a selected portfolio."""

    n_matches: int
    p_best_correct: tuple[float, ...]  # length N+1; index k = P(best=k)
    p_full: float  # P(best_correct = N) aka P(13) when N=13
    p_12_or_better: float
    p_11_or_better: float
    p_10_or_better: float
    expected_best_correct: float

    @property
    def p_13(self) -> float:
        """Alias for full-correct probability when N=13; else P(best=N)."""
        return self.p_full


def _pm_table(
    market_probs: Sequence[dict[Outcome, float]],
) -> np.ndarray:
    n_matches = len(market_probs)
    table = np.empty((n_matches, 3), dtype=np.float64)
    for match_index, pm in enumerate(market_probs):
        for outcome in OUTCOMES:
            value = float(pm[outcome])
            if value <= 0:
                raise ValueError(
                    f"match {match_index} outcome {outcome}: "
                    f"Pm must be > 0 (got {value})"
                )
            table[match_index, OUTCOME_INDEX[outcome]] = value
    return table


def _rows_to_indices(
    rows: Sequence[Sequence[Outcome]],
    n_matches: int,
) -> np.ndarray:
    if not rows:
        raise ValueError("rows must be non-empty for coverage metrics")
    encoded = np.empty((len(rows), n_matches), dtype=np.int8)
    for row_index, row in enumerate(rows):
        if len(row) != n_matches:
            raise ValueError(
                f"row {row_index} length {len(row)} != n_matches {n_matches}"
            )
        for match_index, outcome in enumerate(row):
            encoded[row_index, match_index] = OUTCOME_INDEX[outcome]
    return encoded


def compute_coverage(
    market_probs: Sequence[dict[Outcome, float]],
    rows: Sequence[Sequence[Outcome]],
) -> CoverageMetrics:
    """Exact P(best_correct=k) via full 3^N enumeration (vectorized)."""
    if not market_probs:
        raise ValueError("need at least one match")
    n_matches = len(market_probs)
    pm = _pm_table(market_probs)
    row_idx = _rows_to_indices(rows, n_matches)
    n_worlds = 3**n_matches

    # worlds[w, j] = outcome index for match j in world w (base-3 digits)
    world_ids = np.arange(n_worlds, dtype=np.int32)
    worlds = np.empty((n_worlds, n_matches), dtype=np.int8)
    tmp = world_ids.copy()
    for match_index in range(n_matches - 1, -1, -1):
        worlds[:, match_index] = (tmp % 3).astype(np.int8)
        tmp //= 3

    # world probability = ∏_j Pm_j(ω_j)
    log_pm = np.log(pm)
    log_world = np.zeros(n_worlds, dtype=np.float64)
    for match_index in range(n_matches):
        log_world += log_pm[match_index, worlds[:, match_index]]
    world_prob = np.exp(log_world)

    # agreements[w, r] = #{j : row_r[j] == world[w,j]}
    # Compute in chunks if many rows to limit peak memory.
    best_correct = np.zeros(n_worlds, dtype=np.int16)
    n_rows = int(row_idx.shape[0])
    chunk = max(1, min(n_rows, 64))
    for start in range(0, n_rows, chunk):
        block = row_idx[start : start + chunk]
        # (W, B, N) equality → sum over N
        agreements = (worlds[:, None, :] == block[None, :, :]).sum(axis=2)
        best_correct = np.maximum(best_correct, agreements.max(axis=1))

    hist = np.zeros(n_matches + 1, dtype=np.float64)
    for k in range(n_matches + 1):
        hist[k] = float(world_prob[best_correct == k].sum())

    # Numerical cleanup: tiny drift from float product
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum

    cumul_from = np.cumsum(hist[::-1])[::-1]  # P(best >= k) at index k
    expected = float(np.dot(np.arange(n_matches + 1), hist))

    def cum_at_least(threshold: int) -> float:
        if threshold > n_matches:
            return 0.0
        if threshold < 0:
            return 1.0
        return float(cumul_from[threshold])

    return CoverageMetrics(
        n_matches=n_matches,
        p_best_correct=tuple(float(x) for x in hist),
        p_full=float(hist[n_matches]),
        p_12_or_better=cum_at_least(12),
        p_11_or_better=cum_at_least(11),
        p_10_or_better=cum_at_least(10),
        expected_best_correct=expected,
    )


def row_joint_probability(
    market_probs: Sequence[dict[Outcome, float]],
    outcomes: Sequence[Outcome],
) -> float:
    """P(row) = ∏_j Pm_j(row_j)."""
    if len(outcomes) != len(market_probs):
        raise ValueError("outcomes length must equal number of matches")
    log_p = 0.0
    for match_index, outcome in enumerate(outcomes):
        pm = float(market_probs[match_index][outcome])
        if pm <= 0:
            raise ValueError(
                f"match {match_index} outcome {outcome}: Pm must be > 0"
            )
        log_p += float(np.log(pm))
    return float(np.exp(log_p))
