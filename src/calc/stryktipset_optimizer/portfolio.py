"""Greedy diversified portfolio selection via Hamming similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.calc.stryktipset_optimizer.candidates import CandidateRow
from src.utils.common import Outcome


@dataclass(frozen=True)
class PortfolioRow:
    """Selected row with diversity-adjusted score."""

    rank: int
    outcomes: tuple[Outcome, ...]
    row_score: float
    adjusted_score: float
    avg_hamming_similarity: float
    favorite_count: int
    home_count: int
    draw_count: int
    away_count: int


def hamming_similarity(
    left: Sequence[Outcome],
    right: Sequence[Outcome],
) -> float:
    """Fraction of identical outcomes: identical_count / N."""
    if len(left) != len(right):
        raise ValueError("rows must have the same length for Hamming similarity")
    if not left:
        raise ValueError("rows must be non-empty")
    identical = sum(1 for a, b in zip(left, right) if a == b)
    return identical / len(left)


def hamming_distance(
    left: Sequence[Outcome],
    right: Sequence[Outcome],
) -> int:
    """Number of differing positions."""
    if len(left) != len(right):
        raise ValueError("rows must have the same length for Hamming distance")
    return sum(1 for a, b in zip(left, right) if a != b)


def select_diversified_portfolio(
    candidates: Sequence[CandidateRow],
    *,
    row_count: int,
    lambda_diversity: float,
) -> list[PortfolioRow]:
    """Greedy: adjusted = row_score - lambda_div * avg_HamSim_to_selected.

    lambda_diversity=0 recovers pure top-K by row_score (stable order).
    """
    if row_count < 1:
        raise ValueError(f"row_count must be >= 1, got {row_count}")
    if lambda_diversity < 0:
        raise ValueError(
            f"lambda_diversity must be >= 0, got {lambda_diversity}"
        )
    if not candidates:
        raise ValueError("candidates must be non-empty")

    take = min(row_count, len(candidates))
    selected: list[PortfolioRow] = []
    remaining = list(candidates)

    for rank in range(1, take + 1):
        best_index = -1
        best_adjusted = float("-inf")
        best_avg_sim = 0.0

        for index, candidate in enumerate(remaining):
            if not selected:
                avg_sim = 0.0
            else:
                sims = [
                    hamming_similarity(candidate.outcomes, row.outcomes)
                    for row in selected
                ]
                avg_sim = sum(sims) / len(sims)
            adjusted = candidate.row_score - lambda_diversity * avg_sim
            # Prefer higher adjusted; ties keep earlier (higher raw) candidate.
            if adjusted > best_adjusted + 1e-15 or (
                abs(adjusted - best_adjusted) <= 1e-15 and best_index < 0
            ):
                best_adjusted = adjusted
                best_index = index
                best_avg_sim = avg_sim
            elif abs(adjusted - best_adjusted) <= 1e-15 and index < best_index:
                best_index = index
                best_avg_sim = avg_sim

        chosen = remaining.pop(best_index)
        selected.append(
            PortfolioRow(
                rank=rank,
                outcomes=chosen.outcomes,
                row_score=chosen.row_score,
                adjusted_score=best_adjusted,
                avg_hamming_similarity=best_avg_sim,
                favorite_count=chosen.favorite_count,
                home_count=chosen.home_count,
                draw_count=chosen.draw_count,
                away_count=chosen.away_count,
            )
        )

    return selected
