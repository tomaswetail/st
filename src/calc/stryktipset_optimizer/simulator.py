"""Monte Carlo pool simulation — relative tier metrics only (no SEK).

Truth outcomes are sampled from MARKET probabilities (Pm).
Synthetic public single rows are sampled independently from Pp.
Tier fields correct_13/12/11/10 mean top-4 hit tiers relative to coupon size N
(N / N-1 / N-2 / N-3); for classic Stryktipset N=13 these are absolute 13–10.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.calc.stryktipset_optimizer.data import LEAKAGE_LIMITATIONS
from src.objects.schema.data_classes.stryktipset_optimizer import (
    SimulationMetricsDTO,
    SimulationTierCounts,
)
from src.utils.common import OUTCOMES, Outcome

OUTCOME_INDEX = {"1": 0, "X": 1, "2": 2}


def _sample_outcomes(
    rng: np.random.Generator,
    probs: Sequence[dict[Outcome, float]],
) -> tuple[Outcome, ...]:
    draws: list[Outcome] = []
    for match_probs in probs:
        p = [float(match_probs[outcome]) for outcome in OUTCOMES]
        index = int(rng.choice(3, p=p))
        draws.append(OUTCOMES[index])
    return tuple(draws)


def _correct_count(
    row: Sequence[Outcome],
    truth: Sequence[Outcome],
) -> int:
    return sum(1 for a, b in zip(row, truth) if a == b)


def _tier_bucket(correct: int, coupon_size: int) -> str | None:
    """Map correct count to top-4 tiers relative to coupon size."""
    if coupon_size < 1:
        return None
    if correct >= coupon_size:
        return "correct_13"
    if correct == coupon_size - 1:
        return "correct_12"
    if correct == coupon_size - 2:
        return "correct_11"
    if correct == coupon_size - 3:
        return "correct_10"
    return None


def _bump_tier(tiers: SimulationTierCounts, bucket: str | None) -> None:
    if bucket == "correct_13":
        tiers.correct_13 += 1
    elif bucket == "correct_12":
        tiers.correct_12 += 1
    elif bucket == "correct_11":
        tiers.correct_11 += 1
    elif bucket == "correct_10":
        tiers.correct_10 += 1


DEFAULT_PUBLIC_ROW_COUNT = 1000


def simulate_pool(
    market_probs: Sequence[dict[Outcome, float]],
    public_probs: Sequence[dict[Outcome, float]],
    our_rows: Sequence[Sequence[Outcome]],
    *,
    n_simulations: int = 1000,
    seed: int = 0,
    public_row_count: int | None = None,
) -> SimulationMetricsDTO:
    """Run seeded MC; return relative tier rates and lifts (no SEK).

    For each simulation:
    - sample truth from Pm
    - score each of our portfolio rows
    - sample ``public_row_count`` independent synthetic public rows from Pp
      (default: ``DEFAULT_PUBLIC_ROW_COUNT`` = 1000) and score them
    """
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
    if not our_rows:
        raise ValueError("our_rows must be non-empty")
    if len(market_probs) != len(public_probs):
        raise ValueError("market_probs and public_probs length mismatch")

    coupon_size = len(market_probs)
    n_public = (
        public_row_count
        if public_row_count is not None
        else DEFAULT_PUBLIC_ROW_COUNT
    )
    if n_public < 1:
        raise ValueError(f"public_row_count must be >= 1, got {n_public}")

    rng = np.random.default_rng(seed)
    our_tiers = SimulationTierCounts()
    public_tiers = SimulationTierCounts()
    our_correct_sum = 0.0
    public_correct_sum = 0.0
    our_row_evals = 0
    public_row_evals = 0

    for _ in range(n_simulations):
        truth = _sample_outcomes(rng, market_probs)

        best_ours = 0
        for row in our_rows:
            correct = _correct_count(row, truth)
            best_ours = max(best_ours, correct)
            our_correct_sum += correct
            our_row_evals += 1
        _bump_tier(our_tiers, _tier_bucket(best_ours, coupon_size))

        best_public = 0
        for _p in range(n_public):
            public_row = _sample_outcomes(rng, public_probs)
            correct = _correct_count(public_row, truth)
            best_public = max(best_public, correct)
            public_correct_sum += correct
            public_row_evals += 1
        _bump_tier(public_tiers, _tier_bucket(best_public, coupon_size))

    def _rates(tiers: SimulationTierCounts) -> dict[str, float]:
        n = float(n_simulations)
        return {
            "13": tiers.correct_13 / n,
            "12": tiers.correct_12 / n,
            "11": tiers.correct_11 / n,
            "10": tiers.correct_10 / n,
        }

    our_rates = _rates(our_tiers)
    public_rates = _rates(public_tiers)
    lifts = {
        key: (
            our_rates[key] / public_rates[key]
            if public_rates[key] > 0
            else float("inf") if our_rates[key] > 0 else 1.0
        )
        for key in our_rates
    }

    return SimulationMetricsDTO(
        n_simulations=n_simulations,
        seed=seed,
        our_tiers=our_tiers,
        public_tiers=public_tiers,
        our_tier_rates=our_rates,
        public_tier_rates=public_rates,
        relative_tier_lifts=lifts,
        mean_correct_ours=our_correct_sum / max(our_row_evals, 1),
        mean_correct_public=public_correct_sum / max(public_row_evals, 1),
        limitations=LEAKAGE_LIMITATIONS,
    )
