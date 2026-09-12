"""Vig-free market probabilities and shape features from decimal 1X2 odds."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.common import Outcome


@dataclass
class MarketProbabilityBreakdown:
    """Raw implied, vig-free, overround, and market-shape features for one 1X2 triple.

    Entropy uses the natural log. Probabilities of 0 are clipped to 1e-15 for
    the entropy sum only; vig-free ``p_*`` fields stay unclipped after
    normalization.
    """

    odds_home: float
    odds_draw: float
    odds_away: float
    implied_home: float
    implied_draw: float
    implied_away: float
    p_home: float
    p_draw: float
    p_away: float
    overround: float
    overround_below_one: bool
    market_top_probability: float
    market_second_probability: float
    market_probability_gap: float
    market_entropy: float
    bookmaker: str | None = None
    price_type: str | None = None

    def as_probs(self) -> dict[Outcome, float]:
        return {"1": self.p_home, "X": self.p_draw, "2": self.p_away}
