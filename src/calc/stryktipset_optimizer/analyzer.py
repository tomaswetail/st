"""Banker diagnostic and match-level market-vs-public divergence ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.calc.stryktipset_optimizer.value import log_leverage, value_ratio
from src.utils.common import OUTCOMES, Outcome


@dataclass(frozen=True)
class BankerSuggestion:
    """Diagnostic-only banker pick (not used in portfolio selection)."""

    match_index: int
    outcome: Outcome
    score: float
    log_pm: float
    log_leverage: float


@dataclass(frozen=True)
class MatchAnalysis:
    """Per-match divergence and value diagnostics."""

    match_index: int
    match_label: str | None
    jensen_shannon: float
    max_abs_log_leverage: float
    highest_value_outcome: Outcome
    highest_value_ratio: float
    market_probs: dict[Outcome, float]
    public_probs: dict[Outcome, float]
    value_ratios: dict[Outcome, float]
    log_leverages: dict[Outcome, float]
    banker_scores: dict[Outcome, float]


def _kl_divergence(
    p: Mapping[str, float],
    q: Mapping[str, float],
) -> float:
    total = 0.0
    for outcome in OUTCOMES:
        p_i = float(p[outcome])
        q_i = float(q[outcome])
        if p_i <= 0:
            continue
        if q_i <= 0:
            raise ValueError("q must be > 0 for KL divergence")
        total += p_i * math.log(p_i / q_i)
    return total


def jensen_shannon_divergence(
    pm: Mapping[str, float],
    pp: Mapping[str, float],
) -> float:
    """Jensen–Shannon divergence in nats between Pm and Pp."""
    mixture = {
        outcome: 0.5 * (float(pm[outcome]) + float(pp[outcome]))
        for outcome in OUTCOMES
    }
    return 0.5 * _kl_divergence(pm, mixture) + 0.5 * _kl_divergence(pp, mixture)


def banker_score(
    pm: float,
    pp: float,
    *,
    banker_value_weight: float,
) -> float:
    """log(Pm) + banker_value_weight * log(Pm/Pp) — diagnostic only."""
    if pm <= 0 or pp <= 0:
        raise ValueError("Pm and Pp must be > 0 for banker_score")
    leverage = log_leverage(pm, pp)
    return math.log(pm) + banker_value_weight * leverage


def analyze_matches(
    market_probs: Sequence[dict[Outcome, float]],
    public_probs: Sequence[dict[Outcome, float]],
    *,
    banker_value_weight: float = 1.0,
    match_labels: Sequence[str | None] | None = None,
) -> tuple[list[MatchAnalysis], list[BankerSuggestion]]:
    """Rank matches by JS divergence; also compute banker suggestions."""
    if len(market_probs) != len(public_probs):
        raise ValueError("market_probs and public_probs length mismatch")

    analyses: list[MatchAnalysis] = []
    banker_flat: list[BankerSuggestion] = []

    for match_index, (pm, pp) in enumerate(zip(market_probs, public_probs)):
        ratios: dict[Outcome, float] = {}
        levers: dict[Outcome, float] = {}
        banker_scores: dict[Outcome, float] = {}
        best_outcome: Outcome = "1"
        best_ratio = float("-inf")
        max_abs_lev = 0.0

        for outcome in OUTCOMES:
            pm_i = float(pm[outcome])
            pp_i = float(pp[outcome])
            ratio = value_ratio(pm_i, pp_i)
            lev = log_leverage(pm_i, pp_i)
            score = banker_score(
                pm_i, pp_i, banker_value_weight=banker_value_weight
            )
            ratios[outcome] = ratio
            levers[outcome] = lev
            banker_scores[outcome] = score
            max_abs_lev = max(max_abs_lev, abs(lev))
            if ratio > best_ratio:
                best_ratio = ratio
                best_outcome = outcome
            banker_flat.append(
                BankerSuggestion(
                    match_index=match_index,
                    outcome=outcome,
                    score=score,
                    log_pm=math.log(pm_i),
                    log_leverage=lev,
                )
            )

        label = None
        if match_labels is not None and match_index < len(match_labels):
            label = match_labels[match_index]

        analyses.append(
            MatchAnalysis(
                match_index=match_index,
                match_label=label,
                jensen_shannon=jensen_shannon_divergence(pm, pp),
                max_abs_log_leverage=max_abs_lev,
                highest_value_outcome=best_outcome,
                highest_value_ratio=best_ratio,
                market_probs={o: float(pm[o]) for o in OUTCOMES},
                public_probs={o: float(pp[o]) for o in OUTCOMES},
                value_ratios=ratios,
                log_leverages=levers,
                banker_scores=banker_scores,
            )
        )

    analyses_sorted = sorted(
        analyses,
        key=lambda item: (-item.jensen_shannon, item.match_index),
    )
    bankers_sorted = sorted(
        banker_flat,
        key=lambda item: (-item.score, item.match_index, item.outcome),
    )
    return analyses_sorted, bankers_sorted
