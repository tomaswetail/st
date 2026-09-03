"""Match balance and expected-goal environment features for draw modelling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BalanceAndEnvironmentFeatures:
    attack_strength_difference: float | None
    expected_goal_difference: float | None
    expected_goal_total: float | None
    market_balance: float | None
    defence_strength_difference: float | None
    favourite_strength: float | None
    home_recent_draw_rate: float | None
    away_recent_draw_rate: float | None
    home_one_goal_match_rate: float | None
    away_one_goal_match_rate: float | None
    home_close_match_rate: float | None
    away_close_match_rate: float | None
    home_low_scoring_rate: float | None
    away_low_scoring_rate: float | None
    combined_draw_rate: float | None
    combined_one_goal_match_rate: float | None
    combined_close_match_rate: float | None
    combined_low_scoring_rate: float | None
    home_recent_sample_size: int = 0
    away_recent_sample_size: int = 0
