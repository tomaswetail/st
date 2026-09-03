"""League-level behaviour features for residual ML context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeagueBehaviorFeatures:
    league_draw_rate: float | None
    league_home_win_rate: float | None
    league_away_win_rate: float | None
    league_avg_goals: float | None
    league_goal_std: float | None
    league_favourite_win_rate: float | None
    league_competitive_balance: float | None
    league_promoted_team_effect: float
    league_sample_size: int
    league_data_quality: float
    league_prior_weight: float
