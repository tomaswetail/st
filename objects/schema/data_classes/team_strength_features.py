"""Team strength features derived from stored match advanced stats / shots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class TeamStrengthFeatures:
    team_id: int
    before: datetime
    venue: Literal["home", "away"] | None
    lookback_matches: int
    sample_size: int
    non_penalty_xg_for: float | None
    non_penalty_xg_against: float | None
    average_shot_xg_for: float | None
    average_shot_xg_against: float | None
    home_attack_strength: float | None
    home_defence_strength: float | None
    away_attack_strength: float | None
    away_defence_strength: float | None
    recency_weighted_attack_rating: float | None
    recency_weighted_defence_rating: float | None
    set_piece_xg_for: float | None
    set_piece_xg_against: float | None
    goalkeeper_goals_prevented: float | None
