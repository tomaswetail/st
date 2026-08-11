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
    opponent_adjusted_attack_strength: float | None = None
    opponent_adjusted_defence_strength: float | None = None
    set_piece_attack_strength: float | None = None
    set_piece_defence_strength: float | None = None
    goalkeeper_prevention_rating: float | None = None
    goalkeeper_goals_prevented: float | None = None
    has_xg_data: bool = False
    has_xgot_data: bool = False
    has_set_piece_data: bool = False


@dataclass
class MatchStrengthFeatures:
    match_id: int
    home_team_id: int | None
    away_team_id: int | None
    home: TeamStrengthFeatures | None
    away: TeamStrengthFeatures | None
    home_npxg_for: float | None = None
    home_npxg_against: float | None = None
    away_npxg_for: float | None = None
    away_npxg_against: float | None = None
    home_shot_quality_for: float | None = None
    home_shot_quality_conceded: float | None = None
    away_shot_quality_for: float | None = None
    away_shot_quality_conceded: float | None = None
    home_attack_strength: float | None = None
    home_defence_strength: float | None = None
    away_attack_strength: float | None = None
    away_defence_strength: float | None = None
    home_opponent_adjusted_attack: float | None = None
    home_opponent_adjusted_defence: float | None = None
    away_opponent_adjusted_attack: float | None = None
    away_opponent_adjusted_defence: float | None = None
    home_set_piece_attack: float | None = None
    home_set_piece_defence: float | None = None
    away_set_piece_attack: float | None = None
    away_set_piece_defence: float | None = None
    home_goalkeeper_prevention: float | None = None
    away_goalkeeper_prevention: float | None = None
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    dixon_coles_home_probability: float | None = None
    dixon_coles_draw_probability: float | None = None
    dixon_coles_away_probability: float | None = None
