"""Flat feature vector for the residual 1X2 ML model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date
from typing import Any


@dataclass
class ResidualMLFeatures:
    """Model-ready match features assembled from calc modules."""

    match_id: int
    draw_number: int | None
    feature_cutoff_date: date

    p_home_market: float | None
    p_draw_market: float | None
    p_away_market: float | None

    home_npxg_for: float | None
    home_npxg_against: float | None
    away_npxg_for: float | None
    away_npxg_against: float | None
    home_opponent_adjusted_attack: float | None
    home_opponent_adjusted_defence: float | None
    away_opponent_adjusted_attack: float | None
    away_opponent_adjusted_defence: float | None
    home_shot_quality_for: float | None
    home_shot_quality_conceded: float | None
    away_shot_quality_for: float | None
    away_shot_quality_conceded: float | None
    home_set_piece_attack: float | None
    home_set_piece_defence: float | None
    away_set_piece_attack: float | None
    away_set_piece_defence: float | None
    home_goalkeeper_prevention: float | None
    away_goalkeeper_prevention: float | None

    expected_home_goals: float | None
    expected_away_goals: float | None
    p_home_dc: float | None
    p_draw_dc: float | None
    p_away_dc: float | None

    market_vs_dc_home: float | None
    market_vs_dc_draw: float | None
    market_vs_dc_away: float | None

    attack_strength_difference: float | None
    expected_goal_difference: float | None
    expected_goal_total: float | None
    market_balance: float | None
    defence_strength_difference: float | None
    favourite_strength: float | None
    combined_draw_rate: float | None
    combined_one_goal_match_rate: float | None
    combined_close_match_rate: float | None
    combined_low_scoring_rate: float | None

    league_draw_rate: float | None
    league_home_win_rate: float | None
    league_away_win_rate: float | None
    league_avg_goals: float | None
    league_goal_std: float | None
    league_favourite_win_rate: float | None
    league_competitive_balance: float | None
    league_avg_npxg: float | None
    league_upset_rate: float | None
    league_prior_weight: float | None

    home_rest_days: int | None
    away_rest_days: int | None
    rest_day_difference: int | None
    home_matches_last_14_days: int | None
    away_matches_last_14_days: int | None
    home_short_rest: int | None
    away_short_rest: int | None
    home_congestion: int | None
    away_congestion: int | None
    congestion_difference: int | None
    home_extra_time_in_previous_match: int | None
    away_extra_time_in_previous_match: int | None
    extra_time_x_short_rest: int | None
    fatigue_difference: int | None

    home_advantage_log: float | None
    home_advantage_coefficient: float | None

    travel_distance_km: float | None = None

    # Player availability / injury (None when no pre-cutoff snapshot)
    home_missing_player_value: float | None = None
    away_missing_player_value: float | None = None
    missing_value_difference: float | None = None
    home_unavailable_count: int | None = None
    away_unavailable_count: int | None = None
    home_lineup_changes: int | None = None
    away_lineup_changes: int | None = None
    missing_value_x_favourite: float | None = None
    short_rest_x_missing_value: float | None = None
    congestion_x_squad_depth: float | None = None
    short_rest_x_rotation: float | None = None
    has_availability: int = 0

    _METADATA_FIELDS = frozenset(
        {"match_id", "draw_number", "feature_cutoff_date"}
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_feature_names(self) -> list[str]:
        return [
            field.name
            for field in fields(self)
            if field.name not in self._METADATA_FIELDS
        ]

    def model_feature_vector(self) -> dict[str, float | int | None]:
        return {
            name: getattr(self, name)
            for name in self.model_feature_names()
        }
