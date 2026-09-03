"""Rest, fixture congestion, and fatigue features for residual ML."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RestCongestionFeatures:
    """League-agnostic rest and congestion context for one fixture.

    Missing previous match: rest_days is None (unknown/neutral; no invented
    day count). Derived short_rest and extra-time interactions treat missing
    rest as 0. rest_day_difference is None if either side is missing.
    Squad/lineup fields stay None when that data is not available.
    """

    home_rest_days: int | None
    away_rest_days: int | None
    rest_day_difference: int | None
    home_matches_last_14_days: int
    away_matches_last_14_days: int
    home_short_rest: int
    away_short_rest: int
    home_congestion: int
    away_congestion: int
    home_extra_time_in_previous_match: bool
    away_extra_time_in_previous_match: bool
    home_extra_time_short_rest: int
    away_extra_time_short_rest: int
    extra_time_x_short_rest: int
    home_lineup_changes: int | None
    away_lineup_changes: int | None
    congestion_x_squad_depth: float | None
    short_rest_x_rotation: float | None
