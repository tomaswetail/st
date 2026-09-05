"""Injury / availability columns for residual ML (Phase 2)."""

from __future__ import annotations

INJURY_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "home_missing_player_value",
        "away_missing_player_value",
        "missing_value_difference",
        "home_unavailable_count",
        "away_unavailable_count",
        "home_lineup_changes",
        "away_lineup_changes",
        "missing_value_x_favourite",
        "short_rest_x_missing_value",
        "congestion_x_squad_depth",
        "short_rest_x_rotation",
        "has_availability",
    }
)

DEFAULT_INJURY_HEAVY_THRESHOLD = 0.0
