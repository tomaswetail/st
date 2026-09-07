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

COUNTS_ONLY_INJURY_COLUMNS: frozenset[str] = frozenset(
    {
        "home_unavailable_count",
        "away_unavailable_count",
        "has_availability",
    }
)

DEFAULT_INJURY_HEAVY_THRESHOLD = 0.0


def excluded_injury_feature_columns(
    *,
    exclude_injury_features: bool = False,
    injury_counts_only: bool = False,
) -> frozenset[str]:
    """Columns to drop from HGB. All-or-nothing exclude wins over counts-only."""
    if exclude_injury_features:
        return INJURY_FEATURE_COLUMNS
    if injury_counts_only:
        return INJURY_FEATURE_COLUMNS - COUNTS_ONLY_INJURY_COLUMNS
    return frozenset()
