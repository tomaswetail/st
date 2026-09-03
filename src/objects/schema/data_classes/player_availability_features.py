"""Player availability / injury features for residual ML."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerAvailabilityFeatures:
    """Cutoff-safe squad availability signals for one fixture.

    All fields stay None when no pre-cutoff snapshot exists (missingness).
    ``has_availability`` is 1 when a usable snapshot was found, else 0.
    """

    home_missing_player_value: float | None
    away_missing_player_value: float | None
    missing_value_difference: float | None
    home_unavailable_count: int | None
    away_unavailable_count: int | None
    home_lineup_changes: int | None
    away_lineup_changes: int | None
    missing_value_x_favourite: float | None
    short_rest_x_missing_value: float | None
    has_availability: int
    coverage_level: str | None = None
