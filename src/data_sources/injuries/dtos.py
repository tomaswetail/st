"""DTOs for injury and player availability snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

PlayerStatus = Literal[
    "starter",
    "bench",
    "injured",
    "suspended",
    "unavailable",
    "doubtful",
    "unknown",
]
TeamSide = Literal["home", "away"]
AvailabilitySource = Literal[
    "api_football_injuries_lineups",
]


@dataclass(frozen=True)
class PlayerAvailabilityRecord:
    """One player's availability at a snapshot cutoff."""

    external_id: str
    provider: str
    team_side: TeamSide
    status: PlayerStatus
    is_starter: bool
    name: str | None = None
    team_external_id: str | None = None
    market_value: float | None = None
    shirt_number: str | None = None
    position_id: int | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchAvailabilitySnapshot:
    """Point-in-time squad availability for one fixture."""

    fixture_id: int
    provider: str
    source: AvailabilitySource
    snapshot_at: datetime
    kickoff_at: datetime
    home_team_external_id: str | None
    away_team_external_id: str | None
    players: list[PlayerAvailabilityRecord]
    home_starter_count: int = 0
    away_starter_count: int = 0
    home_unavailable_count: int = 0
    away_unavailable_count: int = 0
    home_missing_value: float | None = None
    away_missing_value: float | None = None
    coverage_level: Literal["full_lineup", "partial", "none"] = "none"
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.home_starter_count == 0 and self.players:
            object.__setattr__(
                self,
                "home_starter_count",
                sum(
                    1
                    for player in self.players
                    if player.team_side == "home" and player.is_starter
                ),
            )
        if self.away_starter_count == 0 and self.players:
            object.__setattr__(
                self,
                "away_starter_count",
                sum(
                    1
                    for player in self.players
                    if player.team_side == "away" and player.is_starter
                ),
            )
        if self.home_unavailable_count == 0 and self.players:
            unavailable_statuses = {"injured", "suspended", "unavailable", "doubtful"}
            object.__setattr__(
                self,
                "home_unavailable_count",
                sum(
                    1
                    for player in self.players
                    if player.team_side == "home" and player.status in unavailable_statuses
                ),
            )
        if self.away_unavailable_count == 0 and self.players:
            unavailable_statuses = {"injured", "suspended", "unavailable", "doubtful"}
            object.__setattr__(
                self,
                "away_unavailable_count",
                sum(
                    1
                    for player in self.players
                    if player.team_side == "away" and player.status in unavailable_statuses
                ),
            )
