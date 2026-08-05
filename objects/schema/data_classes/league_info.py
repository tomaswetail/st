"""External league information schema."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueInfo:
    league_id: int
    name: str
    country: str
    league_type: str
    seasons: list[int]
