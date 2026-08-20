"""External league information from API-Football."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeagueInfo:
    league_id: int
    league_name: str
    league_type: str
    country_name: str
    country_code: str | None = None
    seasons: list[int] = field(default_factory=list)
