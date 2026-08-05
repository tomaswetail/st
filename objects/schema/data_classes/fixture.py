"""API-Football fixture schema."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fixture:
    fixture_id: int
    league_id: int
    league_name: str
    country: str
    season: int
    start_time: datetime
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    status: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    halftime_home_goals: int | None = None
    halftime_away_goals: int | None = None
