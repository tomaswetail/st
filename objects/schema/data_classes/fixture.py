"""API-Football fixture payload (normalized)."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fixture:
    fixture_id: int
    fixture_referee: str | None
    fixture_timezone: str
    fixture_date: datetime
    fixture_timestamp: int

    period_first: int | None
    period_second: int | None

    venue_id: int | None
    venue_name: str | None
    venue_city: str | None

    status_long: str
    status_short: str

    league_id: int
    league_name: str
    league_country: str | None
    league_flag: str | None
    league_season: int
    league_round: str | None

    home_team_id: int
    home_team_name: str
    home_team_winner: bool | None

    away_team_id: int
    away_team_name: str
    away_team_winner: bool | None

    goals_home: int | None
    goals_away: int | None

    score_halftime_home: int | None
    score_halftime_away: int | None
    score_fulltime_home: int | None
    score_fulltime_away: int | None
    score_extratime_home: int | None
    score_extratime_away: int | None
    score_penalty_home: int | None
    score_penalty_away: int | None
