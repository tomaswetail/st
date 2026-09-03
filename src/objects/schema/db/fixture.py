from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class FixtureCreate(BaseModel):
    """Schema for creating a Fixture row from API-Football data."""

    fixture_id: int
    fixture_referee: Optional[str] = None
    fixture_timezone: str
    fixture_date: datetime
    fixture_timestamp: int

    period_first: Optional[int] = None
    period_second: Optional[int] = None

    venue_id: Optional[int] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None

    status_long: str
    status_short: str

    league_id: int
    league_name: str
    league_country: Optional[str] = None
    league_flag: Optional[str] = None
    league_season: int
    league_round: Optional[str] = None

    home_team_id: int
    home_team_name: str
    home_team_winner: Optional[bool] = None

    away_team_id: int
    away_team_name: str
    away_team_winner: Optional[bool] = None

    goals_home: Optional[int] = None
    goals_away: Optional[int] = None

    score_halftime_home: Optional[int] = None
    score_halftime_away: Optional[int] = None
    score_fulltime_home: Optional[int] = None
    score_fulltime_away: Optional[int] = None
    score_extratime_home: Optional[int] = None
    score_extratime_away: Optional[int] = None
    score_penalty_home: Optional[int] = None
    score_penalty_away: Optional[int] = None


class Fixture(FixtureCreate):
    """Schema for Fixture with id (read/response)."""

    id: int

    model_config = {"from_attributes": True}
