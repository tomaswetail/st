from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class STMatchCreate(BaseModel):
    """Schema for creating a STMatch."""

    stryktipset_round_id: int
    external_id: int
    start_time: Optional[datetime] = None
    status: Optional[str] = None
    status_id: Optional[int] = None
    league_name: Optional[str] = None
    league_country_name: Optional[str] = None
    home_team_id: int
    away_team_id: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    stryktipset_result: Optional[str] = None
    halftime_home_score: Optional[int] = None
    halftime_away_score: Optional[int] = None


class STMatch(BaseModel):
    """Schema for STMatch with id (read/response)."""

    id: int
    stryktipset_round_id: int
    external_id: int
    start_time: Optional[datetime] = None
    status: Optional[str] = None
    status_id: Optional[int] = None
    league_name: Optional[str] = None
    league_country_name: Optional[str] = None
    home_team_id: int
    away_team_id: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    stryktipset_result: Optional[str] = None
    halftime_home_score: Optional[int] = None
    halftime_away_score: Optional[int] = None

    model_config = {"from_attributes": True}
