from datetime import date
from typing import Any, Optional

from pydantic import BaseModel


class HistoricalMatchDraft(BaseModel):
    """Inbound match with team names before EntityResolver assigns FKs."""

    source: str
    league: str
    season: str
    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    raw_data: Optional[dict[str, Any]] = None


class HistoricalMatchCreate(BaseModel):
    """Schema for creating a HistoricalMatch with resolved team FKs."""

    source: str
    league: str
    season: str
    match_date: date
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    result: str
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    raw_data: Optional[dict[str, Any]] = None


class HistoricalMatch(BaseModel):
    """Schema for HistoricalMatch with id (read/response)."""

    id: int
    source: str
    league: str
    season: str
    match_date: date
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    raw_data: Optional[dict[str, Any]] = None
    league_id: Optional[str] = None
    actual_outcome: Optional[str] = None
    market_home_probability: Optional[float] = None
    market_draw_probability: Optional[float] = None
    market_away_probability: Optional[float] = None
    public_home_percentage: Optional[float] = None
    public_draw_percentage: Optional[float] = None
    public_away_percentage: Optional[float] = None

    model_config = {"from_attributes": True}
