from typing import Optional

from pydantic import BaseModel


class LeagueCreate(BaseModel):
    """Schema for creating a League."""

    external_id: int
    league_name: str
    league_type: str
    country_name: str
    country_code: Optional[str] = None


class League(BaseModel):
    """Schema for League with id (read/response)."""

    id: int
    external_id: int
    league_name: str
    league_type: str
    country_name: str
    country_code: Optional[str] = None

    model_config = {"from_attributes": True}
