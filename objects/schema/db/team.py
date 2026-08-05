from typing import Optional

from pydantic import BaseModel


class TeamCreate(BaseModel):
    """Schema for creating a Team."""

    external_id: Optional[int] = None
    league_id: Optional[int] = None
    name: str
    short_name: Optional[str] = None
    medium_name: Optional[str] = None
    country_name: Optional[str] = None
    iso_code: Optional[str] = None


class Team(BaseModel):
    """Schema for Team with id (read/response)."""

    id: int
    external_id: Optional[int] = None
    league_id: Optional[int] = None
    name: str
    short_name: Optional[str] = None
    medium_name: Optional[str] = None
    country_name: Optional[str] = None
    iso_code: Optional[str] = None

    model_config = {"from_attributes": True}
