from typing import Optional

from pydantic import BaseModel


class TeamCreate(BaseModel):
    """Schema for creating a Team."""

    external_id: int
    name: str
    code: Optional[str] = None
    country: Optional[str] = None
    national: bool = False


class Team(BaseModel):
    """Schema for Team with id (read/response)."""

    id: int
    external_id: int
    name: str
    code: Optional[str] = None
    country: Optional[str] = None
    national: bool = False

    model_config = {"from_attributes": True}
