from pydantic import BaseModel


class LeagueCreate(BaseModel):
    """Schema for creating a League."""

    name: str
    country: str


class League(BaseModel):
    """Schema for League with id (read/response)."""

    id: int
    name: str
    country: str

    model_config = {"from_attributes": True}
