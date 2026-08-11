from pydantic import BaseModel


class STMatchOddsCreate(BaseModel):
    """Schema for creating a STMatchOdds."""

    stryktipset_match_id: int
    odds_1: float
    odds_X: float
    odds_2: float


class STMatchOdds(BaseModel):
    """Schema for STMatchOdds with id (read/response)."""

    id: int
    stryktipset_match_id: int
    odds_1: float
    odds_X: float
    odds_2: float

    model_config = {"from_attributes": True}
