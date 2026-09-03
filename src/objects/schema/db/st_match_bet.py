from pydantic import BaseModel


class STMatchBetCreate(BaseModel):
    """Schema for creating a STMatchBet."""

    stryktipset_match_id: int
    distribution_1: int
    distribution_X: int
    distribution_2: int


class STMatchBet(BaseModel):
    """Schema for STMatchBet with id (read/response)."""

    id: int
    stryktipset_match_id: int
    distribution_1: int
    distribution_X: int
    distribution_2: int

    model_config = {"from_attributes": True}
