from typing import Optional

from pydantic import BaseModel


class STRoundCreate(BaseModel):
    """Schema for creating a StryktipsetRound."""

    product_id: int
    product_name: str
    draw_number: int
    event_number: int
    description: Optional[str] = None
    comment: Optional[str] = None
    cancelled: bool = False


class STRound(BaseModel):
    """Schema for StryktipsetRound with id (read/response)."""

    id: int
    product_id: int
    product_name: str
    draw_number: int
    event_number: int
    description: str
    comment: str
    cancelled: bool

    model_config = {"from_attributes": True}
