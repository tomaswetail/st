from datetime import date

from pydantic import BaseModel, Field


class STMatchProbabilityResult(BaseModel):
    """Calculated 1X2 probabilities for one Stryktipset coupon match."""

    draw_number: int
    event_number: int = 0
    match_id: int
    home_team: str
    away_team: str
    match_date: date
    probabilities: dict[str, float]
    market_probabilities: dict[str, float] | None = None
    ml_probabilities: dict[str, float] | None = None
    final_probabilities: dict[str, float] | None = None
    ml_enabled: bool = False
    ml_model_version: str | None = None
    ml_notes: list[str] = Field(default_factory=list)
