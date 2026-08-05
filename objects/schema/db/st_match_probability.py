from datetime import date

from pydantic import BaseModel, Field


class STMatchProbabilityResult(BaseModel):
    """Calculated 1X2 probabilities for one Stryktipset coupon match."""

    draw_number: int
    event_number: int
    match_id: int
    home_team: str
    away_team: str
    match_date: date
    probabilities: dict[str, float]
    engine_probabilities: dict[str, float] | None = None
    market_probabilities: dict[str, float] | None = None
    ml_probabilities: dict[str, float] | None = None
    final_probabilities: dict[str, float] | None = None
    ml_enabled: bool = False
    ml_model_version: str | None = None
    draw_boost_score: float = 0.0
    draw_value_gap: float | None = None
    ml_notes: list[str] = Field(default_factory=list)
