from datetime import date

from pydantic import BaseModel, Field

from objects.schema.data_classes.recent_form_stats import RecentFormStats, TeamRestDays


class MatchRecentFormFeatures(BaseModel):
    """Home vs away recent form comparison for one coupon match."""

    home_team: str
    away_team: str
    match_date: date
    lookback: int
    home_form: RecentFormStats
    away_form: RecentFormStats
    points_diff: float
    weighted_form_diff: float
    goal_diff_per_match_diff: float
    goals_for_per_match_diff: float
    goals_against_per_match_diff: float
    form_advantage_score: float
    confidence: float
    notes: list[str] = Field(default_factory=list)


class MatchRestDaysFeatures(BaseModel):
    """Home vs away rest-day comparison for one fixture."""

    home_team: str
    away_team: str
    match_date: date
    home_rest: TeamRestDays
    away_rest: TeamRestDays
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    rest_day_diff: int | None = None
    rest_advantage_score: float
    confidence: float
    notes: list[str] = Field(default_factory=list)
