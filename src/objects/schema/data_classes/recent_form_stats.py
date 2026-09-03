from datetime import date

from pydantic import BaseModel


class RecentFormStats(BaseModel):
    """Recent form statistics for a single team before a fixture date."""

    team: str
    before_date: date
    lookback: int
    matches_used: int
    points: int
    goals_for: int
    goals_against: int
    goal_diff: int
    wins: int
    draws: int
    losses: int
    weighted_points: float
    max_weighted_points: float
    weighted_form_score: float | None = None
    points_per_match: float | None = None
    goals_for_per_match: float | None = None
    goals_against_per_match: float | None = None
    goal_diff_per_match: float | None = None





class TeamRestDays(BaseModel):
    """Rest days for one team before a fixture."""

    team: str
    match_date: date
    previous_match_date: date | None = None
    rest_days: int | None = None
    matches_checked: int
    note: str | None = None