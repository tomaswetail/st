"""Rest days since previous match."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from objects.models.team import TeamModel
from objects.repositories.fixture_repository import FixtureRepository as HistoricalMatchRepository
from utils.fixture_fields import fixture_away_name, fixture_home_name, fixture_match_date
from objects.schema.data_classes.recent_form_stats import TeamRestDays
from objects.schema.data_classes.team_rest_days import MatchRestDaysFeatures
from objects.schema.db.fixture import Fixture


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer value to the inclusive range [low, high]."""
    return max(low, min(high, value))


def build_last_match_dates(
    history: list[Fixture],
    fixtures: list[Any] | None = None,
) -> dict[str, date]:
    """Return the latest known match date for each team from history and fixtures."""
    last: dict[str, date] = {}
    for m in history:
        last[fixture_home_name(m)] = max(last.get(fixture_home_name(m), fixture_match_date(m)), fixture_match_date(m))
        last[fixture_away_name(m)] = max(last.get(fixture_away_name(m), fixture_match_date(m)), fixture_match_date(m))
    for f in fixtures or []:
        d = f.start_time.date()
        if f.status and f.status.lower() in ("finished", "completed", "ft"):
            home = f.home_team.name if hasattr(f.home_team, "name") else f.home_team
            away = f.away_team.name if hasattr(f.away_team, "name") else f.away_team
            last[home] = max(last.get(home, d), d)
            last[away] = max(last.get(away, d), d)
    return last


def rest_days_for_team(team: str, match_date: date, last_dates: dict[str, date]) -> int | None:
    """Return days since a team's last match before match_date."""
    prev = last_dates.get(team)
    if not prev or prev >= match_date:
        return None
    return (match_date - prev).days


def rest_days_for_match(
    home: str,
    away: str,
    match_date: date | datetime,
    last_dates: dict[str, date],
) -> tuple[int | None, int | None, int | None]:
    """Return home rest days, away rest days, and their difference."""
    if isinstance(match_date, datetime):
        match_date = match_date.date()
    h = rest_days_for_team(home, match_date, last_dates)
    a = rest_days_for_team(away, match_date, last_dates)
    diff = (h - a) if h is not None and a is not None else None
    return h, a, diff


def calculate_team_rest_days(
    matches: list[Fixture],
    team: str,
    match_date: date,
) -> TeamRestDays:
    """Compute rest days from historical matches strictly before match_date."""
    relevant = [
        m
        for m in matches
        if fixture_match_date(m) < match_date and team in (fixture_home_name(m), fixture_away_name(m))
    ]
    relevant.sort(key=lambda m: fixture_match_date(m), reverse=True)

    if not relevant:
        return TeamRestDays(
            team=team,
            match_date=match_date,
            previous_match_date=None,
            rest_days=None,
            matches_checked=0,
            note="No previous match found",
        )

    previous = fixture_match_date(relevant[0])
    rest = (match_date - previous).days
    return TeamRestDays(
        team=team,
        match_date=match_date,
        previous_match_date=previous,
        rest_days=rest,
        matches_checked=len(relevant),
    )


def build_match_rest_days_features(
    matches: list[Fixture],
    home_team: str,
    away_team: str,
    match_date: date,
    *,
    diff_cap: int = 5,
) -> MatchRestDaysFeatures:
    """Compare home and away rest days and compute a rest advantage score."""
    home_rest = calculate_team_rest_days(matches, home_team, match_date)
    away_rest = calculate_team_rest_days(matches, away_team, match_date)
    notes: list[str] = []

    if home_rest.rest_days is None or away_rest.rest_days is None:
        notes.append("Missing previous match for one or both teams; neutral rest adjustment.")
        return MatchRestDaysFeatures(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            home_rest=home_rest,
            away_rest=away_rest,
            home_rest_days=home_rest.rest_days,
            away_rest_days=away_rest.rest_days,
            rest_day_diff=None,
            rest_advantage_score=0.0,
            confidence=0.0,
            notes=notes,
        )

    rest_day_diff = home_rest.rest_days - away_rest.rest_days
    capped_diff = _clamp(rest_day_diff, -diff_cap, diff_cap)
    rest_advantage_score = capped_diff / diff_cap

    if home_rest.rest_days > 21 or away_rest.rest_days > 21:
        notes.append("Long break detected; rest-day effect may behave differently.")

    return MatchRestDaysFeatures(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        home_rest=home_rest,
        away_rest=away_rest,
        home_rest_days=home_rest.rest_days,
        away_rest_days=away_rest.rest_days,
        rest_day_diff=rest_day_diff,
        rest_advantage_score=rest_advantage_score,
        confidence=1.0,
        notes=notes,
    )



class RestdayCalculator:
    """Calculate team rest days from historical match data."""

    def __init__(self, session: Session | None = None) -> None:
        """Initialize with an optional shared database session."""
        self.historical_match_repo = HistoricalMatchRepository(session)

    def calculate_rest_days(self, team: TeamModel, match_date: date) -> int | None:
        """Return days since the team's last match before match_date."""
        if team is None:
            raise ValueError("team is required")

        matches = self.historical_match_repo.get_matches_by_team(team)
        result = calculate_team_rest_days(matches, team.name, match_date)
        return result.rest_days
