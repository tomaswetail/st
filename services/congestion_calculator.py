"""Fixture congestion from recent match frequency."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from objects.models.team import TeamModel
from objects.repositories.fixture_repository import FixtureRepository as HistoricalMatchRepository
from utils.fixture_fields import fixture_away_name, fixture_home_name, fixture_match_date
from objects.schema.db.fixture import Fixture


def count_matches_in_window(
    matches: list[Fixture],
    team: str,
    match_date: date,
    days: int,
) -> int:
    """Count team matches in the N-day window strictly before match_date.

    Args:
        matches: Historical matches to filter; home_team and away_team are
            checked for team membership.
        team: Team name to count appearances for.
        match_date: Reference fixture date; only matches before this date count.
        days: Window length in days; includes dates from match_date - days
            through match_date - 1 day inclusive.
    """
    window_start = match_date - timedelta(days=days)
    count = 0
    for m in matches:
        if fixture_match_date(m) >= match_date or fixture_match_date(m) < window_start:
            continue
        if team in (fixture_home_name(m), fixture_away_name(m)):
            count += 1
    return count


def calculate_team_congestion(
    matches: list[Fixture],
    team: str,
    match_date: date,
) -> float:
    """Return a congestion score in [0.0, 1.0] from recent match counts.

    Args:
        matches: Historical matches passed to count_matches_in_window for
            7-, 14-, and 21-day lookbacks.
        team: Team name used when filtering matches in each window.
        match_date: Reference fixture date for all three windows.

    Returns:
        Weighted congestion score capped at 1.0.
    """
    matches_last_7_days = count_matches_in_window(matches, team, match_date, 7)
    matches_last_14_days = count_matches_in_window(matches, team, match_date, 14)
    matches_last_21_days = count_matches_in_window(matches, team, match_date, 21)

    return (
        0.60 * min(matches_last_7_days / 3, 1.0)
        + 0.30 * min(matches_last_14_days / 5, 1.0)
        + 0.10 * min(matches_last_21_days / 7, 1.0)
    )


class CongestionCalculator:
    """Calculate team fixture congestion from historical match data."""

    def __init__(self, session: Session | None = None) -> None:
        """Initialize with an optional shared database session."""
        self.historical_match_repo = HistoricalMatchRepository(session)

    def calculate_congestion(self, team: TeamModel, match_date: date) -> float:
        """Return congestion score for a team before match_date.

        Args:
            team: Used to load historical matches via get_matches_by_team;
                team.name is passed to calculate_team_congestion.
            match_date: Reference fixture date for the congestion windows.

        Returns:
            Congestion score in [0.0, 1.0].
        """
        if team is None:
            raise ValueError("team is required")

        matches = self.historical_match_repo.get_matches_by_team(team)
        return calculate_team_congestion(matches, team.name, match_date)
