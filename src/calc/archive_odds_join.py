"""Resolve ST coupon matches to archive fixture closing odds (eval only)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from src.calc.market_probabilities import load_fixture_market_probabilities
from src.objects.models.fixture import FixtureModel
from src.objects.models.st_match import STMatchModel
from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.repositories.team_repository import TeamRepository
from src.objects.schema.data_classes.market_probability import MarketProbabilityBreakdown
from src.utils.datetime_tz import align_datetime_tzinfo


def _as_calendar_date(kickoff: datetime | date) -> date:
    if isinstance(kickoff, datetime):
        return kickoff.date()
    return kickoff


def _kickoff_distance_seconds(
    fixture: FixtureModel,
    kickoff: datetime,
) -> float:
    fixture_dt = fixture.fixture_date
    if fixture_dt is None:
        return float("inf")
    fixture_dt, aligned_kickoff = align_datetime_tzinfo(fixture_dt, kickoff)
    return abs((fixture_dt - aligned_kickoff).total_seconds())


def pick_closest_fixture(
    candidates: list[FixtureModel],
    kickoff: datetime,
) -> FixtureModel:
    """Pick the fixture whose kickoff is closest to ``kickoff``."""
    if not candidates:
        raise ValueError("candidates is empty")
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda fixture: _kickoff_distance_seconds(fixture, kickoff))


def resolve_archive_fixture(
    session: Session,
    *,
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime | date | None,
) -> FixtureModel | None:
    """ST ``teams.id`` → ``Team.external_id`` → fixture in date ±1 day.

    Fixture ``home_team_id`` is the API-Football external id, not ``teams.id``.
    Multiple fixtures → closest kickoff. Missing team, missing external_id,
    missing kickoff, or no fixture → ``None``.
    """
    if kickoff is None:
        return None
    team_repo = TeamRepository(session)
    home = team_repo.get(home_team_id)
    away = team_repo.get(away_team_id)
    if home is None or away is None:
        return None
    if home.external_id is None or away.external_id is None:
        return None
    kickoff_date = _as_calendar_date(kickoff)
    candidates = FixtureRepository(session).find_by_date_range_and_teams(
        date_from=kickoff_date - timedelta(days=1),
        date_to=kickoff_date + timedelta(days=1),
        home_team_ids=[int(home.external_id)],
        away_team_ids=[int(away.external_id)],
    )
    if not candidates:
        return None
    if isinstance(kickoff, datetime):
        return pick_closest_fixture(candidates, kickoff)
    return candidates[0]


def resolve_archive_fixture_for_st_match(
    session: Session,
    match: STMatchModel,
) -> FixtureModel | None:
    home = match.home_team
    away = match.away_team
    if home is None or away is None:
        return resolve_archive_fixture(
            session,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            kickoff=match.start_time,
        )
    if home.external_id is None or away.external_id is None or match.start_time is None:
        return None
    kickoff_date = _as_calendar_date(match.start_time)
    candidates = FixtureRepository(session).find_by_date_range_and_teams(
        date_from=kickoff_date - timedelta(days=1),
        date_to=kickoff_date + timedelta(days=1),
        home_team_ids=[int(home.external_id)],
        away_team_ids=[int(away.external_id)],
    )
    if not candidates:
        return None
    return pick_closest_fixture(candidates, match.start_time)


def load_archive_market_for_st_match(
    session: Session,
    match: STMatchModel,
) -> MarketProbabilityBreakdown | None:
    """Avg+closing archive market for one ST match, or ``None`` if unresolved."""
    fixture = resolve_archive_fixture_for_st_match(session, match)
    if fixture is None:
        return None
    return load_fixture_market_probabilities(session, fixture.id)
