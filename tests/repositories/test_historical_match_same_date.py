"""Unit tests for fixture upsert by fixture_id."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.schema.db.fixture import FixtureCreate


def _repo() -> FixtureRepository:
    session = MagicMock()
    return FixtureRepository(session)


def _create(*, fixture_id: int = 100) -> FixtureCreate:
    kickoff = datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc)
    return FixtureCreate(
        fixture_id=fixture_id,
        fixture_timezone="UTC",
        fixture_date=kickoff,
        fixture_timestamp=int(kickoff.timestamp()),
        status_long="Match Finished",
        status_short="FT",
        league_id=39,
        league_name="Premier League",
        league_season=2024,
        home_team_id=42,
        home_team_name="Arsenal",
        away_team_id=39,
        away_team_name="Wolves",
        goals_home=1,
        goals_away=0,
    )


def test_upsert_many_writes_by_fixture_id():
    repo = _repo()
    match = _create()

    with patch(
        "src.objects.repositories.fixture_repository.pg_insert"
    ) as insert_mock:
        statement = MagicMock()
        insert_mock.return_value.values.return_value.on_conflict_do_update.return_value = (
            statement
        )
        written = repo.upsert_many([match])

    assert written == 1
    repo.session.execute.assert_called_once()
    repo.session.commit.assert_called_once()
    insert_mock.return_value.values.assert_called_once()


def test_upsert_many_writes_each_fixture():
    repo = _repo()
    first = _create(fixture_id=1)
    second = _create(fixture_id=2)

    with patch(
        "src.objects.repositories.fixture_repository.pg_insert"
    ) as insert_mock:
        statement = MagicMock()
        insert_mock.return_value.values.return_value.on_conflict_do_update.return_value = (
            statement
        )
        written = repo.upsert_many([first, second])

    assert written == 2
    assert insert_mock.call_count == 2
    assert repo.session.execute.call_count == 2
