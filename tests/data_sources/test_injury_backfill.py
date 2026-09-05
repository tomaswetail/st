"""Tests for API-Football injury backfill service."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.data_sources.injuries.backfill import InjuryBackfillService
from src.data_sources.injuries.dtos import MatchAvailabilitySnapshot, PlayerAvailabilityRecord


def _sample_snapshot(fixture_id: int = 42) -> MatchAvailabilitySnapshot:
    kickoff = datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc)
    return MatchAvailabilitySnapshot(
        fixture_id=fixture_id,
        provider="api-football",
        source="api_football_injuries_lineups",
        snapshot_at=kickoff,
        kickoff_at=kickoff,
        home_team_external_id="50",
        away_team_external_id="42",
        players=[
            PlayerAvailabilityRecord(
                external_id="10",
                provider="api-football",
                team_side="home",
                status="starter",
                is_starter=True,
                name="Alice",
                team_external_id="50",
            ),
        ],
        home_starter_count=1,
        away_starter_count=0,
        coverage_level="partial",
    )


def test_backfill_skip_http_counts_candidates() -> None:
    session = MagicMock()
    client = MagicMock()
    service = InjuryBackfillService(session, client=client, request_delay_sec=0)
    service.availability_repo.fixture_ids_with_availability = MagicMock(
        return_value=set()
    )
    fixture = SimpleNamespace(
        id=99,
        fixture_id=1000,
        home_team_id=50,
        away_team_id=42,
        fixture_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    service._load_candidate_fixtures = MagicMock(return_value=[fixture])  # type: ignore[method-assign]

    result = service.backfill(skip_http=True, dry_run=True)

    assert result.requested == 1
    assert result.imported == 1
    assert result.fixture_ids == [99]
    client.get_fixture_lineups.assert_not_called()
    session.commit.assert_not_called()


def test_backfill_skips_existing_without_force_refresh() -> None:
    session = MagicMock()
    client = MagicMock()
    service = InjuryBackfillService(session, client=client, request_delay_sec=0)
    fixture = SimpleNamespace(
        id=42,
        fixture_id=1000,
        home_team_id=50,
        away_team_id=42,
        fixture_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    service._load_candidate_fixtures = MagicMock(return_value=[fixture])  # type: ignore[method-assign]
    service.availability_repo.fixture_ids_with_availability = MagicMock(
        return_value={42}
    )
    service.availability_repo.get_for_fixture = MagicMock(return_value=object())

    result = service.backfill()

    assert result.skipped == 1
    client.get_fixture_lineups.assert_not_called()
    session.commit.assert_called_once()


def test_backfill_fetches_and_persists() -> None:
    session = MagicMock()
    client = MagicMock()
    client.get_fixture_lineups.return_value = {"response": []}
    client.get_fixture_injuries.return_value = {"response": []}
    service = InjuryBackfillService(session, client=client, request_delay_sec=0)
    fixture = SimpleNamespace(
        id=42,
        fixture_id=1000,
        home_team_id=50,
        away_team_id=42,
        fixture_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    service._load_candidate_fixtures = MagicMock(return_value=[fixture])  # type: ignore[method-assign]
    service.availability_repo.fixture_ids_with_availability = MagicMock(
        return_value=set()
    )
    service.availability_repo.get_for_fixture = MagicMock(return_value=None)
    service._persist_snapshot = MagicMock()  # type: ignore[method-assign]

    with patch(
        "src.data_sources.injuries.backfill.parse_api_football_availability",
        return_value=_sample_snapshot(42),
    ):
        result = service.backfill()

    assert result.imported == 1
    client.get_fixture_lineups.assert_called_once_with(1000)
    client.get_fixture_injuries.assert_called_once_with(1000)
    service._persist_snapshot.assert_called_once()
    session.commit.assert_called_once()
