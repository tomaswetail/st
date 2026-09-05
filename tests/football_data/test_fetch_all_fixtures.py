"""Tests for find_missing_stats and fetch_and_store_all_fixtures."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.football_data.service import ExtendedMatchDataService
from src.objects.models.fixture import FixtureModel
from src.objects.repositories.fixture_repository import FixtureRepository
from src.scripts.missing_stats import fetch_missing_stats
from src.utils.common import LEAGUES_EXTERNAL_IDS


# ---------------------------------------------------------------------------
# find_missing_stats
# ---------------------------------------------------------------------------


def _make_fixture(id_: int, status: str = "FT") -> MagicMock:
    f = MagicMock(spec=FixtureModel)
    f.id = id_
    f.status_short = status
    return f


def test_find_missing_stats_returns_scalars(monkeypatch):
    """find_missing_stats should delegate to SQLAlchemy and return the list."""
    session = MagicMock()
    repo = FixtureRepository(session)

    expected = [_make_fixture(1), _make_fixture(2)]
    session.scalars.return_value.all.return_value = expected

    result = repo.find_missing_stats("fotmob")

    assert result == expected
    session.scalars.assert_called_once()


def test_find_missing_stats_with_limit(monkeypatch):
    session = MagicMock()
    repo = FixtureRepository(session)
    session.scalars.return_value.all.return_value = [_make_fixture(5)]

    result = repo.find_missing_stats("fotmob", limit=1)

    assert len(result) == 1
    session.scalars.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_and_store_all_fixtures
# ---------------------------------------------------------------------------


def _make_service() -> ExtendedMatchDataService:
    session = MagicMock()
    service = ExtendedMatchDataService(provider="fotmob", session=session)
    return service


def test_fetch_and_store_all_fixtures_calls_find_missing_stats():
    service = _make_service()
    fixtures = [_make_fixture(10), _make_fixture(11)]
    service.fixture_repo.find_missing_stats = MagicMock(return_value=fixtures)
    service.fetch_and_store_matches = MagicMock(
        return_value=SimpleNamespace(requested=2, imported=2, failed=0, skipped=0)
    )

    service.fetch_and_store_all_fixtures(external_league_ids=[39], limit=50)

    service.fixture_repo.find_missing_stats.assert_called_once_with(
        "fotmob",
        external_league_ids=[39],
        seasons=None,
        before_date=None,
        limit=50,
    )
    service.fetch_and_store_matches.assert_called_once_with([10, 11], force_refresh=False)


def test_fetch_and_store_all_fixtures_force_refresh_uses_get_filtered():
    service = _make_service()
    fixtures = [_make_fixture(20)]
    service.fixture_repo.get_filtered = MagicMock(return_value=fixtures)
    service.fixture_repo.find_missing_stats = MagicMock()
    service.fetch_and_store_matches = MagicMock(
        return_value=SimpleNamespace(requested=1, imported=1, failed=0, skipped=0)
    )

    service.fetch_and_store_all_fixtures(force_refresh=True)

    service.fixture_repo.get_filtered.assert_called_once_with(
        external_league_ids=None,
        seasons=None,
        before_date=None,
        limit=None,
    )
    service.fixture_repo.find_missing_stats.assert_not_called()
    service.fetch_and_store_matches.assert_called_once_with([20], force_refresh=True)


def test_fetch_and_store_all_fixtures_empty_result():
    service = _make_service()
    service.fixture_repo.find_missing_stats = MagicMock(return_value=[])
    service.fetch_and_store_matches = MagicMock(
        return_value=SimpleNamespace(requested=0, imported=0, failed=0, skipped=0)
    )

    result = service.fetch_and_store_all_fixtures()

    service.fetch_and_store_matches.assert_called_once_with([], force_refresh=False)
    assert result.requested == 0


def test_fetch_missing_stats_calls_fetch_and_store_all_fixtures():
    batch = SimpleNamespace(
        requested=2, imported=1, updated=0, skipped=1, unresolved=0, failed=0
    )
    service = MagicMock()
    service.fetch_and_store_all_fixtures.return_value = batch
    session = MagicMock()

    with (
        patch("src.scripts.missing_stats.init_db"),
        patch("src.scripts.missing_stats.SessionLocal", return_value=session),
        patch(
            "src.scripts.missing_stats.ExtendedMatchDataService",
            return_value=service,
        ),
    ):
        result = fetch_missing_stats(
            seasons=[2024],
            before_date=date(2025, 1, 1),
            limit=10,
        )

    assert result is batch
    service.fetch_and_store_all_fixtures.assert_called_once_with(
        external_league_ids=[int(league_id) for league_id in LEAGUES_EXTERNAL_IDS],
        seasons=[2024],
        before_date=date(2025, 1, 1),
        limit=10,
        force_refresh=False,
    )
    service.close.assert_called_once()
    session.close.assert_called_once()
