from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

from src.data_sources.entity_resolver import TeamResolution
from src.data_sources.football_data.http_client import ThrottledHttpClient
from src.data_sources.football_data.providers.fotmob import (
    FotMobProvider,
    parse_fotmob_matches_by_date,
)
from src.data_sources.football_data.service import ExtendedMatchDataService
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.provider_dtos import ProviderMatch


def test_parse_fotmob_matches_by_date() -> None:
    payload = {
        "leagues": [
            {
                "id": 47,
                "matches": [
                    {
                        "id": 1001,
                        "home": {"id": 8456, "name": "Arsenal"},
                        "away": {"id": 10260, "name": "Chelsea"},
                        "status": {"utcTime": "2024-08-17T14:00:00.000Z"},
                    }
                ],
            }
        ]
    }

    matches = parse_fotmob_matches_by_date(payload)

    assert len(matches) == 1
    assert matches[0].provider_match_id == "1001"
    assert matches[0].provider_league_id == "47"
    assert matches[0].home_team_name == "Arsenal"
    assert matches[0].kickoff_at == datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc)


def test_fetch_matches_by_date_uses_cache(tmp_path: Path) -> None:
    payload = {
        "leagues": [
            {
                "id": 47,
                "matches": [
                    {
                        "id": 1001,
                        "home": {"id": 8456, "name": "Arsenal"},
                        "away": {"id": 10260, "name": "Chelsea"},
                        "status": {"utcTime": "2024-08-17T14:00:00.000Z"},
                    }
                ],
            }
        ]
    }
    client = ThrottledHttpClient(
        base_url="https://example.test",
        max_retries=0,
        request_delay_ms=0,
        cache_dir=tmp_path,
    )
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=payload)

    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://example.test",
    )
    provider = FotMobProvider(client=client)

    try:
        first = provider.fetch_matches_by_date(date(2024, 8, 17))
        second = provider.fetch_matches_by_date(date(2024, 8, 17))
    finally:
        provider.close()

    assert len(first) == 1
    assert len(second) == 1
    assert calls["count"] == 1
    assert list(tmp_path.glob("*.json"))


def test_resolve_fixture_teams_prefers_mapping_then_resolver() -> None:
    service = ExtendedMatchDataService(provider="fotmob", session=MagicMock())
    service.teams_repo.get_by_external_id = MagicMock(return_value=SimpleNamespace(id=10))
    service.resolver.resolve_team = MagicMock(
        return_value=TeamResolution(
            team=SimpleNamespace(id=20),
            confidence=0.8,
            method="exact_name",
        )
    )
    service._append_missing_mapping_key = MagicMock()
    fixture = ProviderMatch(
        provider_match_id="1001",
        provider_league_id="47",
        provider_season_id=None,
        home_team_id="8456",
        away_team_id="99999999",
        home_team_name="Arsenal",
        away_team_name="Unknown FC",
        kickoff_at=datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc),
        status="scheduled",
    )

    home, away = service._resolve_fixture_teams(fixture, league_id=1)

    assert home is not None
    assert home.team.id == 10
    assert away is not None
    assert away.team.id == 20
    service._append_missing_mapping_key.assert_called_once_with("99999999", "Unknown FC")
    service.resolver.resolve_team.assert_called_once_with(
        provider_team_id="99999999",
        provider_team_name="Unknown FC",
        league_id=1,
    )


def test_fetch_and_store_matches_for_date_imports_known_league() -> None:
    session = MagicMock()
    service = ExtendedMatchDataService(provider="fotmob", session=session)
    fixture = ProviderMatch(
        provider_match_id="1001",
        provider_league_id="47",
        provider_season_id=None,
        home_team_id="8456",
        away_team_id="10260",
        home_team_name="Arsenal",
        away_team_name="Chelsea",
        kickoff_at=datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc),
        status="scheduled",
    )
    service.provider.fetch_matches_by_date = MagicMock(return_value=[fixture])
    service.leagues_repo.get_by_external_id = MagicMock(
        return_value=SimpleNamespace(id=1, external_id=39)
    )
    service._import_provider_fixture = MagicMock(
        return_value=SimpleNamespace(status="imported")
    )

    result = service.fetch_and_store_matches_for_date(date(2024, 8, 17))

    assert result.requested == 1
    assert result.imported == 1
    service.provider.fetch_matches_by_date.assert_called_once_with(date(2024, 8, 17))
    service.leagues_repo.get_by_external_id.assert_called_once_with(39)
