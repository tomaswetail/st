"""Tests for provider league catalogue discovery and mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from data_sources.football_data.league_catalogue import LeagueCatalogueService
from data_sources.football_data.providers.sofascore import (
    parse_sofascore_available_leagues,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import ProviderLeague
from tests.football_data.conftest import load_fixture


def _sofascore_leagues() -> list[ProviderLeague]:
    categories = load_fixture("sofascore", "categories.json")
    tournaments = {
        "1": load_fixture("sofascore", "unique_tournaments_england.json"),
        "32": load_fixture("sofascore", "unique_tournaments_spain.json"),
    }
    return parse_sofascore_available_leagues(categories, tournaments)


def test_sofascore_available_leagues_parsing():
    leagues = _sofascore_leagues()
    by_id = {league.provider_league_id: league for league in leagues}
    assert by_id["17"].name == "Premier League"
    assert by_id["17"].country == "England"
    assert by_id["8"].name == "LaLiga"


def _catalogue_service(leagues: list[ProviderLeague]) -> LeagueCatalogueService:
    provider = MagicMock()
    provider.name = "sofascore"
    provider.fetch_available_leagues.return_value = leagues
    session = MagicMock()
    return LeagueCatalogueService(provider=provider, session=session)


def test_find_leagues_filters_by_query_and_country():
    leagues = _sofascore_leagues()
    service = _catalogue_service(leagues)
    found = service.find_leagues(query="Premier", country="England")
    assert any(league.provider_league_id == "17" for league in found)
    assert all(
        (league.country or "").startswith("England") or "Premier" in league.name
        for league in found
    )


def test_suggest_mappings_scores_internal_leagues():
    leagues = _sofascore_leagues()
    service = _catalogue_service(leagues)
    service.league_repo.get_all = MagicMock(
        return_value=[
            SimpleNamespace(id=1, league_name="England Premier League", country_name="England"),
            SimpleNamespace(id=2, league_name="Unknown Cup XYZ", country_name="Nowhere"),
        ]
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    suggestions = service.suggest_mappings()
    by_id = {item.internal_league_id: item for item in suggestions}
    assert by_id[1].candidate is not None
    assert by_id[1].candidate.provider_league_id == "17"
    assert by_id[1].method in {"exact", "fuzzy"}
    assert by_id[2].method == "unresolved"


def test_map_league_with_explicit_external_id():
    leagues = _sofascore_leagues()
    service = _catalogue_service(leagues)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=1, league_name="England Premier League", country_name="England")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(1, external_entity_id="17")
    assert result.status == "mapped"
    assert result.external_entity_id == "17"
    service.mapping_repo.upsert.assert_called_once()
    service.session.commit.assert_called_once()


def test_map_league_high_confidence_query():
    leagues = _sofascore_leagues()
    service = _catalogue_service(leagues)
    service.config = DataSourceConfig(fuzzy_match_threshold=85)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=1, league_name="England Premier League", country_name="England")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(1, query="Premier League")
    assert result.status == "mapped"
    assert result.external_entity_id == "17"


def test_map_league_unresolved_does_not_write():
    leagues = _sofascore_leagues()
    service = _catalogue_service(leagues)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=9, league_name="Mystery League", country_name="Atlantis")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(9, query="Completely Unknown Tournament")
    assert result.status == "unresolved"
    service.mapping_repo.upsert.assert_not_called()
    service.session.commit.assert_not_called()
