"""Tests for provider league catalogue discovery and mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_sources.football_data.league_catalogue import LeagueCatalogueService
from data_sources.football_data.providers.fotmob import parse_fotmob_all_leagues
from data_sources.football_data.providers.sofascore import (
    parse_sofascore_available_leagues,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import ProviderLeague
from tests.football_data.conftest import load_fixture


def test_fotmob_all_leagues_parsing():
    payload = load_fixture("fotmob", "all_leagues.json")
    leagues = parse_fotmob_all_leagues(payload)
    by_id = {league.provider_league_id: league for league in leagues}
    assert "47" in by_id
    assert by_id["47"].name == "Premier League"
    assert by_id["47"].country_code == "ENG"
    assert "48" in by_id
    assert "87" in by_id
    assert "42" in by_id
    # popular + countries should not duplicate Premier League
    assert sum(1 for league in leagues if league.provider_league_id == "47") == 1


def test_sofascore_available_leagues_parsing():
    categories = load_fixture("sofascore", "categories.json")
    tournaments = {
        "1": load_fixture("sofascore", "unique_tournaments_england.json"),
        "32": load_fixture("sofascore", "unique_tournaments_spain.json"),
    }
    leagues = parse_sofascore_available_leagues(categories, tournaments)
    by_id = {league.provider_league_id: league for league in leagues}
    assert by_id["17"].name == "Premier League"
    assert by_id["17"].country == "England"
    assert by_id["8"].name == "LaLiga"


def _catalogue_service(leagues: list[ProviderLeague]) -> LeagueCatalogueService:
    provider = MagicMock()
    provider.name = "fotmob"
    provider.fetch_available_leagues.return_value = leagues
    session = MagicMock()
    service = LeagueCatalogueService(provider=provider, session=session)
    return service


def test_find_leagues_filters_by_query_and_country():
    leagues = parse_fotmob_all_leagues(load_fixture("fotmob", "all_leagues.json"))
    service = _catalogue_service(leagues)
    found = service.find_leagues(query="Premier", country="England")
    assert any(league.provider_league_id == "47" for league in found)
    assert all(
        league.country_code == "ENG" or (league.country or "").startswith("England")
        or "Premier" in league.name
        for league in found
    )


def test_suggest_mappings_scores_internal_leagues():
    leagues = parse_fotmob_all_leagues(load_fixture("fotmob", "all_leagues.json"))
    service = _catalogue_service(leagues)
    service.league_repo.get_all = MagicMock(
        return_value=[
            SimpleNamespace(id=1, name="England Premier League", country="England"),
            SimpleNamespace(id=2, name="Unknown Cup XYZ", country="Nowhere"),
        ]
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    suggestions = service.suggest_mappings()
    by_id = {item.internal_league_id: item for item in suggestions}
    assert by_id[1].candidate is not None
    assert by_id[1].candidate.provider_league_id == "47"
    assert by_id[1].method in {"exact", "fuzzy"}
    assert by_id[2].method == "unresolved"


def test_map_league_with_explicit_external_id():
    leagues = parse_fotmob_all_leagues(load_fixture("fotmob", "all_leagues.json"))
    service = _catalogue_service(leagues)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=1, name="England Premier League", country="England")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(1, external_entity_id="47")
    assert result.status == "mapped"
    assert result.external_entity_id == "47"
    service.mapping_repo.upsert.assert_called_once()
    service.session.commit.assert_called_once()


def test_map_league_high_confidence_query():
    leagues = parse_fotmob_all_leagues(load_fixture("fotmob", "all_leagues.json"))
    service = _catalogue_service(leagues)
    service.config = DataSourceConfig(fuzzy_match_threshold=85)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=1, name="England Premier League", country="England")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(1, query="Premier League")
    assert result.status == "mapped"
    assert result.external_entity_id == "47"


def test_map_league_unresolved_does_not_write():
    leagues = parse_fotmob_all_leagues(load_fixture("fotmob", "all_leagues.json"))
    service = _catalogue_service(leagues)
    service.league_repo.get = MagicMock(
        return_value=SimpleNamespace(id=9, name="Mystery League", country="Atlantis")
    )
    service.mapping_repo.get_by_internal = MagicMock(return_value=None)
    service.mapping_repo.upsert = MagicMock()
    result = service.map_league(9, query="Completely Unknown Tournament")
    assert result.status == "unresolved"
    service.mapping_repo.upsert.assert_not_called()
    service.session.commit.assert_not_called()


def test_map_leagues_from_all_leagues_csv(tmp_path):
    csv_path = tmp_path / "all_leagues.csv"
    csv_path.write_text(
        "ccode,country,id,name,page_url\n"
        "ENG,England,47,Premier League,/leagues/47\n"
        "SWE,Sweden,67,Allsvenskan,/leagues/67\n"
        "XXX,Nowhere,999,Unknown Cup,/leagues/999\n",
        encoding="latin-1",
    )
    service = _catalogue_service([])
    premier = SimpleNamespace(id=1, name="Premier League", country="England")
    allsvenskan = SimpleNamespace(id=2, name="Allsvenskan", country="Sweden")

    def lookup(name: str, country: str):
        if name == "Premier League" and country == "England":
            return premier
        if name == "Allsvenskan" and country == "Sweden":
            return allsvenskan
        return None

    service.league_repo.get_by_name_and_country = MagicMock(side_effect=lookup)
    service.mapping_repo.upsert = MagicMock()

    counts = service.map_leagues_from_all_leagues_csv(csv_path)

    assert counts == {"mapped": 2, "unmatched": 1}
    assert service.mapping_repo.upsert.call_count == 2
    service.mapping_repo.upsert.assert_any_call(
        provider="fotmob",
        entity_type="league",
        internal_entity_id=1,
        external_entity_id="47",
        external_name="Premier League",
        metadata={"ccode": "ENG", "country": "England"},
    )
    service.session.commit.assert_called_once()

    service.mapping_repo.upsert.reset_mock()
    service.session.commit.reset_mock()
    dry_counts = service.map_leagues_from_all_leagues_csv(csv_path, dry_run=True)
    assert dry_counts == {"mapped": 2, "unmatched": 1}
    service.mapping_repo.upsert.assert_not_called()
    service.session.commit.assert_not_called()
