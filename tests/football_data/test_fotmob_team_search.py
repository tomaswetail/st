"""Tests for FotMob team search parsing and resolver wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.data_sources.football_data.fotmob_entity_resolver import FotMobEntityResolver
from src.data_sources.football_data.providers.fotmob import (
    FotMobProvider,
    parse_fotmob_team_search,
)
from src.objects.schema.data_classes.provider_dtos import ProviderTeam


def test_parse_fotmob_team_search_nested_suggestions():
    payload = [
        {
            "title": {"key": "all", "value": "All"},
            "suggestions": [
                {
                    "type": "team",
                    "id": "9825",
                    "name": "Arsenal",
                    "leagueId": 47,
                },
                {
                    "type": "match",
                    "id": "1",
                    "homeTeamName": "Arsenal",
                    "awayTeamName": "Chelsea",
                },
                {
                    "type": "team",
                    "id": "258657",
                    "name": "Arsenal (W)",
                },
            ],
        }
    ]

    teams = parse_fotmob_team_search(payload)

    assert [(t.provider_team_id, t.name) for t in teams] == [
        ("9825", "Arsenal"),
        ("258657", "Arsenal (W)"),
    ]


def test_search_teams_calls_suggest_endpoint():
    client = MagicMock()
    client.get_json.return_value = [
        {
            "suggestions": [
                {"type": "team", "id": "9825", "name": "Arsenal"},
            ]
        }
    ]
    provider = FotMobProvider(client=client)

    teams = provider.search_teams("Arsenal", hits=10)

    client.get_json.assert_called_once_with(
        "search/suggest",
        params={"term": "Arsenal", "hits": 10, "lang": "en"},
    )
    assert teams[0].provider_team_id == "9825"
    assert teams[0].name == "Arsenal"


def test_resolve_team_falls_back_to_search():
    provider = MagicMock()
    provider.search_teams.return_value = [
        ProviderTeam(provider_team_id="999", name="Unknown FC"),
    ]
    session = MagicMock()
    config = SimpleNamespace(
        fuzzy_match_threshold=85,
        missing_teams_csv_path="/tmp/missing_teams_test.csv",
    )

    with patch(
        "src.data_sources.football_data.fotmob_entity_resolver._get_team_names",
        return_value=["Arsenal", "Chelsea"],
    ), patch(
        "src.data_sources.football_data.fotmob_entity_resolver._load_aliases",
        return_value={},
    ):
        resolver = FotMobEntityResolver(
            session=session,
            config=config,  # type: ignore[arg-type]
            provider=provider,
        )

    result = resolver.resolve_team("Unknown FC", team_external_id=123456789)

    assert result == 999
    provider.search_teams.assert_called_once_with("Unknown FC")


def test_resolve_team_uses_static_api_to_fotmob_mapping():
    provider = MagicMock()
    session = MagicMock()
    config = SimpleNamespace(
        fuzzy_match_threshold=85,
        missing_teams_csv_path="/tmp/missing_teams_test.csv",
    )

    with patch(
        "src.data_sources.football_data.fotmob_entity_resolver._get_team_names",
        return_value=["Arsenal", "Chelsea"],
    ), patch(
        "src.data_sources.football_data.fotmob_entity_resolver._load_aliases",
        return_value={},
    ), patch.dict(
        "src.data_sources.football_data.fotmob_entity_resolver._API_FOOTBALL_TO_FOTMOB_TEAMS",
        {46: 8197},
        clear=False,
    ):
        resolver = FotMobEntityResolver(
            session=session,
            config=config,  # type: ignore[arg-type]
            provider=provider,
        )
        result = resolver.resolve_team("Leicester", team_external_id=46)

    assert result == 8197
    provider.search_teams.assert_not_called()
