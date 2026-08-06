"""FotMob leagues endpoint uses id + season + ccode3."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_sources.football_data.providers.fotmob import FotMobProvider, parse_fotmob_team
from objects.schema.data_classes.data_sources import DataSourceConfig


def test_fotmob_fetch_season_matches_sends_ccode3():
    client = MagicMock()
    client.get_json.return_value = {"matches": []}
    provider = FotMobProvider(client=client, config=DataSourceConfig())

    provider.fetch_season_matches("47", "2025/2026", country_code="ENG")

    assert client.get_json.call_count == 1
    args, kwargs = client.get_json.call_args
    assert args[0] == "leagues"
    assert kwargs["params"]["id"] == "47"
    assert kwargs["params"]["ccode3"] == "ENG"


def test_fotmob_fetch_season_matches_requires_ccode3():
    provider = FotMobProvider(client=MagicMock(), config=DataSourceConfig())
    with pytest.raises(ValueError, match="ccode3"):
        provider.fetch_season_matches("47", "2025/2026")


def test_fotmob_fetch_team_calls_api_teams_endpoint():
    client = MagicMock()
    client.get_json.return_value = {
        "details": {
            "id": 8456,
            "name": "Man City",
            "shortName": "MCI",
            "country": "ENG",
        }
    }
    provider = FotMobProvider(client=client, config=DataSourceConfig())
    team = provider.fetch_team("8456")

    assert team.provider_team_id == "8456"
    assert team.name == "Man City"
    assert team.short_name == "MCI"
    assert team.country_code == "ENG"
    args, kwargs = client.get_json.call_args
    assert args[0].endswith("/teams")
    assert "/data" not in args[0].rstrip("/teams")
    assert kwargs["params"] == {"id": "8456"}


def test_parse_fotmob_team_requires_id_and_name():
    with pytest.raises(ValueError):
        parse_fotmob_team({"details": {"id": 1}})
