"""FotMob leagues endpoint uses id + season + ccode3."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_sources.football_data.providers.fotmob import FotMobProvider
from objects.schema.data_classes.data_sources import DataSourceConfig


def test_fotmob_fetch_season_matches_sends_ccode3():
    client = MagicMock()
    client.get_json.return_value = {"matches": []}
    provider = FotMobProvider(client=client, config=DataSourceConfig())

    provider.fetch_season_matches("47", "2025/2026", country_code="ENG")

    client.get_json.assert_called_once_with(
        "leagues",
        params={"id": "47", "season": "2025/2026", "ccode3": "ENG"},
    )


def test_fotmob_fetch_season_matches_requires_ccode3():
    provider = FotMobProvider(client=MagicMock(), config=DataSourceConfig())
    with pytest.raises(ValueError, match="ccode3"):
        provider.fetch_season_matches("47", "2025/2026")
