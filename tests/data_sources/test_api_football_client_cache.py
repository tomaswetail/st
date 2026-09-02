from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from data_sources.api_football_client import (
    API_FOOTBALL_SOURCE,
    APIFootballClient,
    fixture_to_create,
    get_all_leagues,
    season_to_api_year,
)
from objects.schema.data_classes.data_sources import (
    DISK_CACHE_TTL_ONE_YEAR,
    DataSourceConfig,
)
from objects.schema.data_classes.fixture import Fixture


def _sample_fixture(**overrides: Any) -> Fixture:
    base = dict(
        fixture_id=1,
        fixture_referee=None,
        fixture_timezone="UTC",
        fixture_date=datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc),
        fixture_timestamp=1723906800,
        period_first=None,
        period_second=None,
        venue_id=None,
        venue_name=None,
        venue_city=None,
        status_long="Match Finished",
        status_short="FT",
        league_id=39,
        league_name="Premier League",
        league_country="England",
        league_flag=None,
        league_season=2024,
        league_round="Regular Season - 1",
        home_team_id=42,
        home_team_name="Arsenal",
        home_team_winner=True,
        away_team_id=39,
        away_team_name="Wolves",
        away_team_winner=False,
        goals_home=2,
        goals_away=1,
        score_halftime_home=1,
        score_halftime_away=0,
        score_fulltime_home=2,
        score_fulltime_away=1,
        score_extratime_home=None,
        score_extratime_away=None,
        score_penalty_home=None,
        score_penalty_away=None,
    )
    base.update(overrides)
    return Fixture(**base)


def test_api_football_cache_ttl_default_one_year() -> None:
    config = DataSourceConfig()
    assert config.football_data_cache_ttl_seconds == DISK_CACHE_TTL_ONE_YEAR


def test_shared_api_cache_root_under_data_cache() -> None:
    """FotMob / API-Football / Svenska Spel share repo data/cache/."""
    from objects.schema.data_classes.svenska_spel_config import SvenskaSpelConfig

    config = DataSourceConfig()
    ss = SvenskaSpelConfig()
    cache_root = config.football_data_cache_dir.resolve()
    assert cache_root.name == "cache"
    assert cache_root.parent.name == "data"
    assert (cache_root / API_FOOTBALL_SOURCE) == (
        cache_root / "api-football"
    )
    assert ss.cache_dir.resolve() == (cache_root / "svenskaspel").resolve()
    assert (cache_root / "fotmob").parent == cache_root


def test_season_to_api_year_accepts_yyxx_and_calendar() -> None:
    assert season_to_api_year("2425") == 2024
    assert season_to_api_year("2024") == 2024
    assert season_to_api_year(2023) == 2023


def test_fixture_to_create_maps_api_fields() -> None:
    draft = fixture_to_create(_sample_fixture())
    assert draft is not None
    assert draft.fixture_id == 1
    assert draft.league_id == 39
    assert draft.league_season == 2024
    assert draft.home_team_name == "Arsenal"
    assert draft.goals_home == 2
    assert draft.status_short == "FT"


def test_api_football_client_disk_cache_hit(tmp_path: Path) -> None:
    config = DataSourceConfig(football_data_cache_dir=tmp_path)
    client = APIFootballClient(
        api_key="test-key",
        config=config,
        cache_dir=tmp_path / "api-football",
    )

    response_payload: dict[str, Any] = {"ok": True, "response": {"x": 1}}
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=response_payload)

    with patch("data_sources.api_football_client.requests.get", return_value=response) as get_mock:
        first = client.get("fixtures", {"league": 1})
        second = client.get("fixtures", {"league": 1})

    assert first == response_payload
    assert second == response_payload
    assert get_mock.call_count == 1
    assert client.cache_ttl_seconds == DISK_CACHE_TTL_ONE_YEAR


def test_get_all_leagues_normalizes_response() -> None:
    client = MagicMock()
    client.get.return_value = {
        "response": [
            {
                "league": {
                    "id": 39,
                    "name": "Premier League",
                    "type": "League",
                    "logo": "https://example/logo.png",
                },
                "country": {"name": "England", "code": "GB"},
                "seasons": [{"year": 2024}, {"year": 2025}],
            }
        ]
    }
    leagues = get_all_leagues(client, country="England", season="2425")
    assert len(leagues) == 1
    assert leagues[0].league_id == 39
    assert leagues[0].league_name == "Premier League"
    assert leagues[0].country_name == "England"
    assert leagues[0].country_code == "GB"
    assert leagues[0].seasons == [2024, 2025]
    client.get.assert_called_once_with(
        "leagues", {"country": "England", "season": 2024}
    )
