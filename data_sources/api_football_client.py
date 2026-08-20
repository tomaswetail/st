"""API-Football v3 client for leagues, teams, fixtures, and odds."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.fixture import Fixture
from objects.schema.data_classes.league_info import LeagueInfo
from objects.schema.db.fixture import FixtureCreate
from objects.schema.db.league import LeagueCreate

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_SOURCE = "api-football"


def season_to_api_year(season: str | int) -> int:
    """Normalize YYXX or calendar year to API-Football season start year."""
    text = str(season).strip()
    if text.isdigit() and len(text) == 4:
        start_yy = int(text[:2])
        end_yy = int(text[2:])
        if (end_yy - start_yy) % 100 == 1:
            return 2000 + start_yy
        return int(text)
    if isinstance(season, int):
        return season
    raise ValueError(f"Unsupported season code: {season!r}")


class APIFootballClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: DataSourceConfig | None = None,
        enable_cache: bool = True,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.config = config or DataSourceConfig()
        self.api_key = api_key or self.config.api_football_key
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set API_FOOTBALL_KEY environment variable."
            )

        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else self.config.football_data_cache_ttl_seconds
        )
        self.cache_dir = cache_dir or (
            self.config.football_data_cache_dir / API_FOOTBALL_SOURCE
        )
        self.enable_cache = enable_cache and self.cache_ttl_seconds > 0
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


    def _cache_key(self, endpoint: str, params: dict[str, Any] | None) -> str:
        params_obj = params or {}
        raw = f"{BASE_URL}/{endpoint.lstrip('/')}|{repr(sorted(params_obj.items()))}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.enable_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > self.cache_ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, key: str, data: dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write API-Football cache %s: %s", path, exc)

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cache_key = self._cache_key(endpoint, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        response = requests.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            headers={"x-apisports-key": self.api_key},
            params=params or {},
            timeout=20,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning(
                "API-Football request failed: %s %s",
                response.status_code,
                response.text,
            )
            if response.status_code == 429:
                time.sleep(5)
                return self.get(endpoint, params)
            raise RuntimeError(
                f"API-Football request failed: {response.status_code} {response.text}"
            ) from exc

        data = response.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(f"API-Football returned errors: {data['errors']}")

        self._write_cache(cache_key, data)
        return data


def normalize_fixture(raw: dict[str, Any]) -> Fixture:
    fixture = raw["fixture"]
    league = raw["league"]
    teams = raw["teams"]
    goals = raw.get("goals") or {}
    score = raw.get("score") or {}
    periods = fixture.get("periods") or {}
    venue = fixture.get("venue") or {}
    status = fixture.get("status") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    halftime = score.get("halftime") or {}
    fulltime = score.get("fulltime") or {}
    extratime = score.get("extratime") or {}
    penalty = score.get("penalty") or {}

    fixture_date = datetime.fromisoformat(fixture["date"])
    return Fixture(
        fixture_id=fixture["id"],
        fixture_referee=fixture.get("referee"),
        fixture_timezone=fixture.get("timezone") or "UTC",
        fixture_date=fixture_date,
        fixture_timestamp=int(fixture.get("timestamp") or fixture_date.timestamp()),
        period_first=periods.get("first"),
        period_second=periods.get("second"),
        venue_id=venue.get("id"),
        venue_name=venue.get("name"),
        venue_city=venue.get("city"),
        status_long=str(status.get("long") or status.get("short") or ""),
        status_short=str(status.get("short") or ""),
        league_id=league["id"],
        league_name=league["name"],
        league_country=league.get("country"),
        league_flag=league.get("flag"),
        league_season=int(league["season"]),
        league_round=league.get("round"),
        home_team_id=home["id"],
        home_team_name=home["name"],
        home_team_winner=home.get("winner"),
        away_team_id=away["id"],
        away_team_name=away["name"],
        away_team_winner=away.get("winner"),
        goals_home=goals.get("home"),
        goals_away=goals.get("away"),
        score_halftime_home=halftime.get("home"),
        score_halftime_away=halftime.get("away"),
        score_fulltime_home=fulltime.get("home"),
        score_fulltime_away=fulltime.get("away"),
        score_extratime_home=extratime.get("home"),
        score_extratime_away=extratime.get("away"),
        score_penalty_home=penalty.get("home"),
        score_penalty_away=penalty.get("away"),
    )


def normalize_league(raw: dict[str, Any]) -> LeagueInfo:
    league = raw["league"]
    country = raw["country"]
    seasons = raw.get("seasons", [])
    return LeagueInfo(
        league_id=league["id"],
        league_name=league["name"],
        league_type=str(league.get("type") or "League"),
        country_name=country["name"],
        country_code=country.get("code"),
        seasons=[season["year"] for season in seasons],
    )


def fixture_to_create(fixture: Fixture) -> FixtureCreate | None:
    """Map a normalized API fixture to a DB create schema.

    Skips fixtures without a final/full-time score when both goals are missing.
    """
    if fixture.goals_home is None and fixture.goals_away is None:
        if fixture.status_short not in {"FT", "AET", "PEN"}:
            return None
    return FixtureCreate(
        fixture_id=fixture.fixture_id,
        fixture_referee=fixture.fixture_referee,
        fixture_timezone=fixture.fixture_timezone,
        fixture_date=fixture.fixture_date,
        fixture_timestamp=fixture.fixture_timestamp,
        period_first=fixture.period_first,
        period_second=fixture.period_second,
        venue_id=fixture.venue_id,
        venue_name=fixture.venue_name,
        venue_city=fixture.venue_city,
        status_long=fixture.status_long,
        status_short=fixture.status_short,
        league_id=fixture.league_id,
        league_name=fixture.league_name,
        league_country=fixture.league_country,
        league_flag=fixture.league_flag,
        league_season=fixture.league_season,
        league_round=fixture.league_round,
        home_team_id=fixture.home_team_id,
        home_team_name=fixture.home_team_name,
        home_team_winner=fixture.home_team_winner,
        away_team_id=fixture.away_team_id,
        away_team_name=fixture.away_team_name,
        away_team_winner=fixture.away_team_winner,
        goals_home=fixture.goals_home,
        goals_away=fixture.goals_away,
        score_halftime_home=fixture.score_halftime_home,
        score_halftime_away=fixture.score_halftime_away,
        score_fulltime_home=fixture.score_fulltime_home,
        score_fulltime_away=fixture.score_fulltime_away,
        score_extratime_home=fixture.score_extratime_home,
        score_extratime_away=fixture.score_extratime_away,
        score_penalty_home=fixture.score_penalty_home,
        score_penalty_away=fixture.score_penalty_away,
    )


def league_info_to_create(league: LeagueInfo) -> LeagueCreate:
    return LeagueCreate(
        external_id=league.league_id,
        league_name=league.league_name,
        league_type=league.league_type,
        country_name=league.country_name,
        country_code=league.country_code,
    )


def get_all_leagues(
    client: APIFootballClient,
    *,
    country: str | None = None,
    season: str | int | None = None,
    league_type: str | None = None,
) -> list[LeagueInfo]:
    """Fetch leagues from API-Football ``GET /leagues``."""
    params: dict[str, Any] = {}
    if country:
        params["country"] = country
    if season is not None:
        params["season"] = season_to_api_year(season)
    if league_type:
        params["type"] = league_type

    data = client.get("leagues", params or None)
    return [normalize_league(item) for item in data.get("response") or []]



def get_team_names_by_league(
    client: APIFootballClient,
    league_id: int,
    season: str | int,
) -> list[tuple[str, str]]:
    """Return (team_id, name) for a league season via GET teams?league=&season=."""
    data = client.get(
        "teams",
        {"league": league_id, "season": season_to_api_year(season)},
    )
    teams: list[tuple[str, str]] = []
    for item in data.get("response") or []:
        team = item.get("team") or {}
        name = team.get("name")
        if not name:
            continue
        team_name = str(name).strip()
        team_id = team.get("id")
        external_id = str(team_id) if team_id is not None else team_name
        teams.append((external_id, team_name))
    return teams


def normalize_name(name: str) -> str:
    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "Å": "A",
        "Ä": "A",
        "Ö": "O",
    }
    name = "".join(replacements.get(char, char) for char in name)
    return (
        name.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
        .strip()
    )


def get_fixtures_by_league(
    client: APIFootballClient,
    league_id: int,
    season: str | int,
) -> list[Fixture]:
    try:
        data = client.get(
            "fixtures",
            {
                "league": league_id,
                "season": season_to_api_year(season),
            },
        )
    except RuntimeError:
        return []
    return [normalize_fixture(item) for item in data.get("response") or []]


def get_fixtures_by_league_and_date(
    client: APIFootballClient,
    league_id: int,
    season: int,
    match_date: str,
) -> list[Fixture]:
    data = client.get(
        "fixtures",
        {
            "league": league_id,
            "season": season,
            "date": match_date,
        },
    )
    return [normalize_fixture(item) for item in data["response"]]


def get_fixture_by_id(
    client: APIFootballClient,
    fixture_id: int,
) -> Fixture | None:
    data = client.get("fixtures", {"id": fixture_id})
    response = data["response"]
    if not response:
        return None
    return normalize_fixture(response[0])
