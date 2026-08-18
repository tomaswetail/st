import time

import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any
from pathlib import Path

import requests

from data_sources.football_data_uk_xlsx_provider import start_year_to_season_code, logger
from objects.schema.data_classes.fixture import Fixture
from objects.schema.data_classes.league_info import LeagueInfo
from objects.schema.db.historical_match import HistoricalMatchDraft
from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.common import LEAGUE_COUNTRIES, LEAGUE_MAPPINGS, LEAGUE_NAMES, get_season_rev

BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_SOURCE = "api-football"


# ---------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------

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
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", '6de75a404b5c1c996418d07d6ac70144')

        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set API_FOOTBALL_KEY environment variable."
            )

        self.config = config or DataSourceConfig()
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else self.config.football_data_cache_ttl_seconds
        )
        self.cache_dir = (
            cache_dir
            or (self.config.football_data_cache_dir / API_FOOTBALL_SOURCE)
        )
        self.enable_cache = enable_cache and self.cache_ttl_seconds > 0
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, endpoint: str, params: dict[str, Any] | None) -> str:
        import hashlib

        # Deterministic key for endpoint + params.
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
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, key: str, data: dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        path = self._cache_path(key)
        try:
            import json

            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write API-Football cache %s: %s", path, exc)

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"

        cache_key = self._cache_key(endpoint, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        headers = {
            "x-apisports-key": self.api_key,
        }

        response = requests.get(
            url,
            headers=headers,
            params=params or {},
            timeout=20,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning( f"API-Football request failed: {response.status_code} {response.text}")

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


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_fixture(raw: dict[str, Any]) -> Fixture:
    fixture = raw["fixture"]
    league = raw["league"]
    teams = raw["teams"]
    goals = raw.get("goals") or {}
    score = raw.get("score") or {}
    halftime = score.get("halftime") or {}

    return Fixture(
        fixture_id=fixture["id"],
        league_id=league["id"],
        league_name=league["name"],
        country=league["country"],
        season=league["season"],
        start_time=datetime.fromisoformat(fixture["date"]),
        home_team=teams["home"]["name"],
        away_team=teams["away"]["name"],
        home_team_id=teams["home"]["id"],
        away_team_id=teams["away"]["id"],
        status=fixture["status"]["short"],

        home_goals=goals.get("home"),
        away_goals=goals.get("away"),
        halftime_home_goals=halftime.get("home"),
        halftime_away_goals=halftime.get("away"),
    )


def normalize_league(raw: dict[str, Any]) -> LeagueInfo:
    league = raw["league"]
    country = raw["country"]
    seasons = raw.get("seasons", [])

    return LeagueInfo(
        league_id=league["id"],
        name=league["name"],
        country=country["name"],
        league_type=league["type"],
        seasons=[season["year"] for season in seasons],
    )


def league_code_for_fixture(fixture: Fixture) -> str:
    normalized = normalize_name(fixture.league_name)
    for league_name, code in LEAGUE_MAPPINGS.items():
        if normalize_name(league_name) == normalized:
            return code
        if normalize_name(league_name) in normalized:
            return code
    return f"AF{fixture.league_id}"


def fixture_to_historical_match_create(
    fixture: Fixture,
) -> HistoricalMatchDraft | None:
    if fixture.home_goals is None or fixture.away_goals is None:
        return None

    if fixture.home_goals > fixture.away_goals:
        result = "1"
    elif fixture.home_goals == fixture.away_goals:
        result = "X"
    else:
        result = "2"

    return HistoricalMatchDraft(
        source=API_FOOTBALL_SOURCE,
        league=league_code_for_fixture(fixture),
        season=start_year_to_season_code(fixture.season),
        match_date=fixture.start_time.date(),
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        home_goals=fixture.home_goals,
        away_goals=fixture.away_goals,
        result=result,
        raw_data={
            "fixture_id": fixture.fixture_id,
            "league_id": fixture.league_id,
            "league_name": fixture.league_name,
            "country": fixture.country,
            "status": fixture.status,
            "fixture": asdict(fixture),
        },
    )


# ---------------------------------------------------------------------
# League discovery
# ---------------------------------------------------------------------

def get_sweden_leagues(client: APIFootballClient) -> list[LeagueInfo]:
    data = client.get("leagues", {"country": "Sweden"})
    return [normalize_league(item) for item in data["response"]]


def find_league_by_name(
    client: APIFootballClient,
    target_name: str,
) -> LeagueInfo | None:
    leagues = get_sweden_leagues(client)

    target = normalize_name(target_name)

    for league in leagues:
        if normalize_name(league.name) == target:
            return league

    for league in leagues:
        if target in normalize_name(league.name):
            return league

    return None


def league_search_names_for_code(league_code: str) -> list[str]:
    """Candidate API-Football search names for a football-data league code."""
    names: list[str] = []
    for mapping_name, mapping_code in LEAGUE_MAPPINGS.items():
        if mapping_code == league_code:
            names.append(mapping_name)
    full_name = LEAGUE_NAMES.get(league_code)
    if full_name:
        names.append(full_name)
        country = LEAGUE_COUNTRIES.get(league_code)
        if country and full_name.startswith(f"{country} "):
            names.append(full_name[len(country) + 1 :])
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    return unique_names


def find_league_for_code(
    client: APIFootballClient,
    league_code: str,
) -> LeagueInfo | None:
    """Resolve an API-Football league via GET leagues?name= for a football-data code."""
    for name in league_search_names_for_code(league_code):
        try:
            data = client.get("leagues", {"name": name})
        except RuntimeError:
            logger.warning(
                "API-Football league search failed for code=%s name=%s",
                league_code,
                name,
            )
            continue
        items = [normalize_league(item) for item in data.get("response") or []]
        if not items:
            continue
        league_type_matches = [
            item for item in items if item.league_type == "league"
        ]
        return (league_type_matches or items)[0]
    logger.warning("No API-Football league for football-data code %s", league_code)
    return None


def get_team_names_by_league(
    client: APIFootballClient,
    league_id: int,
    season: str,
) -> list[tuple[str, str]]:
    """Return (team_id, name) for a league season via GET teams?league=&season=."""
    data = client.get(
        "teams",
        {"league": league_id, "season": get_season_rev(season)},
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



# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def get_fixtures_by_league(
    client: APIFootballClient,
    league_id: int,
    season: str,
) -> list[Fixture]:

    try:
        data = client.get(
            "fixtures",
            {
                "league": league_id,
                "season": get_season_rev(season),
            },
        )
    except RuntimeError as e:
        return []

    return [normalize_fixture(item) for item in data["response"]]


def get_fixtures_by_league_and_date(
    client: APIFootballClient,
    league_id: int,
    season: int,
    match_date: str,
) -> list[Fixture]:
    """
    match_date format: YYYY-MM-DD
    """
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
    data = client.get(
        "fixtures",
        {
            "id": fixture_id,
        },
    )

    response = data["response"]

    if not response:
        return None

    return normalize_fixture(response[0])


# ---------------------------------------------------------------------
# Helpers for Superettan / Ettan Norra
# ---------------------------------------------------------------------

def get_superettan_league(client: APIFootballClient) -> LeagueInfo | None:
    return find_league_by_name(client, "Superettan")


def get_ettan_norra_league(client: APIFootballClient) -> LeagueInfo | None:
    # API-Football usually names it like "Ettan - Norra"
    league = find_league_by_name(client, "Ettan - Norra")

    if league:
        return league

    return find_league_by_name(client, "Ettan Norra")


def print_league_info(league: LeagueInfo | None, label: str) -> None:
    if league is None:
        print(f"{label}: not found")
        return

    print(f"{label}:")
    print(f"  id: {league.league_id}")
    print(f"  name: {league.name}")
    print(f"  country: {league.country}")
    print(f"  type: {league.league_type}")
    print(f"  seasons: {league.seasons[-8:]}")


def print_fixtures(fixtures: list[Fixture], limit: int = 20) -> None:
    for fixture in fixtures[:limit]:
        print(
            f"{fixture.fixture_id} | "
            f"{fixture.start_time.isoformat()} | "
            f"{fixture.league_name} | "
            f"{fixture.home_team} vs {fixture.away_team} | "
            f"{fixture.status}"
        )


