import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

import requests

from data_sources.football_data_uk_xlsx_provider import start_year_to_season_code, logger
from objects.schema.data_classes.fixture import Fixture
from objects.schema.data_classes.league_info import LeagueInfo
from objects.schema.db.historical_match import HistoricalMatchCreate
from utils.common import LEAGUE_MAPPINGS, get_season_rev

BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_SOURCE = "api-football"


# ---------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------

class APIFootballClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", '6de75a404b5c1c996418d07d6ac70144')

        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set API_FOOTBALL_KEY environment variable."
            )

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"

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
            raise RuntimeError(
                f"API-Football request failed: {response.status_code} {response.text}"
            ) from exc

        data = response.json()

        if "errors" in data and data["errors"]:
            raise RuntimeError(f"API-Football returned errors: {data['errors']}")

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
) -> HistoricalMatchCreate | None:
    if fixture.home_goals is None or fixture.away_goals is None:
        return None

    if fixture.home_goals > fixture.away_goals:
        result = "1"
    elif fixture.home_goals == fixture.away_goals:
        result = "X"
    else:
        result = "2"

    return HistoricalMatchCreate(
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


