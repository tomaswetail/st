"""Load API-Football league_code → league_id map from config JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from objects.schema.data_classes.data_sources import DataSourceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiFootballLeagueEntry:
    code: str
    league_id: int
    name: str
    country: str


def load_api_football_leagues(
    path: Path | None = None,
) -> dict[str, ApiFootballLeagueEntry]:
    """Return mapping of internal league code → API-Football league entry."""
    config_path = path or DataSourceConfig().api_football_leagues_path
    if not config_path.exists():
        logger.warning("API-Football leagues map missing at %s", config_path)
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Failed reading API-Football leagues map %s", config_path)
        return {}
    if not isinstance(raw, dict):
        return {}

    entries: dict[str, ApiFootballLeagueEntry] = {}
    for code, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        league_id = payload.get("league_id")
        name = payload.get("name")
        country = payload.get("country")
        if league_id is None or not name or not country:
            logger.warning("Skipping incomplete API-Football league entry %s", code)
            continue
        entries[str(code)] = ApiFootballLeagueEntry(
            code=str(code),
            league_id=int(league_id),
            name=str(name),
            country=str(country),
        )
    return entries


def code_for_api_football_league_id(
    league_id: int,
    leagues: dict[str, ApiFootballLeagueEntry] | None = None,
) -> str | None:
    """Return internal league code for an API-Football league id, if mapped."""
    mapping = leagues if leagues is not None else load_api_football_leagues()
    for code, entry in mapping.items():
        if entry.league_id == league_id:
            return code
    return None


def all_api_football_league_ids(
    leagues: dict[str, ApiFootballLeagueEntry] | None = None,
) -> list[int]:
    mapping = leagues if leagues is not None else load_api_football_leagues()
    return sorted({entry.league_id for entry in mapping.values()})


def all_api_football_league_codes(
    leagues: dict[str, ApiFootballLeagueEntry] | None = None,
) -> list[str]:
    mapping = leagues if leagues is not None else load_api_football_leagues()
    return sorted(mapping.keys())
