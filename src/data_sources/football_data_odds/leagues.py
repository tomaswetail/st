"""League code / name lookup for football-data.co.uk ingest."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_sources.football_data_odds.constants import EXTRA_CODES, MAIN_CODES
from src.objects.schema.data_classes.data_sources import DataSourceConfig

COVERED_CODES = MAIN_CODES + EXTRA_CODES


def load_covered_leagues(
    leagues_path: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Load CSV code → {league_id, name, country} for codes this ingest covers."""
    path = leagues_path or DataSourceConfig().api_football_leagues_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    covered: dict[str, dict[str, object]] = {}
    for code in COVERED_CODES:
        entry = raw.get(code)
        if not isinstance(entry, dict):
            continue
        league_id = entry.get("league_id")
        if league_id is None:
            continue
        covered[code] = {
            "league_id": int(league_id),
            "name": str(entry.get("name") or code),
            "country": entry.get("country"),
        }
    return covered


def resolve_league_codes(
    league: str | None,
    *,
    leagues_path: Path | None = None,
) -> list[str]:
    """Accept a CSV code or our league name; None means all covered codes."""
    covered = load_covered_leagues(leagues_path)
    if league is None or not league.strip():
        return list(COVERED_CODES)
    token = league.strip()
    upper = token.upper()
    if upper in covered:
        return [upper]
    matches = [
        code
        for code, meta in covered.items()
        if str(meta["name"]).casefold() == token.casefold()
    ]
    if len(matches) == 1:
        return matches
    if len(matches) > 1:
        raise ValueError(
            f"League name {token!r} matches multiple codes: {', '.join(matches)}"
        )
    raise ValueError(f"Unknown league {token!r}; use a CSV code or mapped league name")
