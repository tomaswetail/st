from __future__ import annotations

from typing import Any

from data_sources.football_data_uk_xlsx_provider import (
    FTR_MAP,
    start_year_to_season_code,
)
from objects.schema.db.historical_match import HistoricalMatchDraft


def normalize_season_code(season: str) -> str:
    """Normalize soccerdata season values to football-data season codes."""
    text = str(season).strip()
    if len(text) == 4 and text.isdigit():
        start = int(text[:2])
        end = int(text[2:])
        if end == (start + 1) % 100:
            return text

        year = int(text)
        if year >= 1900:
            return start_year_to_season_code(year)

    return text


def to_historical_match_create(
    match: dict[str, Any],
    *,
    league_code: str,
) -> HistoricalMatchDraft:
    result = FTR_MAP[match["result"]]
    return HistoricalMatchDraft(
        source=match["source"],
        league=league_code,
        season=normalize_season_code(str(match["season"])),
        match_date=match["match_date"],
        home_team=match["home_team"],
        away_team=match["away_team"],
        home_goals=match["home_goals"],
        away_goals=match["away_goals"],
        result=result,
        odds_home=match.get("odds_home"),
        odds_draw=match.get("odds_draw"),
        odds_away=match.get("odds_away"),
        raw_data=match.get("raw_data"),
    )
