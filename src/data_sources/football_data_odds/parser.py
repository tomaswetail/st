"""Parse football-data.co.uk CSV rows into complete 1X2 odds triples."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any

from src.data_sources.football_data_odds.constants import (
    CLOSING_BOOK_COLUMNS,
    OPENING_BOOK_COLUMNS,
    PRICE_CLOSING,
    PRICE_OPENING,
)


@dataclass(frozen=True)
class OddsTriple:
    bookmaker: str
    price_type: str
    odds_home: float
    odds_draw: float
    odds_away: float


@dataclass
class ParsedMatchOdds:
    match_date: date
    kickoff_at: datetime
    home_team: str
    away_team: str
    source: str
    season: str
    triples: list[OddsTriple] = field(default_factory=list)
    raw_row: dict[str, str] = field(default_factory=dict)


def parse_odds_csv(
    text: str,
    *,
    source: str,
    season: str,
    closing_only: bool = False,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[ParsedMatchOdds]:
    """Parse a football-data.co.uk CSV into match rows with complete 1X2 triples."""
    reader = csv.DictReader(io.StringIO(_strip_bom(text)))
    if reader.fieldnames is None:
        return []
    parsed: list[ParsedMatchOdds] = []
    for raw in reader:
        row = {str(key).strip(): (value or "").strip() for key, value in raw.items() if key}
        home_team = _cell(row, "HomeTeam", "Home")
        away_team = _cell(row, "AwayTeam", "Away")
        match_date = parse_csv_date(_cell(row, "Date"))
        if not home_team or not away_team or match_date is None:
            continue
        if date_from is not None and match_date < date_from:
            continue
        if date_to is not None and match_date > date_to:
            continue
        kickoff_at = _kickoff_at(match_date, _cell(row, "Time"))
        row_season = _cell(row, "Season") or season
        triples = extract_odds_triples(row, closing_only=closing_only)
        parsed.append(
            ParsedMatchOdds(
                match_date=match_date,
                kickoff_at=kickoff_at,
                home_team=home_team,
                away_team=away_team,
                source=source,
                season=row_season,
                triples=triples,
                raw_row=row,
            )
        )
    return parsed


def extract_odds_triples(
    row: dict[str, str],
    *,
    closing_only: bool = False,
) -> list[OddsTriple]:
    """Emit complete 1X2 triples; skip incomplete or non-numeric books."""
    triples: list[OddsTriple] = []
    if not closing_only:
        triples.extend(_triples_from_spec(row, OPENING_BOOK_COLUMNS, PRICE_OPENING))
    triples.extend(_triples_from_spec(row, CLOSING_BOOK_COLUMNS, PRICE_CLOSING))
    return triples


def parse_csv_date(value: str) -> date | None:
    text = (value or "").strip()
    if not text:
        return None
    for date_format in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def parse_odds_value(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if parsed <= 1.0:
        return None
    return parsed


def _triples_from_spec(
    row: dict[str, str],
    spec: tuple[tuple[str, str, str, str], ...],
    price_type: str,
) -> list[OddsTriple]:
    triples: list[OddsTriple] = []
    for bookmaker, home_col, draw_col, away_col in spec:
        odds_home = parse_odds_value(row.get(home_col, ""))
        odds_draw = parse_odds_value(row.get(draw_col, ""))
        odds_away = parse_odds_value(row.get(away_col, ""))
        if odds_home is None or odds_draw is None or odds_away is None:
            continue
        triples.append(
            OddsTriple(
                bookmaker=bookmaker,
                price_type=price_type,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
            )
        )
    return triples


def _kickoff_at(match_date: date, time_text: str) -> datetime:
    parsed_time = _parse_time(time_text)
    return datetime.combine(match_date, parsed_time, tzinfo=timezone.utc)


def _parse_time(time_text: str) -> time:
    text = (time_text or "").strip()
    for time_format in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, time_format).time()
        except ValueError:
            continue
    return time(12, 0)


def _cell(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def _strip_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text.lstrip("\ufeff")
    return text


def odds_payload(row: dict[str, str]) -> dict[str, Any]:
    """CSV row as JSON-safe strings (full row is acceptable)."""
    return dict(row)
