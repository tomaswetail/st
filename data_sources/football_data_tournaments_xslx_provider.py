"""Football-Data.co.uk World Cup XLSX tournament provider."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.historical_match import HistoricalMatchCreate

logger = logging.getLogger(__name__)

TOURNAMENT_SHEET_CODES: dict[str, str] = {
    "WorldCup2022": "WC2022",
    "WorldCup2018": "WC2018",
    "WorldCup2014": "WC2014",
    "WorldCup2026Qualifiers": "WC2026Q",
}

TOURNAMENT_SEASONS: dict[str, str] = {
    "WC2022": "2022",
    "WC2018": "2018",
    "WC2014": "2014",
    "WC2026Q": "2026",
}

TOURNAMENT_CODE_TO_SHEET: dict[str, str] = {
    code: sheet for sheet, code in TOURNAMENT_SHEET_CODES.items()
}


def all_tournament_codes() -> list[str]:
    return sorted(TOURNAMENT_SEASONS.keys())


def is_tournament_code(code: str) -> bool:
    return code.upper() in TOURNAMENT_SEASONS


def _parse_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _derive_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "1"
    if home_goals == away_goals:
        return "X"
    return "2"


def _extract_tournament_odds(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Prefer average, then max, then bet365 odds from tournament XLSX."""
    for h_col, d_col, a_col in (
        ("H-Avg", "D-Avg", "A-Avg"),
        ("H-Max", "D-Max", "A-Max"),
        ("bet365-H", "bet365-D", "bet365-A"),
    ):
        h, d, a = (
            _parse_float(row.get(h_col)),
            _parse_float(row.get(d_col)),
            _parse_float(row.get(a_col)),
        )
        if h and d and a:
            return h, d, a
    return None, None, None


def parse_tournament_sheet(
    df: pd.DataFrame,
    *,
    league: str,
    season: str,
    source: str = "football-data.co.uk",
) -> tuple[list[HistoricalMatchCreate], list[str]]:
    """Parse a tournament worksheet into HistoricalMatch rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    matches: list[HistoricalMatchCreate] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            home = row.get("Home")
            away = row.get("Away")
            date_val = row.get("Date")
            if pd.isna(home) or pd.isna(away) or pd.isna(date_val):
                errors.append(f"Row {idx}: missing home, away, or date")
                continue
            home_team = str(home).strip()
            away_team = str(away).strip()
            if not home_team or not away_team:
                errors.append(f"Row {idx}: missing home, away, or date")
                continue

            hg_val = row.get("HGFT")
            ag_val = row.get("AGFT")
            if pd.isna(hg_val) or pd.isna(ag_val):
                continue

            home_goals = int(hg_val)
            away_goals = int(ag_val)
            match_date = pd.to_datetime(date_val, dayfirst=True).date()
            oh, od, oa = _extract_tournament_odds(row)
            raw = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            matches.append(
                HistoricalMatchCreate(
                    source=source,
                    league=league,
                    season=season,
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    result=_derive_result(home_goals, away_goals),  # type: ignore[arg-type]
                    odds_home=oh,
                    odds_draw=od,
                    odds_away=oa,
                    raw_data={str(k): raw[k] for k in raw},
                )
            )
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
    return matches, errors


class FootballDataTournamentProvider:
    """Download and parse Football-Data.co.uk World Cup XLSX files."""

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        self.config = DataSourceConfig()

    @property
    def _local_xlsx_path(self) -> Path:
        return (
            self.config.raw_football_data_dir
            / "tournaments"
            / "WorldCup2026.xlsx"
        )

    def download_xlsx(self) -> bytes:
        """Download World Cup XLSX bytes."""

        url = self.config.football_data_world_cup_xlsx_url
        try:
            import httpx

            resp = httpx.get(url, timeout=60.0, follow_redirects=True)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            local = self._local_xlsx_path
            if local.exists():
                logger.info("Using local XLSX %s", local)
                content = local.read_bytes()
            else:
                raise RuntimeError(
                    f"Failed to download {url} and no local file at {local}: {exc}"
                ) from exc

        dest = self._local_xlsx_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return content

    def _log_parse_errors(self, league: str, errors: list[str]) -> None:
        for err in errors[:5]:
            logger.warning("%s: %s", league, err)
        if len(errors) > 5:
            logger.warning("%s: %d more row errors skipped", league, len(errors) - 5)

    def fetch_historical_matches(self, tournaments: list[str]) -> list[HistoricalMatchCreate]:
        content = self.download_xlsx()
        workbook = pd.read_excel(io.BytesIO(content), sheet_name=None)
        all_matches: list[HistoricalMatchCreate] = []

        for code in tournaments:
            sheet_name = TOURNAMENT_CODE_TO_SHEET.get(code)
            if not sheet_name:
                raise ValueError(f"Unknown tournament code: {code}")
            if sheet_name not in workbook:
                raise ValueError(
                    f"Sheet {sheet_name!r} not found in XLSX for tournament {code}"
                )

            season = TOURNAMENT_SEASONS[code]
            logger.info("Parsing tournament %s from sheet %s", code, sheet_name)
            matches, errors = parse_tournament_sheet(
                workbook[sheet_name],
                league=code,
                season=season,
            )
            self._log_parse_errors(code, errors)
            all_matches.extend(matches)

        return all_matches
