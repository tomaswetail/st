"""Football-Data.co.uk CSV historical results and odds provider."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.historical_match import HistoricalMatchDraft
from utils.common import EXTRA_LEAGUE_CODES, MAIN_LEAGUE_CODES

logger = logging.getLogger(__name__)

FTR_MAP = {"H": "1", "D": "X", "A": "2"}


def season_code_to_start_year(season: str) -> int:
    """Convert football-data season code (e.g. 2324) to start calendar year (2023)."""
    return 2000 + int(season[:2])


def start_year_to_season_code(start_year: int) -> str:
    """Convert start calendar year (2023) to football-data season code (2324)."""
    end = start_year + 1
    return f"{start_year % 100:02d}{end % 100:02d}"


def parse_extra_season_start_year(season_val: Any) -> int | None:
    """Parse extra-league Season column (2012, 2012/2013, etc.) to start year."""
    if season_val is None or (isinstance(season_val, float) and pd.isna(season_val)):
        return None
    text = str(season_val).strip()
    if not text:
        return None
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    elif "-" in text and len(text) > 4:
        text = text.split("-", 1)[0].strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def is_extra_league(league: str) -> bool:
    return league.upper() in EXTRA_LEAGUE_CODES


def all_football_data_league_codes() -> list[str]:
    """Return all supported main and extra Football-Data league codes."""
    return sorted(MAIN_LEAGUE_CODES | EXTRA_LEAGUE_CODES)


def _parse_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _extract_odds(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Prefer Avg, then B365, then Max odds columns."""
    for prefix in ("Avg", "B365", "Max"):
        h, d, a = (
            _parse_float(row.get(f"{prefix}H")),
            _parse_float(row.get(f"{prefix}D")),
            _parse_float(row.get(f"{prefix}A")),
        )
        if h and d and a:
            return h, d, a
    return None, None, None


def _extract_extra_odds(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Prefer closing average odds from extra-league combined CSV files."""
    for prefix in ("AvgC", "MaxC", "B365C", "PSC", "BFEC"):
        h, d, a = (
            _parse_float(row.get(f"{prefix}H")),
            _parse_float(row.get(f"{prefix}D")),
            _parse_float(row.get(f"{prefix}A")),
        )
        if h and d and a:
            return h, d, a
    return None, None, None


def _read_csv(content: str | bytes) -> pd.DataFrame:
    df = pd.read_csv(
        io.StringIO(content) if isinstance(content, str) else content.decode(),
        on_bad_lines="skip",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_football_data_csv(
    content: str | bytes,
    *,
    league: str,
    season: str,
    source: str = "football-data.co.uk",
) -> tuple[list[HistoricalMatchDraft], list[str]]:
    """Parse CSV content into HistoricalMatch rows. Returns (matches, errors)."""
    df = _read_csv(content)
    matches: list[HistoricalMatchDraft] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            date_val = row.get("Date")
            home = row.get("HomeTeam")
            away = row.get("AwayTeam")
            if pd.isna(date_val) or pd.isna(home) or pd.isna(away):
                errors.append(f"Row {idx}: missing date or teams")
                continue
            match_date = pd.to_datetime(date_val, dayfirst=True).date()
            ftr = str(row.get("FTR", "")).strip().upper()
            if ftr not in FTR_MAP:
                errors.append(f"Row {idx}: invalid FTR '{ftr}'")
                continue
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            oh, od, oa = _extract_odds(row)
            raw = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            matches.append(
                HistoricalMatchDraft(
                    source=source,
                    league=league,
                    season=season,
                    match_date=match_date,
                    home_team=str(home).strip(),
                    away_team=str(away).strip(),
                    home_goals=hg,
                    away_goals=ag,
                    result=FTR_MAP[ftr],  # type: ignore[arg-type]
                    odds_home=oh,
                    odds_draw=od,
                    odds_away=oa,
                    raw_data={str(k): raw[k] for k in raw},
                )
            )
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
    return matches, errors


def parse_extra_league_csv(
    content: str | bytes,
    *,
    league: str,
    seasons: list[str] | None = None,
    source: str = "football-data.co.uk",
) -> tuple[list[HistoricalMatchDraft], list[str]]:
    """Parse combined extra-league CSV, optionally filtering by season codes."""
    df = _read_csv(content)
    allowed_start_years: set[int] | None = None
    if seasons:
        allowed_start_years = {season_code_to_start_year(s) for s in seasons}

    matches: list[HistoricalMatchDraft] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            date_val = row.get("Date")
            home = row.get("Home")
            away = row.get("Away")
            if pd.isna(date_val) or pd.isna(home) or pd.isna(away):
                errors.append(f"Row {idx}: missing date or teams")
                continue

            season_val = row.get("Season")
            start_year = parse_extra_season_start_year(season_val)
            if start_year is None:
                errors.append(f"Row {idx}: missing or invalid Season")
                continue
            if allowed_start_years is not None and start_year not in allowed_start_years:
                continue

            match_date = pd.to_datetime(date_val, dayfirst=True).date()
            res = str(row.get("Res", "")).strip().upper()
            if res not in FTR_MAP:
                errors.append(f"Row {idx}: invalid Res '{res}'")
                continue

            hg = int(row["HG"])
            ag = int(row["AG"])
            oh, od, oa = _extract_extra_odds(row)
            raw = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            matches.append(
                HistoricalMatchDraft(
                    source=source,
                    league=league,
                    season=start_year_to_season_code(start_year),
                    match_date=match_date,
                    home_team=str(home).strip(),
                    away_team=str(away).strip(),
                    home_goals=hg,
                    away_goals=ag,
                    result=FTR_MAP[res],  # type: ignore[arg-type]
                    odds_home=oh,
                    odds_draw=od,
                    odds_away=oa,
                    raw_data={str(k): raw[k] for k in raw},
                )
            )
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
    return matches, errors


class FootballDataUKProvider:
    """Download and parse Football-Data.co.uk league CSV files."""

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        self.config = config or DataSourceConfig()

    def csv_url(self, league: str, season: str) -> str:
        return f"{self.config.football_data_base_url}/{season}/{league}.csv"

    def extra_csv_url(self, league: str) -> str:
        return f"{self.config.football_data_extra_base_url}/{league}.csv"

    def _local_csv_path(self, league: str, season: str) -> Path:
        return self.config.raw_football_data_dir / season / f"{league}.csv"

    def _local_extra_csv_path(self, league: str) -> Path:
        return self.config.raw_football_data_dir / "extra" / f"{league}.csv"

    def download_csv(self, league: str, season: str) -> str:
        local = self._local_csv_path(league, season)
        if local.exists():
            logger.info("Using cached CSV %s", local)
            return local.read_text(encoding="utf-8", errors="replace")

        url = self.csv_url(league, season)
        try:
            import httpx

            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {url} and no local file at {local}: {exc}"
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text, encoding="utf-8")
        logger.info("Cached CSV %s", local)
        return text

    def download_extra_csv(self, league: str) -> str:
        """Download combined extra-league CSV text."""
        local = self._local_extra_csv_path(league)
        if local.exists():
            logger.info("Using cached CSV %s", local)
            return local.read_text(encoding="utf-8", errors="replace")

        url = self.extra_csv_url(league)
        try:
            import httpx

            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {url} and no local file at {local}: {exc}"
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text, encoding="utf-8")
        logger.info("Cached CSV %s", local)
        return text

    def parse_file(self, path: Path, league: str, season: str) -> tuple[list[HistoricalMatchDraft], list[str]]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return parse_football_data_csv(content, league=league, season=season)

    def _log_parse_errors(self, league: str, label: str, errors: list[str]) -> None:
        for err in errors[:5]:
            logger.warning("%s/%s: %s", league, label, err)
        if len(errors) > 5:
            logger.warning("%s/%s: %d more row errors skipped", league, label, len(errors) - 5)

    def fetch_historical_matches(
        self, leagues: list[str], seasons: list[str]
    ) -> list[HistoricalMatchDraft]:
        all_matches: list[HistoricalMatchDraft] = []
        main_leagues = [lg for lg in leagues if not is_extra_league(lg)]
        extra_leagues = [lg for lg in leagues if is_extra_league(lg)]

        for season in seasons:
            for league in main_leagues:
                logger.info("Fetching %s %s", league, season)
                text = self.download_csv(league, season)
                matches, errors = parse_football_data_csv(
                    text, league=league, season=season
                )
                self._log_parse_errors(league, season, errors)
                all_matches.extend(matches)


        for league in extra_leagues:

            logger.info("Fetching extra league %s (seasons %s)", league, ",".join(seasons))
            text = self.download_extra_csv(league)
            matches, errors = parse_extra_league_csv(
                text, league=league, seasons=seasons
            )
            self._log_parse_errors(league, "extra", errors)
            all_matches.extend(matches)

        return all_matches
