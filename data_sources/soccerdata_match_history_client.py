"""Wrapper around soccerdata.MatchHistory for historical match ingestion.

Example:
    from datetime import date

    from data_sources.soccerdata_match_history_client import SoccerDataMatchHistoryClient

    client = SoccerDataMatchHistoryClient()
    matches = client.fetch_matches_from_date(
        league="ENG-Premier League",
        from_date=date(2025, 8, 1),
        seasons=["2025"],
    )
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, IO, Literal

import httpx
import pandas as pd
from soccerdata import MatchHistory

logger = logging.getLogger(__name__)

SOURCE_FOOTBALL_DATA = "football-data.co.uk"

DATE_COLUMNS = ("date", "Date")
HOME_TEAM_COLUMNS = ("home_team", "HomeTeam")
AWAY_TEAM_COLUMNS = ("away_team", "AwayTeam")
HOME_GOALS_COLUMNS = ("FTHG", "home_goals")
AWAY_GOALS_COLUMNS = ("FTAG", "away_goals")
ODDS_HOME_COLUMNS = ("B365H",)
ODDS_DRAW_COLUMNS = ("B365D",)
ODDS_AWAY_COLUMNS = ("B365A",)


class SoccerDataSchemaError(ValueError):
    """Raised when soccerdata output is missing required columns."""


class SoccerDataMatchHistoryClient:
    """Fetch and normalize historical matches from soccerdata.MatchHistory."""

    def __init__(
        self,
        *,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: Path | None = None,
    ) -> None:
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = data_dir

    def fetch_matches_from_date(
        self,
        league: str,
        from_date: date,
        seasons: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._fetch_matches(league=league, seasons=seasons, from_date=from_date)

    def fetch_matches_for_seasons(
        self,
        league: str,
        seasons: list[str],
    ) -> list[dict[str, Any]]:
        return self._fetch_matches(league=league, seasons=seasons, from_date=None)

    def _fetch_matches(
        self,
        *,
        league: str,
        seasons: list[str] | None,
        from_date: date | None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "leagues": league,
            "no_cache": self.no_cache,
            "no_store": self.no_store,
        }
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir
        if seasons is not None:
            kwargs["seasons"] = seasons

        match_history = MatchHistory(**kwargs)
        games = self._read_games(match_history)
        if games.empty:
            logger.info("No games returned for league=%s seasons=%s", league, seasons)
            return []

        frame = games.reset_index()
        self._validate_required_columns(frame)

        matches: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            row_league = str(row.get("league", league))
            season = str(row.get("season", ""))
            normalized = self._normalize_row(
                row,
                league=row_league,
                season=season,
                from_date=from_date,
            )
            if normalized is not None:
                matches.append(normalized)

        logger.info(
            "Fetched %d matches for league=%s seasons=%s from_date=%s",
            len(matches),
            league,
            seasons,
            from_date,
        )
        return matches

    def _read_games(self, match_history: MatchHistory) -> pd.DataFrame:
        """Read games via soccerdata, downloading CSVs with httpx.

        soccerdata uses ``tls_requests`` for downloads. football-data.co.uk
        often returns 503 to that client even though the same URL works in a
        browser or with httpx/requests. Patch the downloader for MatchHistory
        only so parsing and normalization stay in soccerdata.
        """
        original_download = match_history._download_and_save

        def download_with_httpx(
            url: str,
            filepath: Path | None = None,
            var: Any = None,
        ) -> IO[bytes]:
            del var
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            payload = response.content
            if not match_history.no_store and filepath is not None:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_bytes(payload)
            return io.BytesIO(payload)

        match_history._download_and_save = download_with_httpx
        try:
            return match_history.read_games()
        finally:
            match_history._download_and_save = original_download

    def _validate_required_columns(self, frame: pd.DataFrame) -> None:
        column_sets = (
            DATE_COLUMNS,
            HOME_TEAM_COLUMNS,
            AWAY_TEAM_COLUMNS,
            HOME_GOALS_COLUMNS,
            AWAY_GOALS_COLUMNS,
        )
        missing: list[str] = []
        for aliases in column_sets:
            if not any(name in frame.columns for name in aliases):
                missing.append("/".join(aliases))

        if missing:
            message = (
                "soccerdata MatchHistory output is missing required columns: "
                + ", ".join(missing)
            )
            logger.error(message)
            raise SoccerDataSchemaError(message)

    def _normalize_row(
        self,
        row: pd.Series,
        *,
        league: str,
        season: str,
        from_date: date | None,
    ) -> dict[str, Any] | None:
        match_date = self._parse_match_date(self._get_first(row, *DATE_COLUMNS))
        if match_date is None:
            logger.debug("Skipping row with unparseable date in league=%s", league)
            return None

        if from_date is not None and match_date < from_date:
            return None

        home_team = self._get_first(row, *HOME_TEAM_COLUMNS)
        away_team = self._get_first(row, *AWAY_TEAM_COLUMNS)
        if self._none_if_nan(home_team) is None or self._none_if_nan(away_team) is None:
            logger.debug("Skipping row with missing teams in league=%s", league)
            return None

        home_goals = self._to_int_or_none(self._get_first(row, *HOME_GOALS_COLUMNS))
        away_goals = self._to_int_or_none(self._get_first(row, *AWAY_GOALS_COLUMNS))
        if home_goals is None or away_goals is None:
            logger.debug(
                "Skipping unfinished match %s vs %s on %s",
                home_team,
                away_team,
                match_date,
            )
            return None

        return {
            "source": SOURCE_FOOTBALL_DATA,
            "league": league,
            "season": season,
            "match_date": match_date,
            "home_team": str(home_team).strip(),
            "away_team": str(away_team).strip(),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": self._result_from_goals(home_goals, away_goals),
            "odds_home": self._to_float_or_none(self._get_first(row, *ODDS_HOME_COLUMNS)),
            "odds_draw": self._to_float_or_none(self._get_first(row, *ODDS_DRAW_COLUMNS)),
            "odds_away": self._to_float_or_none(self._get_first(row, *ODDS_AWAY_COLUMNS)),
            "raw_data": self._raw_row_dict(row),
        }

    def _raw_row_dict(self, row: pd.Series) -> dict[str, Any]:
        raw = row.to_dict()
        return {str(key): self._none_if_nan(value) for key, value in raw.items()}

    def _get_first(self, row: pd.Series, *names: str) -> Any:
        for name in names:
            if name in row.index:
                return row[name]
        return None

    def _parse_match_date(self, value: Any) -> date | None:
        if self._none_if_nan(value) is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    def _result_from_goals(
        self, home_goals: int, away_goals: int
    ) -> Literal["H", "D", "A"]:
        if home_goals > away_goals:
            return "H"
        if home_goals == away_goals:
            return "D"
        return "A"

    def _none_if_nan(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._none_if_nan(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._none_if_nan(item) for item in value]
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def _to_int_or_none(self, value: Any) -> int | None:
        value = self._none_if_nan(value)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float_or_none(self, value: Any) -> float | None:
        value = self._none_if_nan(value)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
