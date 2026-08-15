"""Wrapper around soccerdata.ESPN for historical match ingestion with scores.

soccerdata.ESPN.read_schedule() omits scores; this client reuses the same
scoreboard download path and extracts home/away scores from each event.
"""

from __future__ import annotations

import itertools
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from soccerdata import ESPN
from soccerdata._config import LEAGUE_DICT, TEAMNAME_REPLACEMENTS
from soccerdata.espn import ESPN_API

logger = logging.getLogger(__name__)

SOURCE_ESPN = "espn"

CUSTOM_ESPN_LEAGUES: dict[str, dict[str, str]] = {
    "ENG-FA Cup": {
        "ESPN": "eng.fa",
        "season_start": "Aug",
        "season_end": "May",
    },
}


def ensure_espn_league_dict() -> None:
    """Register custom ESPN leagues on soccerdata.LEAGUE_DICT (idempotent)."""
    changed = False
    for league_name, league_meta in CUSTOM_ESPN_LEAGUES.items():
        existing = LEAGUE_DICT.get(league_name)
        if existing is None:
            LEAGUE_DICT[league_name] = dict(league_meta)
            changed = True
            continue
        if existing.get("ESPN") != league_meta["ESPN"]:
            existing["ESPN"] = league_meta["ESPN"]
            existing.setdefault("season_start", league_meta["season_start"])
            existing.setdefault("season_end", league_meta["season_end"])
            changed = True
    if changed and hasattr(ESPN, "_all_leagues_dict"):
        delattr(ESPN, "_all_leagues_dict")


class SoccerDataEspnClient:
    """Fetch and normalize historical matches from soccerdata.ESPN scoreboards."""

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
        ensure_espn_league_dict()

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
        ensure_espn_league_dict()
        kwargs: dict[str, Any] = {
            "leagues": league,
            "no_cache": self.no_cache,
            "no_store": self.no_store,
        }
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir
        if seasons is not None:
            kwargs["seasons"] = seasons

        espn = ESPN(**kwargs)
        rows = self._read_scoreboard_matches(espn)
        if not rows:
            logger.info("No games returned for ESPN league=%s seasons=%s", league, seasons)
            return []

        matches: list[dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_row(row, from_date=from_date)
            if normalized is not None:
                matches.append(normalized)

        logger.info(
            "Fetched %d matches for ESPN league=%s seasons=%s from_date=%s",
            len(matches),
            league,
            seasons,
            from_date,
        )
        return matches

    def _read_scoreboard_matches(self, espn: ESPN) -> list[dict[str, Any]]:
        """Download ESPN scoreboards and extract finished matches with scores."""
        urlmask = ESPN_API + "/{}/scoreboard?dates={}"
        filemask = "Schedule_{}_{}.json"
        rows: list[dict[str, Any]] = []
        seen_game_ids: set[int] = set()

        for espn_league_id, season_key in itertools.product(
            espn._selected_leagues.values(),
            espn.seasons,
        ):
            if int(season_key[:2]) > int(str(datetime.now(tz=timezone.utc).year + 1)[-2:]):
                start_date = "".join(["19", season_key[:2], "07", "01"])
            else:
                start_date = "".join(["20", season_key[:2], "07", "01"])

            url = urlmask.format(espn_league_id, start_date)
            filepath = espn.data_dir / filemask.format(espn_league_id, start_date)
            reader = espn.get(url, filepath)
            data = json.load(reader)

            leagues = data.get("leagues") or []
            if not leagues:
                logger.warning(
                    "ESPN scoreboard missing leagues for %s on %s",
                    espn_league_id,
                    start_date,
                )
                continue

            league_payload = leagues[0]
            expected_season_year = 2000 + int(season_key[:2])
            season_year = (league_payload.get("season") or {}).get("year")
            if season_year is not None and int(season_year) != expected_season_year:
                # Cup calendars often still show the previous season on July 1.
                retry_date = f"{expected_season_year}0801"
                url = urlmask.format(espn_league_id, retry_date)
                filepath = espn.data_dir / filemask.format(espn_league_id, retry_date)
                reader = espn.get(url, filepath)
                data = json.load(reader)
                leagues = data.get("leagues") or []
                if not leagues:
                    logger.warning(
                        "ESPN scoreboard missing leagues for %s on %s",
                        espn_league_id,
                        retry_date,
                    )
                    continue
                league_payload = leagues[0]
                season_year = (league_payload.get("season") or {}).get("year")
                if season_year is not None and int(season_year) != expected_season_year:
                    logger.warning(
                        "ESPN season mismatch for %s: wanted %s got %s",
                        espn_league_id,
                        expected_season_year,
                        season_year,
                    )

            date_queries = self._scoreboard_date_queries(league_payload)
            if not date_queries:
                logger.warning(
                    "No ESPN calendar dates for %s season %s",
                    espn_league_id,
                    season_key,
                )
                continue

            canonical_league = self._canonical_league_name(espn, espn_league_id)
            current_season = not espn._is_complete(espn_league_id, season_key)
            for query_start, query_end in date_queries:
                dates_param = (
                    query_start
                    if query_start == query_end
                    else f"{query_start}-{query_end}"
                )
                url = urlmask.format(espn_league_id, dates_param)
                filepath = espn.data_dir / filemask.format(espn_league_id, dates_param)
                reader = espn.get(
                    url,
                    filepath,
                    no_cache=current_season and not espn.no_cache,
                )
                day_data = json.load(reader)
                for event in day_data.get("events") or []:
                    parsed = self._parse_event(
                        event,
                        league=canonical_league,
                        season=season_key,
                        espn_league_id=espn_league_id,
                    )
                    if parsed is None:
                        continue
                    game_id = parsed.get("game_id")
                    if game_id is not None:
                        if game_id in seen_game_ids:
                            continue
                        seen_game_ids.add(game_id)
                    rows.append(parsed)
        return rows

    @classmethod
    def _scoreboard_date_queries(
        cls,
        league_payload: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Build YYYYMMDD start/end query windows from an ESPN league calendar.

        League calendars are either a list of match-day ISO strings (e.g. EPL)
        or round blocks with start/end dates (e.g. FA Cup).
        """
        calendar = league_payload.get("calendar") or []
        queries: list[tuple[str, str]] = []

        if calendar and all(isinstance(day, str) for day in calendar):
            for day in calendar:
                day_key = cls._espn_timestamp_to_yyyymmdd(day)
                if day_key is not None:
                    queries.append((day_key, day_key))
            return queries

        for block in calendar:
            if not isinstance(block, dict):
                continue
            entries = block.get("entries") or []
            if entries:
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    start_key = cls._espn_timestamp_to_yyyymmdd(entry.get("startDate"))
                    end_key = cls._espn_timestamp_to_yyyymmdd(entry.get("endDate"))
                    if start_key and end_key:
                        queries.append((start_key, end_key))
            else:
                start_key = cls._espn_timestamp_to_yyyymmdd(block.get("startDate"))
                end_key = cls._espn_timestamp_to_yyyymmdd(block.get("endDate"))
                if start_key and end_key:
                    queries.append((start_key, end_key))

        if queries:
            return queries

        start_key = cls._espn_timestamp_to_yyyymmdd(
            league_payload.get("calendarStartDate")
        )
        end_key = cls._espn_timestamp_to_yyyymmdd(league_payload.get("calendarEndDate"))
        if start_key and end_key:
            return [(start_key, end_key)]
        return []

    @staticmethod
    def _espn_timestamp_to_yyyymmdd(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d")
        except ValueError:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(parsed):
                return None
            return parsed.strftime("%Y%m%d")

    @staticmethod
    def _canonical_league_name(espn: ESPN, espn_league_id: str) -> str:
        flip = {source_id: name for name, source_id in espn._selected_leagues.items()}
        return flip.get(espn_league_id, espn_league_id)

    def _parse_event(
        self,
        event: dict[str, Any],
        *,
        league: str,
        season: str,
        espn_league_id: str,
    ) -> dict[str, Any] | None:
        competitions = event.get("competitions") or []
        if not competitions:
            return None
        competition = competitions[0]
        status = (event.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            return None

        home: dict[str, Any] | None = None
        away: dict[str, Any] | None = None
        for competitor in competition.get("competitors") or []:
            side = competitor.get("homeAway")
            if side == "home":
                home = competitor
            elif side == "away":
                away = competitor
        if home is None or away is None:
            competitors = competition.get("competitors") or []
            if len(competitors) < 2:
                return None
            home, away = competitors[0], competitors[1]

        home_team = (home.get("team") or {}).get("name")
        away_team = (away.get("team") or {}).get("name")
        if not home_team or not away_team:
            return None

        home_team = TEAMNAME_REPLACEMENTS.get(home_team, home_team)
        away_team = TEAMNAME_REPLACEMENTS.get(away_team, away_team)

        home_goals = self._to_int_or_none(home.get("score"))
        away_goals = self._to_int_or_none(away.get("score"))
        if home_goals is None or away_goals is None:
            return None

        match_date = self._parse_match_date(event.get("date"))
        if match_date is None:
            return None

        return {
            "league": league,
            "season": season,
            "match_date": match_date,
            "home_team": str(home_team).strip(),
            "away_team": str(away_team).strip(),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "game_id": int(event["id"]) if event.get("id") is not None else None,
            "espn_league_id": espn_league_id,
            "raw_event": event,
        }

    def _normalize_row(
        self,
        row: dict[str, Any],
        *,
        from_date: date | None,
    ) -> dict[str, Any] | None:
        match_date = row["match_date"]
        if from_date is not None and match_date < from_date:
            return None

        home_goals = row["home_goals"]
        away_goals = row["away_goals"]
        return {
            "source": SOURCE_ESPN,
            "league": row["league"],
            "season": row["season"],
            "match_date": match_date,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": self._result_from_goals(home_goals, away_goals),
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
            "raw_data": {
                "game_id": row.get("game_id"),
                "espn_league_id": row.get("espn_league_id"),
                "event": row.get("raw_event"),
            },
        }

    def _parse_match_date(self, value: Any) -> date | None:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
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

    def _to_int_or_none(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
