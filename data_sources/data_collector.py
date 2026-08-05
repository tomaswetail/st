"""Import and persist historical football data."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from data_sources.api_football_client import APIFootballClient
from data_sources.football_data_tournaments_xslx_provider import (
    FootballDataTournamentProvider,
    all_tournament_codes,
)
from data_sources.football_data_uk_xlsx_provider import (
    FootballDataUKProvider,
    all_football_data_league_codes,
)
from data_sources.soccerdata_importer import to_historical_match_create
from data_sources.soccerdata_league_mapping import (
    partition_leagues_by_source,
    soccerdata_league_for_code,
)
from data_sources import api_football_client
from data_sources.soccerdata_match_history_client import SoccerDataMatchHistoryClient
from database import SessionLocal
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.meta_data_repository import MetaDataRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.fixture import Fixture
from objects.schema.db.historical_match import HistoricalMatchCreate
from services.draw_manager import STDrawManager
from utils.seasons import last_n_season_codes

logger = logging.getLogger(__name__)

HISTORICAL_MATCHES_LAST_REFRESH_KEY = "historical_matches_last_refresh"

API_FOOTBALL_SWEDISH_LEAGUE_IDS = [
    114,
    563,
    564,
    592,
    593,
    594,
    595,
    596,
    597,
    115,
    549,
    737,
    736,
    1053,
    1055,
]
API_FOOTBALL_SEASONS = [2022]#, 2023, 2024]


class DataCollector:
    """Import Football-Data.co.uk CSVs and rebuild Elo."""

    def __init__(
        self,
        session: Session | None = None,
    ) -> None:
        self.config = DataSourceConfig()

        self.historical_matches_repo = HistoricalMatchRepository(session)
        self.draw_manager = STDrawManager(session)
        self.leagues_repo = LeagueRepository(session)
        self.teams_repo = TeamRepository(session)
        self.metadata_repo = MetaDataRepository(session)
        self.provider = FootballDataUKProvider(self.config)
        self.tournament_provider = FootballDataTournamentProvider(self.config)
        self.soccerdata_client = SoccerDataMatchHistoryClient()
        self.api_football_client = APIFootballClient()

    def _ensure_teams(
        self,
        entries: list[tuple[str, str, int]],
    ) -> None:
        """Ensure teams for (home_team, away_team, league_id) entries."""
        seen: set[tuple[str, int]] = set()
        for home_team, away_team, league_id in entries:
            for team_name in (home_team, away_team):
                key = (team_name, league_id)
                if key in seen:
                    continue
                seen.add(key)
                self.teams_repo.ensure_from_historical(team_name, league_id)
        self.teams_repo.flush()

    def _get_last_refresh_at(self) -> date | None:
        raw = self.metadata_repo.get_value(HISTORICAL_MATCHES_LAST_REFRESH_KEY)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            logger.warning("Invalid last refresh metadata value: %s", raw)
            return None

    def _set_last_refresh_at(self, refreshed_at: datetime) -> None:
        if refreshed_at.tzinfo is None:
            refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
        self.metadata_repo.set_value(
            HISTORICAL_MATCHES_LAST_REFRESH_KEY,
            refreshed_at.isoformat(),
        )

    def _team_entries_from_matches(
        self, matches: list[HistoricalMatchCreate]
    ) -> list[tuple[str, str, int]]:
        entries: list[tuple[str, str, int]] = []
        for match in matches:
            league = self.leagues_repo.get_by_code(match.league)
            if league is None:
                continue
            entries.append((match.home_team, match.away_team, league.id))
        return entries

    def import_api_football_swedish_matches(
        self,
        seasons: list[int] | None = None,
    ) -> int:
        league_ids = API_FOOTBALL_SWEDISH_LEAGUE_IDS
        seasons = seasons or API_FOOTBALL_SEASONS

        all_fixtures: list[Fixture] = []
        for league_id in league_ids:

            for season in seasons:
                fixtures = api_football_client.get_fixtures_by_league(
                    self.api_football_client,
                    league_id,
                    season,
                )
                all_fixtures.extend(fixtures)
                logger.info(
                    "Fetched %d fixtures for API-Football league %d season %s",
                    len(fixtures),
                    league_id,
                    season,
                )

        matches = [
            match
            for fixture in all_fixtures
            if (match := api_football_client.fixture_to_historical_match_create(fixture)) is not None
        ]
        if all_fixtures:
            entries: list[tuple[str, str, int]] = []
            for fixture in all_fixtures:
                league = self.leagues_repo.upsert_from_match(
                    {"name": fixture.league_name, "country": fixture.country}
                )
                entries.append((fixture.home_team, fixture.away_team, league.id))
            self._ensure_teams(entries)

        count = self.historical_matches_repo.upsert_many(matches)
        logger.info(
            "Saved %d API-Football historical matches from %d fixtures",
            count,
            len(all_fixtures),
        )
        return count

    def import_football_data(
        self,
        leagues: list[str],
        seasons: list[str],
        *,
        from_date: date | None = None,
    ) -> int:
        for code in leagues:
            if code != 'SWE':
                continue
            self.leagues_repo.ensure_from_code(code)
        self.leagues_repo.flush()

        matches = self.provider.fetch_historical_matches(leagues, seasons)
        if from_date is not None:
            matches = [match for match in matches if match.match_date >= from_date]
        self._ensure_teams(self._team_entries_from_matches(matches))
        count = self.historical_matches_repo.upsert_many(matches)
        logger.info("Saved %d historical matches", count)
        return count

    def import_soccerdata_matches(
        self,
        sd_codes: list[str],
        season_list: list[str],
        *,
        from_date: date | None = None,
    ) -> list[HistoricalMatchCreate]:
        all_matches: list[HistoricalMatchCreate] = []
        for code in sd_codes:
            soccerdata_league = soccerdata_league_for_code(code)
            assert soccerdata_league is not None

            self.leagues_repo.ensure_from_code(code)
            if from_date is not None:
                rows = self.soccerdata_client.fetch_matches_from_date(
                    soccerdata_league,
                    from_date,
                    seasons=season_list,
                )
            else:
                rows = self.soccerdata_client.fetch_matches_for_seasons(
                    soccerdata_league,
                    season_list,
                )
            league_matches = [
                to_historical_match_create(row, league_code=code) for row in rows
            ]
            all_matches.extend(league_matches)
            logger.info(
                "Fetched %d matches for league %s (%s) via soccerdata",
                len(league_matches),
                code,
                soccerdata_league,
            )
        return all_matches

    def import_tournaments(self, tournaments: list[str]) -> int:
        matches = self.tournament_provider.fetch_historical_matches(tournaments)
        count = self.historical_matches_repo.upsert_many(matches)
        logger.info("Saved %d tournament matches", count)
        return count


    def refresh_all_data(self, seasons: list[str] | None = None) -> int:
        season_list = seasons or last_n_season_codes(1)
        from_date = None# self._get_last_refresh_at()
        soccerdata_codes, csv_codes = partition_leagues_by_source(
            all_football_data_league_codes()
        )

        if from_date is not None:
            logger.info("Incremental refresh from %s", from_date)
        else:
            logger.info("Full refresh for last %d seasons", len(season_list))
        soccerdata_count, csv_count = 0, 0
        """soccerdata_matches = self.import_soccerdata_matches(
            soccerdata_codes,
            season_list,
            from_date=from_date,
        )


        if soccerdata_matches:
            self.leagues_repo.flush()
            self._ensure_teams(self._team_entries_from_matches(soccerdata_matches))
            soccerdata_count = self.historical_matches_repo.upsert_many(soccerdata_matches)
"""

        if csv_codes:
            csv_count = self.import_football_data(
                csv_codes,
                season_list,
                from_date=from_date,
            )
        self.import_api_football_swedish_matches(season_list)

        self.import_tournaments(all_tournament_codes())

        total = soccerdata_count + csv_count
        self._set_last_refresh_at(datetime.now(timezone.utc))
        logger.info(
            "Refreshed %d historical matches across %d seasons "
            "(%d via soccerdata, %d via football-data CSV)",
            total,
            len(season_list),
            soccerdata_count,
            csv_count,
        )
        return total

    def close(self) -> None:
        if self._owns_session:
            self.historical_matches_repo.close()
            self.leagues_repo.close()
            self.teams_repo.close()
            self.metadata_repo.close()

    def refresh_all_st_rounds(self):
        DRAW_NUMBER = 4267
        DRAW_NUMBER_MAX = 4959


        for i in range(DRAW_NUMBER, DRAW_NUMBER_MAX):
            events = self.draw_manager.import_draw(i)
