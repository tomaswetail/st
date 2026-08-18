"""Import and persist historical football data."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from sqlalchemy import select

from sqlalchemy.orm import Session

from data_sources.api_football_client import APIFootballClient
from data_sources.entity_resolver import EntityResolver
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
    espn_league_for_code,
    partition_leagues_by_source,
    soccerdata_league_for_code,
)
from data_sources import api_football_client
from data_sources.soccerdata_espn_client import SoccerDataEspnClient
from data_sources.soccerdata_match_history_client import SoccerDataMatchHistoryClient
from objects.models import ExternalEntityMappingModel

from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.meta_data_repository import MetaDataRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.fixture import Fixture
from objects.schema.db.historical_match import HistoricalMatchCreate, HistoricalMatchDraft
from services.draw_manager import STDrawManager
from utils.seasons import last_n_season_codes

logger = logging.getLogger(__name__)

HISTORICAL_MATCHES_LAST_REFRESH_KEY = "historical_matches_last_refresh"

def mapped_fotmob_league_ids(session: Session) -> list[tuple[ str | None]]:
    """Return (leagues.id, fotmob_id, external_name) for all mapped leagues."""
    rows = session.scalars(
        select(ExternalEntityMappingModel).where(
            ExternalEntityMappingModel.provider == "fotmob",
            ExternalEntityMappingModel.entity_type == "league",
            ExternalEntityMappingModel.external_entity_id != "40",
        )
    ).all()
    return [
        (row.external_entity_id)
        for row in rows
    ]


def refresh_league_codes(session: Session, leagues: list[str] | None = None) -> dict[str, list]:
    """Return the league codes/ids refresh_all_data assigns to each source."""
    codes = leagues if leagues is not None else all_football_data_league_codes()
    soccerdata_codes, espn_codes, csv_codes = partition_leagues_by_source(codes)
    if leagues is None:
        tournament_codes = all_tournament_codes()
        api_football_ids = list(API_FOOTBALL_SWEDISH_LEAGUE_IDS)
    else:
        tournament_codes = [
            code for code in all_tournament_codes() if code in codes
        ]
        api_football_ids = []

    fotmob_leagues = mapped_fotmob_league_ids(session)
    return {
        "soccerdata": soccerdata_codes,
        "espn": espn_codes,
        "football-data.co.uk": csv_codes,
        "tournaments": tournament_codes,
        "api-football": api_football_ids,
        "fotmob": fotmob_leagues,
    }

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
API_FOOTBALL_SEASONS = [2022]  # , 2023, 2024]


class DataCollector:
    """Import Football-Data.co.uk CSVs and rebuild Elo."""

    def __init__(
        self,
        session: Session | None = None,
    ) -> None:
        self.config = DataSourceConfig()
        self._owns_session = session is None

        self.historical_matches_repo = HistoricalMatchRepository(session)
        self.draw_manager = STDrawManager(session)
        self.leagues_repo = LeagueRepository(session)
        self.teams_repo = TeamRepository(session)
        self.metadata_repo = MetaDataRepository(session)
        self.provider = FootballDataUKProvider(self.config)
        self.tournament_provider = FootballDataTournamentProvider(self.config)
        self.soccerdata_client = SoccerDataMatchHistoryClient()
        self.espn_client = SoccerDataEspnClient()
        self.api_football_client = APIFootballClient()
        self._session = self.historical_matches_repo.session

    def matches_with_resolved_teams(
        self,
        drafts: list[HistoricalMatchDraft],
    ) -> list[HistoricalMatchCreate]:
        """Resolve draft team names to TeamModel FKs via EntityResolver."""
        resolved: list[HistoricalMatchCreate] = []
        resolvers: dict[str, EntityResolver] = {}

        for draft in drafts:
            league = self.leagues_repo.get_by_code(draft.league)


            resolver = resolvers.get(draft.source)
            if resolver is None:
                resolver = EntityResolver(
                    self._session,
                    config=self.config,
                    provider=draft.source,
                )
                resolvers[draft.source] = resolver

            home = resolver.resolve_team(
                provider_team_id=draft.home_team,
                provider_team_name=draft.home_team,
                league_id=league.id if league else 0,
                create_if_missing=True,
            )
            away = resolver.resolve_team(
                provider_team_id=draft.away_team,
                provider_team_name=draft.away_team,
                league_id=league.id if league else 0,
                create_if_missing=True,
            )
            if home.team is None or away.team is None:
                logger.warning(
                    "Skipping match with unresolved teams: %s vs %s",
                    draft.home_team,
                    draft.away_team,
                )
                continue

            resolved.append(
                HistoricalMatchCreate(
                    source=draft.source,
                    league=draft.league,
                    season=draft.season,
                    match_date=draft.match_date,
                    home_team_id=home.team.id,
                    away_team_id=away.team.id,
                    home_goals=draft.home_goals,
                    away_goals=draft.away_goals,
                    result=draft.result,
                    odds_home=draft.odds_home,
                    odds_draw=draft.odds_draw,
                    odds_away=draft.odds_away,
                    raw_data=draft.raw_data,
                )
            )

        self.teams_repo.flush()
        return resolved

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

        for fixture in all_fixtures:
            self.leagues_repo.upsert_from_match(
                {"name": fixture.league_name, "country": fixture.country}
            )
        self.leagues_repo.flush()

        drafts = [
            match
            for fixture in all_fixtures
            if (match := api_football_client.fixture_to_historical_match_create(fixture))
            is not None
        ]
        matches = self.matches_with_resolved_teams(drafts)
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
            self.leagues_repo.ensure_from_code(code)
        self.leagues_repo.flush()

        drafts = self.provider.fetch_historical_matches(leagues, seasons)
        if from_date is not None:
            drafts = [match for match in drafts if match.match_date >= from_date]
        matches = self.matches_with_resolved_teams(drafts)
        count = self.historical_matches_repo.upsert_many(matches)
        logger.info("Saved %d historical matches", count)
        return count

    def import_soccerdata_matches(
        self,
        sd_codes: list[str],
        season_list: list[str],
        *,
        from_date: date | None = None,
    ) -> list[HistoricalMatchDraft]:
        all_matches: list[HistoricalMatchDraft] = []
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

    def import_espn_matches(
        self,
        espn_codes: list[str],
        season_list: list[str],
        *,
        from_date: date | None = None,
    ) -> list[HistoricalMatchDraft]:
        all_matches: list[HistoricalMatchDraft] = []

        for code in espn_codes:
            espn_league = espn_league_for_code(code)
            if espn_league is None:
                logger.warning("No ESPN mapping for league code %s", code)
                continue

            self.leagues_repo.ensure_from_code(code)
            if from_date is not None:
                rows = self.espn_client.fetch_matches_from_date(
                    espn_league,
                    from_date,
                    seasons=season_list,
                )
            else:
                rows = self.espn_client.fetch_matches_for_seasons(
                    espn_league,
                    season_list,
                )
            league_matches = [
                to_historical_match_create(row, league_code=code) for row in rows
            ]
            all_matches.extend(league_matches)
            logger.info(
                "Fetched %d matches for league %s (%s) via ESPN",
                len(league_matches),
                code,
                espn_league,
            )
        return all_matches

    def import_tournaments(self, tournaments: list[str]) -> int:
        drafts = self.tournament_provider.fetch_historical_matches(tournaments)
        for draft in drafts:
            self.leagues_repo.ensure_from_code(draft.league)
        self.leagues_repo.flush()
        matches = self.matches_with_resolved_teams(drafts)
        count = self.historical_matches_repo.upsert_many(matches)
        logger.info("Saved %d tournament matches", count)
        return count

    def league_codes_by_source(
        self, leagues: list[str] | None = None
    ) -> dict[str, list]:
        """League codes/ids each source uses in refresh_all_data."""
        return refresh_league_codes(self._session, leagues)

    def refresh_all_data(self, seasons: list[str] | None = None) -> int:
        season_list = seasons or last_n_season_codes(1)
        from_date = None  # self._get_last_refresh_at()
        source_codes = self.league_codes_by_source()
        soccerdata_codes = source_codes["soccerdata"]
        espn_codes = source_codes["espn"]
        csv_codes = source_codes["football-data.co.uk"]

        if from_date is not None:
            logger.info("Incremental refresh from %s", from_date)
        else:
            logger.info("Full refresh for last %d seasons", len(season_list))
        soccerdata_count, espn_count, csv_count = 0, 0, 0
        soccerdata_drafts = self.import_soccerdata_matches(
            soccerdata_codes,
            season_list,
            from_date=from_date,
        )

        if soccerdata_drafts:
            self.leagues_repo.flush()
            soccerdata_matches = self.matches_with_resolved_teams(soccerdata_drafts)
            soccerdata_count = self.historical_matches_repo.upsert_many(soccerdata_matches)
        espn_drafts = self.import_espn_matches(
            espn_codes,
            season_list,
            from_date=from_date,
        )
        if espn_drafts:
            self.leagues_repo.flush()
            espn_matches = self.matches_with_resolved_teams(espn_drafts)
            espn_count = self.historical_matches_repo.upsert_many(espn_matches)

        if csv_codes:
            csv_count = self.import_football_data(
                csv_codes,
                season_list,
                from_date=from_date,
            )
        self.import_api_football_swedish_matches(season_list)
        self.import_tournaments(source_codes["tournaments"])
        total = soccerdata_count + espn_count + csv_count
        self._set_last_refresh_at(datetime.now(timezone.utc))
        logger.info(
            "Refreshed %d historical matches across %d seasons "
            "(%d via soccerdata, %d via ESPN, %d via football-data CSV)",
            total,
            len(season_list),
            soccerdata_count,
            espn_count,
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
