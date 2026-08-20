"""Import and persist historical football data from API-Football."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from data_sources import api_football_client
from data_sources.api_football_client import APIFootballClient, API_FOOTBALL_SOURCE

from objects.repositories.external_entity_mapping_repository import (
    ExternalEntityMappingRepository,
)
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.meta_data_repository import MetaDataRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.fixture import FixtureCreate
from services.draw_manager import STDrawManager
from utils.common import LEAGUES_EXTERNAL_IDS
from utils.seasons import last_n_season_codes

logger = logging.getLogger(__name__)

HISTORICAL_MATCHES_LAST_REFRESH_KEY = "historical_matches_last_refresh"


def refresh_league_codes(
    session: Session | None = None,
    leagues: list[str] | None = None,
) -> dict[str, list]:
    """Return API-Football league codes (optionally filtered)."""
    del session  # kept for call-site compatibility
    codes = all_api_football_league_codes()
    if leagues is not None:
        wanted = set(leagues)
        codes = [code for code in codes if code in wanted]
    return {API_FOOTBALL_SOURCE: codes}


class DataCollector:
    """Import API-Football fixtures into the fixtures table."""

    def __init__(
        self,
        session: Session | None = None,
    ) -> None:
        self.config = DataSourceConfig()
        self._owns_session = session is None

        self.fixtures_repo = FixtureRepository(session)
        # Alias kept for call sites / tests still using the old name.
        self.draw_manager = STDrawManager(session)
        self.leagues_repo = LeagueRepository(session)
        self.teams_repo = TeamRepository(session)
        self.metadata_repo = MetaDataRepository(session)
        self.mapping_repo = ExternalEntityMappingRepository(
            self.fixtures_repo.session
        )
        self.api_football_client = APIFootballClient(config=self.config)
        self._session = self.fixtures_repo.session

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


    def _ensure_teams_from_fixture(self, create: FixtureCreate) -> None:
        self.teams_repo.create_from_provider_team(
            external_id=create.home_team_id,
            name=create.home_team_name,
        )
        self.teams_repo.create_from_provider_team(
            external_id=create.away_team_id,
            name=create.away_team_name,
        )

    def import_api_football_matches(
        self,
        seasons: list[str] | None = None,
        leagues: list[str] | None = None,
        *,
        from_date: date | None = None,
    ) -> int:
        season_list = seasons or last_n_season_codes(1)
        codes = leagues

        creates: list[FixtureCreate] = []
        for code in codes:
            entry = self.leagues_repo.get_by_external_id(int(code))
            if entry is None:
                logger.warning("No API-Football map entry for league code %s", code)
                continue
            for season in season_list:
                fixtures = api_football_client.get_fixtures_by_league(
                    self.api_football_client,
                    code,
                    season,
                )
                logger.info(
                    "Fetched %d fixtures for %s (id=%s) season %s",
                    len(fixtures),
                    code,
                    code,
                    season,
                )
                for fixture in fixtures:
                    create = api_football_client.fixture_to_create(fixture)
                    if create is None:
                        continue
                    if from_date is not None and create.fixture_date.date() < from_date:
                        continue
                    self._ensure_teams_from_fixture(create)
                    creates.append(create)

        self.leagues_repo.flush()
        self.teams_repo.flush()
        count = self.fixtures_repo.upsert_many(creates)
        logger.info(
            "Saved %d API-Football fixtures from %d creates",
            count,
            len(creates),
        )
        return count

    def league_codes_by_source(
        self, leagues: list[str] | None = None
    ) -> dict[str, list]:
        """League codes each source uses in refresh_all_data."""
        return refresh_league_codes(self._session, leagues)

    def refresh_all_data(
        self,
        seasons: list[str] | None = None,
    ) -> int:

        season_list = seasons or last_n_season_codes(1)
        logger.info(
            "Full API-Football refresh for %d seasons (%s)",
            len(season_list),
            season_list,
        )
        total = self.import_api_football_matches(
            seasons=season_list,
            leagues=LEAGUES_EXTERNAL_IDS,
        )
        self._set_last_refresh_at(datetime.now(timezone.utc))
        logger.info("Refreshed %d fixtures via API-Football", total)
        return total

    def close(self) -> None:
        if self._owns_session:
            self.fixtures_repo.close()
            self.leagues_repo.close()
            self.teams_repo.close()
            self.metadata_repo.close()

    def refresh_all_st_rounds(self):
        DRAW_NUMBER = 4267
        DRAW_NUMBER_MAX = 4959

        for i in range(DRAW_NUMBER, DRAW_NUMBER_MAX):
            self.draw_manager.import_draw(i)
