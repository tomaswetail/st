"""Import SofaScore advanced stats and shots for mapped leagues."""

from __future__ import annotations

import logging

from sqlalchemy import select

from calc.probality_manager import ProbabilityManager
from data_sources.api_football_client import APIFootballClient, get_all_leagues
from data_sources.data_collector import DataCollector
from data_sources.football_data import ExtendedMatchDataService
from database import SessionLocal, init_db
from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.repositories.league_repository import LeagueRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from services.draw_manager import STDrawManager
from utils.common import FOTMOB_TO_API_FOOTBALL_LEAGUE_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SEASON_YEAR = 2020


def mapped_sofascore_league_ids(session) -> list[tuple[int, str, str | None]]:
    """Return (leagues.id, sofascore_id, external_name) for mapped leagues."""
    return FOTMOB_TO_API_FOOTBALL_LEAGUE_MAPPING.values()


def calc() -> None:
    init_db()
    session = SessionLocal()
    STDrawManager(session).import_draw(4750)
    prob_manager = ProbabilityManager(session)
    prob_manager.process(4750)

def _get_leagues(session):
    client = APIFootballClient()
    all_leagues = get_all_leagues(client)
    league_repo = LeagueRepository(session)

    for l in all_leagues:
        league_repo.upsert_from_api(
            external_id=l.league_id,
            league_name=l.league_name,
            league_type=l.league_type,
            country_name=l.country_name,
            country_code=l.country_code,
        )
    league_repo.commit()
def main() -> None:
    init_db()
    session = SessionLocal()
    #collector = DataCollector(session)
    #collector.refresh_all_data(["2223", "2324", "2425", "2526"])
    main_extra_data()


def main_extra_data() -> None:
    init_db()
    session = SessionLocal()
    config = DataSourceConfig(
        football_data_request_delay_ms=750,
    )
    service = ExtendedMatchDataService(
        provider="fotmob",
        session=session,
        config=config,
    )
    try:
        totals = {
            "requested": 0,
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "unresolved": 0,
            "failed": 0,
        }

        for fotmob_league_id, api_fotball_league_id in FOTMOB_TO_API_FOOTBALL_LEAGUE_MAPPING.items():
            result = service.fetch_and_store_league_history(
                external_league_id=api_fotball_league_id,
                provider_league_id=fotmob_league_id,
                season=None,
                force_refresh=False,
                min_season_year=MIN_SEASON_YEAR,
            )
            logger.info(
                "league_id=%s requested=%s imported=%s updated=%s "
                "skipped=%s unresolved=%s failed=%s",
                fotmob_league_id,
                result.requested,
                result.imported,
                result.updated,
                result.skipped,
                result.unresolved,
                result.failed,
            )
            totals["requested"] += result.requested
            totals["imported"] += result.imported
            totals["updated"] += result.updated
            totals["skipped"] += result.skipped
            totals["unresolved"] += result.unresolved
            totals["failed"] += result.failed

        logger.info("DONE totals=%s", totals)
    finally:
        service.close()
        session.close()


if __name__ == "__main__":
    main()
