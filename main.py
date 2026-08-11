"""Import FotMob advanced stats and shots for all mapped leagues."""

from __future__ import annotations

import logging

from sqlalchemy import select

from calc.probality_manager import ProbabilityManager
from data_sources.data_collector import DataCollector
from data_sources.football_data import ExtendedMatchDataService, LeagueCatalogueService
from database import SessionLocal, init_db
from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.schema.data_classes.data_sources import DataSourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Browser-like UA — FotMob often blocks the default bot UA.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MIN_SEASON_YEAR = 2020


def mapped_fotmob_league_ids(session) -> list[tuple[int, str, str | None]]:
    """Return (leagues.id, fotmob_id, external_name) for all mapped leagues."""
    rows = session.scalars(
        select(ExternalEntityMappingModel).where(
            ExternalEntityMappingModel.provider == "fotmob",
            ExternalEntityMappingModel.entity_type == "league",
        )
    ).all()
    return [
        (row.internal_entity_id, row.external_entity_id, row.external_name)
        for row in rows
    ]

def calc() -> None:
    init_db()
    session = SessionLocal()
    prob_manager = ProbabilityManager(session)
    prob_manager.process(4400)

def main() -> None:
    init_db()
    session = SessionLocal()
    collector = DataCollector(session)

    #collector.refresh_all_data(['2223','2324','2324','2526'])

    main_extra_data()

def main_extra_data() -> None:
    init_db()
    session = SessionLocal()
    config = DataSourceConfig(
        football_data_user_agent=BROWSER_USER_AGENT,
        football_data_request_delay_ms=750,
    )
    service = ExtendedMatchDataService(
        provider="fotmob",
        session=session,
        config=config,
    )
    try:
        leagues = mapped_fotmob_league_ids(session)
        if not leagues:
            league_catalogue_service = LeagueCatalogueService(session)
            league_catalogue_service.map_leagues_from_all_leagues_csv()
            logger.error("No fotmob league mappings found in external_entity_mapping")


        logger.info(
            "Importing advanced stats/shots for %d mapped leagues (from %s)",
            len(leagues),
            MIN_SEASON_YEAR,
        )
        totals = {
            "requested": 0,
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "unresolved": 0,
            "failed": 0,
        }

        for league_id, fotmob_id, external_name in leagues:
            logger.info(
                "=== league_id=%s fotmob=%s (%s) ===",
                league_id,
                fotmob_id,
                external_name,
            )
            result = service.fetch_and_store_league_history(
                league_id=league_id,
                season=None,
                force_refresh=False,
                min_season_year=MIN_SEASON_YEAR,
            )
            logger.info(
                "league_id=%s requested=%s imported=%s updated=%s "
                "skipped=%s unresolved=%s failed=%s",
                league_id,
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
