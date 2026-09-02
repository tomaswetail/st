"""Import SofaScore advanced stats and shots for mapped leagues."""

from __future__ import annotations

import logging

from sqlalchemy import select

from calc.probability_manager import ProbabilityManager
from data_sources.api_football_client import APIFootballClient, get_all_leagues
from data_sources.data_collector import DataCollector
from data_sources.football_data import ExtendedMatchDataService
from data_sources.football_data.providers.fotmob import FotMobProvider
from database import SessionLocal, init_db
from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.repositories.league_repository import LeagueRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from services.draw_manager import STDrawManager
from utils.common import API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING, FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SEASON_YEAR = 2020


def calc() -> None:
    init_db()
    session = SessionLocal()
    STDrawManager(session).import_draw(4750)
    prob_manager = ProbabilityManager(session)
    prob_manager.process(4750)


def main() -> None:
    init_db()
    session = SessionLocal()
    country_codes = {}
    """
    for k,v in API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.items():
        country_codes[v] = FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE[k]
    teams = FotMobProvider().fetch_teams_for_leagues(
        API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.values(),
        country_codes=country_codes,
    )
    for i in teams:
        print(f"{i.provider_team_id}, {i.name}")
    """
    collector = DataCollector(session)
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
        dry_run=False
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

        for api_football_league_id, fotmob_league_id in (
            API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.items()
        ):
            if api_football_league_id != 39:
                pass
            if fotmob_league_id is None:
                logger.info(
                    "Skipping api_league_id=%s (no FotMob mapping)",
                    api_football_league_id,
                )
                continue
            result = service.fetch_and_store_league_history(
                external_league_id=api_football_league_id,
                provider_league_id=fotmob_league_id,
                season=None,
                force_refresh=False,
                min_season_year=MIN_SEASON_YEAR,
            )
            logger.info(
                "api_league_id=%s fotmob_league_id=%s requested=%s imported=%s "
                "updated=%s skipped=%s unresolved=%s failed=%s",
                api_football_league_id,
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


def import_st():
    from services.draw_manager import STDrawManager
    init_db()
    session = SessionLocal()
    d = STDrawManager(session)
    h = d.import_all_draws()
    print(h)

if __name__ == "__main__":
    import_st()
