#!/usr/bin/env python3
"""Import FotMob advanced stats and shots for finished fixtures missing them."""

from __future__ import annotations

import argparse
import logging
from datetime import date

from data_sources.football_data import ExtendedMatchDataService
from data_sources.football_data.fotmob_entity_resolver import FotMobEntityResolver
from data_sources.football_data.results import BatchImportResult
from database import SessionLocal, init_db
from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.common import LEAGUES_EXTERNAL_IDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_missing_stats(
    *,
    seasons: list[str | int] | None = None,
    before_date: date | None = None,
    limit: int | None = None,
    force_refresh: bool = False,
    provider: str = "fotmob",
) -> BatchImportResult:
    """Fetch and store MatchAdvancedStats + MatchShots for fixtures lacking them."""
    init_db()
    session = SessionLocal()
    service = ExtendedMatchDataService(
        provider=provider,
        session=session,
        team_resolver=FotMobEntityResolver(session),
        config=DataSourceConfig(),
    )
    try:
        return service.fetch_and_store_all_fixtures(
            external_league_ids=[int(id) for id in LEAGUES_EXTERNAL_IDS],
            seasons=seasons,
            before_date=before_date,
            limit=limit,
            force_refresh=force_refresh,
        )
    finally:
        print(service._alias_candidate)
        service.close()
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=None)
    parser.add_argument("--before-date", type=date.fromisoformat, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--provider", default="fotmob")
    args = parser.parse_args()

    result = fetch_missing_stats(
        seasons=args.seasons,
        before_date=args.before_date,
        limit=args.limit,
        force_refresh=args.force_refresh,
        provider=args.provider,
    )
    logger.info(
        "requested=%s imported=%s updated=%s skipped=%s unresolved=%s failed=%s",
        result.requested,
        result.imported,
        result.updated,
        result.skipped,
        result.unresolved,
        result.failed,
    )


if __name__ == "__main__":
    main()
