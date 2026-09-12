#!/usr/bin/env python3
"""Ingest historical bookmaker 1X2 odds from football-data.co.uk onto fixtures."""

from __future__ import annotations

import argparse
import logging

from src.data_sources.football_data_odds.constants import ENGLISH_TIER_CODES
from src.data_sources.football_data_odds.service import (
    FootballDataOddsIngestService,
    english_tier_resolution_pct,
    format_stats_line,
)
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        default=None,
        help="CSV code (E0, SWE, …) or mapped league name (Premier League, Allsvenskan, …)",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Main-file season YYXX (e.g. 2324). Extra files ignore season path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and resolve only; do not upsert fixture_odds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upsert even when the fixture already has football-data.co.uk odds",
    )
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    service = FootballDataOddsIngestService(session, config=DataSourceConfig())
    try:
        totals = service.ingest(
            league=args.league,
            season=args.season,
            dry_run=args.dry_run,
            force=args.force,
        )
    finally:
        service.close()
        session.close()

    for stats in totals.summaries:
        print(format_stats_line(stats))

    print(
        f"TOTAL: csv_rows={totals.csv_rows} resolved={totals.resolved} "
        f"unresolved={totals.unresolved} upserted={totals.upserted} "
        f"skipped={totals.skipped}"
    )
    english_pct = english_tier_resolution_pct(totals.summaries)
    if english_pct is not None:
        english_rows = sum(
            item.csv_rows
            for item in totals.summaries
            if item.source in ENGLISH_TIER_CODES
        )
        english_resolved = sum(
            item.resolved
            for item in totals.summaries
            if item.source in ENGLISH_TIER_CODES
        )
        print(
            f"ENGLISH E0-E3: resolved={english_resolved}/{english_rows} "
            f"({english_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()
