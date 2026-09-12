#!/usr/bin/env python3
"""Retry football-data.co.uk odds ingest for matches in the unresolved miss log."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data_sources.football_data.http_client import FootballDataHttpError
from src.data_sources.football_data_odds.service import (
    FootballDataOddsIngestService,
    format_retry_totals,
)
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import repo_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = DataSourceConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.unresolved_football_data_odds_csv_path,
        help="Phase 1 unresolved odds CSV (default: data/unresolved_football_data_odds.csv)",
    )
    parser.add_argument(
        "--remaining",
        type=Path,
        default=repo_root() / "data" / "unresolved_football_data_odds_remaining.csv",
        help="Overwrite with still-unresolved rows (default: data/unresolved_football_data_odds_remaining.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve only; do not upsert fixture_odds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upsert even when the fixture already has football-data.co.uk odds",
    )
    parser.add_argument(
        "--rewrite-input",
        action="store_true",
        help="Replace --input with remaining unresolved rows after the run",
    )
    parser.add_argument(
        "--league",
        default=None,
        help="CSV code (E0, SWE, …) or mapped league name (Premier League, Allsvenskan, …)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    init_db()
    session = SessionLocal()
    service = FootballDataOddsIngestService(session, config=config)
    try:
        totals = service.retry_unresolved(
            input_path=args.input,
            remaining_path=args.remaining,
            dry_run=args.dry_run,
            force=args.force,
            league=args.league,
            rewrite_input=args.rewrite_input,
        )
    except FootballDataHttpError as exc:
        logger.error("Required odds CSV fetch failed: %s", exc)
        sys.exit(1)
    finally:
        service.close()
        session.close()

    print(format_retry_totals(totals))


if __name__ == "__main__":
    main()
