#!/usr/bin/env python3
"""Backfill injury/availability snapshots from API-Football.

Sole source: API-Football ``/fixtures/lineups`` + ``/injuries`` (cached HTTP).
FotMob/SofaScore are not used.

Example:

```bash
export PYTHONPATH=src
export API_FOOTBALL_KEY=...

# Dry-run: resolve ST matches → fixtures, no DB writes (still may HTTP unless --skip-http)
python src/scripts/backfill_injury_snapshots.py \\
  --draw-min 4760 --draw-max 4960 --limit 5 --dry-run

# Full backfill for draw window
python src/scripts/backfill_injury_snapshots.py \\
  --draw-min 4760 --draw-max 4960
```
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from data_sources.injuries.backfill import InjuryBackfillService
from database import SessionLocal, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill injury/availability from API-Football lineups + injuries. "
            "Does not use FotMob or SofaScore."
        ),
    )
    parser.add_argument(
        "--draw-min",
        type=int,
        default=STRYKETIPSET_DRAW_MIN,
        help=f"Minimum Stryktipset draw number (default: {STRYKETIPSET_DRAW_MIN})",
    )
    parser.add_argument(
        "--draw-max",
        type=int,
        default=STRYKETIPSET_DRAW_MAX,
        help=f"Maximum Stryktipset draw number (default: {STRYKETIPSET_DRAW_MAX})",
    )
    parser.add_argument("--after-date", type=date.fromisoformat, default=None)
    parser.add_argument("--before-date", type=date.fromisoformat, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch and upsert even when a snapshot already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch/parse without writing to the database",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Only resolve candidate fixtures (no API calls); for planning counts",
    )
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=0.25,
        help="Delay between API-Football requests (default: 0.25)",
    )
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        service = InjuryBackfillService(
            session,
            request_delay_sec=args.request_delay_sec,
        )
        result = service.backfill(
            draw_min=args.draw_min,
            draw_max=args.draw_max,
            after_date=args.after_date,
            before_date=args.before_date,
            limit=args.limit,
            force_refresh=args.force_refresh,
            dry_run=args.dry_run,
            skip_http=args.skip_http,
        )
    finally:
        session.close()

    logger.info(
        "requested=%s imported=%s updated=%s skipped=%s no_data=%s failed=%s dry_run=%s",
        result.requested,
        result.imported,
        result.updated,
        result.skipped,
        result.no_data,
        result.failed,
        args.dry_run,
    )
    if result.fixture_ids:
        logger.info("fixture_ids sample (up to 10): %s", result.fixture_ids[:10])


if __name__ == "__main__":
    main()
