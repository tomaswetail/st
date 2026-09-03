#!/usr/bin/env python3
"""Walk-forward classic Dixon–Coles vs Stryktipset market for one league."""

from __future__ import annotations

import argparse
from datetime import date

from calc.dixon_coles import DixonColesService
from database import SessionLocal, init_db
from objects.schema.data_classes.data_sources import DataSourceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id",
        type=int,
        required=True,
        help="API-Football league id (fixtures.league_id / leagues.external_id)",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        type=date.fromisoformat,
        required=True,
        help="Inclusive validation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=date.fromisoformat,
        required=True,
        help="Inclusive validation end date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        service = DixonColesService(session, config=DataSourceConfig())
        result = service.walk_forward_league(
            args.league_id,
            args.date_from,
            args.date_to,
        )
        print(
            f"league_id={args.league_id} "
            f"validation={result.validation_start}–{result.validation_end}"
        )
        print(
            f"scored={result.n_scored} skipped={result.n_skipped} "
            f"skip_reasons={result.skip_reasons}"
        )
        print(f"dc log loss:     {result.dc_log_loss:.4f}")
        print(f"market log loss: {result.market_log_loss:.4f}")
        print(f"dc RPS:          {result.dc_rps:.4f}")
        print(f"market RPS:      {result.market_rps:.4f}")
        if result.n_scored == 0:
            print("No scored matches")
        elif result.beats_market:
            print(
                f"DC beats market by "
                f"{result.market_log_loss - result.dc_log_loss:.4f} log loss"
            )
        else:
            print(
                f"DC does not beat market "
                f"(gap {result.dc_log_loss - result.market_log_loss:.4f})"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
