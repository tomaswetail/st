#!/usr/bin/env python3
"""Fit classic Dixon–Coles for one league as of a cutoff date."""

from __future__ import annotations

import argparse
from datetime import date

from src.calc.dixon_coles import DixonColesService
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id",
        type=int,
        required=True,
        help="API-Football league id (fixtures.league_id)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        required=True,
        help="Fit using matches strictly before this date (YYYY-MM-DD)",
    )
    parser.add_argument("--xi", type=float, default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        service = DixonColesService(session, config=DataSourceConfig())
        model = service.fit_league(
            args.league_id,
            args.as_of,
            xi=args.xi,
            lookback_days=args.lookback_days,
            rho=args.rho,
        )
        print(f"league_id={args.league_id} as_of={args.as_of}")
        print(f"teams={len(model.team_ids)} training_matches={model.n_training_matches}")
        print(f"xi={model.xi} lookback_days={model.lookback_days} rho={model.rho}")
        print(f"home_advantage={model.home_advantage:.4f}")
        if len(model.team_ids) >= 2:
            home_id, away_id = model.team_ids[0], model.team_ids[1]
            prediction = model.predict(home_id, away_id)
            print(
                f"sample {home_id} vs {away_id}: "
                f"λ=({prediction.lambda_home:.3f},{prediction.lambda_away:.3f}) "
                f"1X2=({prediction.p_home:.3f},{prediction.p_draw:.3f},{prediction.p_away:.3f})"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
