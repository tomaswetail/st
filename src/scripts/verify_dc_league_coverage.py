#!/usr/bin/env python3
"""Verify classic DC league params cover all eligible validation leagues."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from src.calc.dixon_coles.optimizer import group_eval_matches_by_league
from src.calc.dixon_coles.service import DixonColesService
from src.calc.dixon_coles.walk_forward import EvalMatch
from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from src.data_sources.classic_dc_config import default_league_params_path, load_league_params
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path
from src.utils.time_split import DEFAULT_VALIDATION_FRACTION, time_split_rows


def time_split_eval_matches(
    eval_matches: list[EvalMatch],
    *,
    validation_fraction: float,
) -> tuple[list[EvalMatch], list[EvalMatch]]:
    return time_split_rows(
        eval_matches,
        validation_fraction=validation_fraction,
        sort_key=lambda match: (match.match_date, match.home_team_external_id),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-min", type=int, default=STRYKETIPSET_DRAW_MIN)
    parser.add_argument("--draw-max", type=int, default=STRYKETIPSET_DRAW_MAX)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--min-eval-matches",
        type=int,
        default=15,
        help="Minimum validation matches for a league to require tuned params",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=default_league_params_path(),
        help="Path to classic_dc_league_params.json",
    )
    args = parser.parse_args()

    init_db()
    config = DataSourceConfig()
    session = SessionLocal()
    try:
        service = DixonColesService(session, config=config)
        all_eval = service.load_all_stryktipset_eval_matches(
            date_from=date(2000, 1, 1),
            date_to=date(2100, 1, 1),
            min_draw_number=args.draw_min,
            max_draw_number=args.draw_max,
        )
        _train_matches, validation_matches = time_split_eval_matches(
            all_eval,
            validation_fraction=args.validation_fraction,
        )
        if not validation_matches:
            print("No validation matches after filters", flush=True)
            return 1

        eval_by_league = group_eval_matches_by_league(validation_matches)
        params_path = resolve_repo_path(args.params)
        tuned_params = load_league_params(params_path)

        print(
            f"Validation slice: {len(validation_matches)} matches, "
            f"{len(eval_by_league)} leagues "
            f"(fraction={args.validation_fraction:g}, draws {args.draw_min}–{args.draw_max})",
            flush=True,
        )
        print(f"Params file: {params_path} ({len(tuned_params)} tuned leagues)", flush=True)
        print("", flush=True)

        eligible: list[tuple[int, int]] = []
        skipped: list[tuple[int, int, str]] = []
        tuned: list[tuple[int, int]] = []
        missing: list[tuple[int, int]] = []

        for league_id in sorted(eval_by_league):
            match_count = len(eval_by_league[league_id])
            if match_count >= args.min_eval_matches:
                eligible.append((league_id, match_count))
                if league_id in tuned_params:
                    tuned.append((league_id, match_count))
                else:
                    missing.append((league_id, match_count))
            else:
                skipped.append(
                    (league_id, match_count, f"< {args.min_eval_matches} validation matches"),
                )

        print("Eligible leagues (>= min_eval_matches):", flush=True)
        for league_id, match_count in eligible:
            status = "tuned" if league_id in tuned_params else "MISSING"
            print(f"  league={league_id} matches={match_count} status={status}", flush=True)

        if skipped:
            print("\nSkipped leagues:", flush=True)
            for league_id, match_count, reason in skipped:
                print(
                    f"  league={league_id} matches={match_count} reason={reason}",
                    flush=True,
                )

        extra_tuned = sorted(set(tuned_params) - set(eval_by_league))
        if extra_tuned:
            print("\nTuned leagues not in validation slice:", flush=True)
            for league_id in extra_tuned:
                print(f"  league={league_id}", flush=True)

        print("", flush=True)
        print(
            f"Summary: eligible={len(eligible)} tuned={len(tuned)} "
            f"missing={len(missing)} skipped={len(skipped)}",
            flush=True,
        )

        if missing:
            print("\nGap report — eligible leagues without tuned params:", flush=True)
            for league_id, match_count in missing:
                print(f"  league={league_id} matches={match_count}", flush=True)
            return 1

        print("\nAll eligible leagues have tuned params.", flush=True)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
