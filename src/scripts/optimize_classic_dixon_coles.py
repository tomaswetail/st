#!/usr/bin/env python3
"""Per-league walk-forward grid search for classic Dixon–Coles hyperparameters."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.calc.dixon_coles.optimizer import DixonColesOptimizer, group_eval_matches_by_league
from src.calc.dixon_coles.service import DixonColesService
from src.calc.dixon_coles.walk_forward import EvalMatch
from config.eval_protocol import TUNING_DRAW_MAX
from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from src.data_sources.classic_dc_config import (
    ClassicDcLeagueParams,
    default_league_params_path,
    load_optimization_grid,
    write_league_params,
)
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path
from src.utils.time_split import DEFAULT_VALIDATION_FRACTION, time_split_rows

DEFAULT_DRAW_MIN = STRYKETIPSET_DRAW_MIN
DEFAULT_DRAW_MAX = TUNING_DRAW_MAX


def resolve_optimize_fit_rho(
    *,
    cli_fit_rho: bool | None,
    grid_fit_rho: bool,
    config_fit_rho: bool,
) -> bool:
    """Resolve MLE-ρ for an optimize run.

    An explicit ``--fit-rho`` / ``--no-fit-rho`` wins. Otherwise live config
    (``DataSourceConfig.classic_dc_fit_rho``, default True) or the grid JSON
    ``fit_rho`` key enable MLE ρ, so a missing grid key cannot silently
    grid-search when the live default is on.
    """
    if cli_fit_rho is not None:
        return bool(cli_fit_rho)
    return bool(config_fit_rho) or bool(grid_fit_rho)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-min", type=int, default=DEFAULT_DRAW_MIN)
    parser.add_argument(
        "--draw-max",
        type=int,
        default=DEFAULT_DRAW_MAX,
        help=(
            f"Maximum draw number (default {DEFAULT_DRAW_MAX}=TUNING_DRAW_MAX; "
            f"use {STRYKETIPSET_DRAW_MAX} with --include-holdout for Phase 5)"
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument("--from", dest="date_from", type=date.fromisoformat)
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("config/classic_dc_optimization_grid.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_league_params_path(),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes for per-league optimization (default: 1)",
    )
    parser.add_argument(
        "--fit-rho",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Estimate rho by MLE inside each fit instead of grid-searching it "
            "(drops the rho grid dimension). Default follows "
            "DataSourceConfig.classic_dc_fit_rho and/or the grid JSON fit_rho "
            "key. Use --no-fit-rho to force a rho grid search."
        ),
    )
    return parser


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


def filter_eval_matches_by_date(
    eval_matches: list[EvalMatch],
    *,
    date_from: date,
    date_to: date,
) -> list[EvalMatch]:
    return [
        match
        for match in eval_matches
        if date_from <= match.match_date <= date_to
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.date_from and args.date_to and args.date_from > args.date_to:
        raise SystemExit("--from must be on or before --to")

    init_db()
    grid = load_optimization_grid(resolve_repo_path(args.grid_config))
    config = DataSourceConfig()
    fit_rho = resolve_optimize_fit_rho(
        cli_fit_rho=args.fit_rho,
        grid_fit_rho=grid.fit_rho,
        config_fit_rho=config.classic_dc_fit_rho,
    )
    session = SessionLocal()
    try:
        service = DixonColesService(session, config=config)
        all_eval = service.load_all_stryktipset_eval_matches(
            date_from=date(2000, 1, 1),
            date_to=date(2100, 1, 1),
            min_draw_number=args.draw_min,
            max_draw_number=args.draw_max,
        )
        if args.date_from and args.date_to:
            validation_matches = filter_eval_matches_by_date(
                all_eval,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            split_description = f"dates {args.date_from}–{args.date_to}"
        else:
            _train_matches, validation_matches = time_split_eval_matches(
                all_eval,
                validation_fraction=args.validation_fraction,
            )
            split_description = (
                f"time-split validation fraction={args.validation_fraction:g}"
            )

        if not validation_matches:
            raise SystemExit("No validation matches after filters")

        validation_start = min(match.match_date for match in validation_matches)
        validation_end = max(match.match_date for match in validation_matches)
        eval_by_league = group_eval_matches_by_league(validation_matches)
        league_ids = sorted(eval_by_league)
        max_lookback = max(grid.lookback_values)
        fixtures_by_league = service.preload_league_fixtures(
            league_ids,
            before_date=validation_end,
            max_lookback_days=max_lookback,
        )

        print(
            f"Optimizing {len(league_ids)} leagues on {len(validation_matches)} "
            f"validation matches ({split_description})",
            flush=True,
        )

        optimizer = DixonColesOptimizer()
        result = optimizer.optimize(
            eval_matches_by_league=eval_by_league,
            fixtures_by_league=fixtures_by_league,
            validation_start=validation_start,
            validation_end=validation_end,
            xi_values=grid.xi_values,
            lookback_values=grid.lookback_values,
            rho_values=grid.rho_values,
            min_training_matches=grid.min_training_matches,
            min_team_matches=grid.min_team_matches,
            min_eval_matches_per_league=grid.min_eval_matches_per_league,
            max_goals=config.dixon_coles_max_goals,
            jobs=max(1, args.jobs),
            fit_rho=fit_rho,
            rho_min=config.classic_dc_rho_min,
            rho_max=config.classic_dc_rho_max,
        )

        output_path = resolve_repo_path(args.output)
        league_params = {
            league_id: ClassicDcLeagueParams(
                xi=league_result.best_xi,
                lookback=league_result.best_lookback,
                rho=league_result.best_rho,
                log_loss=league_result.log_loss,
                rps=league_result.rps,
                n_evaluated=league_result.evaluated_matches,
                validation_start=league_result.validation_start.isoformat()
                if league_result.validation_start
                else None,
                validation_end=league_result.validation_end.isoformat()
                if league_result.validation_end
                else None,
            )
            for league_id, league_result in result.league_results.items()
        }
        write_league_params(league_params, output_path)

        print(f"Wrote per-league params to {output_path}", flush=True)
        print(f"Pooled weighted DC log loss: {result.pooled_log_loss:.4f}", flush=True)
        for league_id, league_result in sorted(result.league_results.items()):
            print(
                f"league={league_id} best xi={league_result.best_xi} "
                f"lookback={league_result.best_lookback} rho={league_result.best_rho} "
                f"log_loss={league_result.log_loss:.4f} rps={league_result.rps:.4f} "
                f"scored={league_result.evaluated_matches} "
                f"skipped={league_result.skipped_matches} "
                f"{league_result.rho_diagnostics.summary()}",
                flush=True,
            )
            for index, row in enumerate(league_result.top_parameter_results(3), start=1):
                print(
                    f"  top{index}: xi={row.xi} lookback={row.lookback} rho={row.rho} "
                    f"log_loss={row.log_loss:.4f} rps={row.rps:.4f}",
                    flush=True,
                )
        if result.skipped_leagues:
            print("Skipped leagues:", flush=True)
            for league_id, reason in sorted(result.skipped_leagues.items()):
                print(f"  league={league_id}: {reason}", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
