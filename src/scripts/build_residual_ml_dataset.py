#!/usr/bin/env python3
"""Build residual ML training dataset from ST matches or historical fixtures.

    python -m src.scripts.build_residual_ml_dataset
    python -m src.scripts.build_residual_ml_dataset --source fixtures
    python -m src.scripts.build_residual_ml_dataset --source fixtures \\
        --league E0 --season 2023 --output data/residual_ml/fixtures_e0_2023.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from src.calc.residual_ml import ResidualMLDatasetBuilder
from src.data_sources.football_data_odds.leagues import (
    load_covered_leagues,
    resolve_league_codes,
)
from src.database import SessionLocal, init_db
from src.objects.schema.data_classes.data_sources import DataSourceConfig

ST_CSV_OUTPUT = Path("data/residual_ml/dataset.csv")
FIXTURES_CSV_OUTPUT = Path("data/residual_ml/fixtures_dataset.csv")
FIXTURES_CLOSING_ONLY_BACKUP = Path(
    "data/residual_ml/fixtures_dataset_closing_only_bck.csv"
)
ST_PRE_PHASE3_BACKUP = Path("data/residual_ml/dataset_pre_phase3_bck.csv")


def copy_aside_if_needed(source: Path, backup: Path) -> None:
    """Copy ``source`` to ``backup`` once; do not overwrite an existing backup."""
    if source.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        print(f"Copied existing {source} → {backup}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("stryktipset", "fixtures"),
        default="stryktipset",
        help="Row source (default: stryktipset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV path. Default for stryktipset: data/residual_ml/dataset.csv. "
            "Default for fixtures: data/residual_ml/fixtures_dataset.csv."
        ),
    )
    parser.add_argument(
        "--league",
        default=None,
        help=(
            "Optional for --source fixtures. CSV code (E0), mapped name "
            "(Premier League), or API-Football league id. Omit with --season "
            "to build every fixture that has archive odds."
        ),
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "Optional for --source fixtures. league_season start year "
            "(e.g. 2023). Omit with --league for a full odds-backed build."
        ),
    )
    parser.add_argument(
        "--price-type",
        choices=("opening", "closing"),
        default=None,
        help="Override FIXTURE_ODDS_PRICE_TYPE for this fixtures run.",
    )
    return parser.parse_args(argv)


def default_output_path(source: str, output: Path | None) -> Path:
    if output is not None:
        return output
    if source == "fixtures":
        return FIXTURES_CSV_OUTPUT
    return ST_CSV_OUTPUT


def resolve_fixture_league_external_id(league: str) -> int:
    token = league.strip()
    if not token:
        raise ValueError("League token is empty")
    if token.isdigit():
        return int(token)
    codes = resolve_league_codes(token)
    covered = load_covered_leagues()
    return int(covered[codes[0]]["league_id"])


def build_residual_ml_dataset(argv: list[str] | None = None) -> int:
    """Stream the residual ML dataset CSV."""
    args = parse_args(argv)
    output = default_output_path(args.source, args.output)
    if args.source == "fixtures" and bool(args.league) != (args.season is not None):
        raise SystemExit(
            "--source fixtures needs both --league and --season, or neither "
            "(omit both to build every fixture that has archive odds)"
        )
    if args.source == "fixtures" and output == FIXTURES_CSV_OUTPUT:
        copy_aside_if_needed(FIXTURES_CSV_OUTPUT, FIXTURES_CLOSING_ONLY_BACKUP)
    if args.source == "stryktipset" and output == ST_CSV_OUTPUT:
        copy_aside_if_needed(ST_CSV_OUTPUT, ST_PRE_PHASE3_BACKUP)
    init_db()
    session = SessionLocal()
    try:
        if args.source == "fixtures":
            config = DataSourceConfig()
            dual_price = args.price_type is None
            if args.price_type is not None:
                config = config.model_copy(
                    update={"fixture_odds_price_type": args.price_type}
                )
            builder = ResidualMLDatasetBuilder(session, config=config)
            fixture_kwargs: dict = {"dual_price": dual_price}
            if args.league:
                fixture_kwargs["league_external_id"] = (
                    resolve_fixture_league_external_id(args.league)
                )
                fixture_kwargs["league_season"] = args.season
            rows = builder.iter_fixture_rows(**fixture_kwargs)
        else:
            builder = ResidualMLDatasetBuilder(session)
            rows = builder.iter_rows(
                min_draw_number=STRYKETIPSET_DRAW_MIN,
                max_draw_number=STRYKETIPSET_DRAW_MAX,
            )
        row_count = builder.export_csv_from_iter(output, rows)
        print(f"Wrote {row_count} rows to {output}", flush=True)
        return row_count
    finally:
        session.close()


if __name__ == "__main__":
    build_residual_ml_dataset()
