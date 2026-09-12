#!/usr/bin/env python3
"""PREDICTION / MAX_P13 backtest of probability sources on real ST coupons.

Sources:
  st_market — vig-free Svenska Spel Pm already on the coupon
  archive_market — archive Avg closing odds per match (skip coupon if any miss)

Does not call simulate_pool. Does not change VALUE defaults.
Archive odds are eval-only (DEC-002 / DEC-016). Residual ML stays off.

Realized rank within the selected portfolio is 1-based optimizer order if
the truth row is selected, else N+1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.calc.archive_odds_join import load_archive_market_for_st_match
from src.calc.residual_ml.io import load_dataset_rows
from src.calc.residual_ml.p13 import score_coupons_max_p13
from src.calc.stryktipset_optimizer.data import (
    PreparedCoupon,
    load_rounds_for_backtest,
    with_replaced_market_probs,
)
from src.objects.repositories.st_match_repository import STMatchRepository
from src.utils.repo_paths import resolve_repo_path

DEFAULT_BUDGETS = (64, 128, 256, 512, 1024)
ST_EXTRACT_PATH = Path("data/residual_ml/dataset.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        nargs="*",
        type=int,
        default=list(DEFAULT_BUDGETS),
        help="MAX_P13 row counts (default: 64 128 256 512 1024)",
    )
    parser.add_argument("--min-draw", type=int, default=None)
    parser.add_argument("--max-draw", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for coupon MAX_P13 (default: all CPUs)",
    )
    return parser.parse_args(argv)


def _open_session():
    try:
        from src.database import SessionLocal, init_db

        init_db()
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        return session
    except Exception as exc:
        raise RuntimeError(f"Database unavailable: {exc}") from exc


def _st_market_probs(coupon: PreparedCoupon) -> list[dict[str, float]]:
    return [dict(match.market_probs) for match in coupon.matches]


def _archive_probs_or_none(
    session,
    coupon: PreparedCoupon,
    matches_by_id: dict[int, Any],
) -> list[dict[str, float]] | None:
    probs: list[dict[str, float]] = []
    for match in coupon.matches:
        if match.st_match_id is None:
            return None
        st_match = matches_by_id.get(match.st_match_id)
        if st_match is None:
            return None
        breakdown = load_archive_market_for_st_match(session, st_match)
        if breakdown is None:
            return None
        probs.append(breakdown.as_probs())
    return probs


def _coupon_st_coverage(
    coupons: list[PreparedCoupon],
    dataset_match_ids: set[int],
) -> tuple[int, int]:
    usable = 0
    for coupon in coupons:
        ids = [match.st_match_id for match in coupon.matches]
        if all(match_id in dataset_match_ids for match_id in ids if match_id is not None) and None not in ids:
            usable += 1
    return usable, len(coupons)


def _accumulate(score, sink: dict[str, Any]) -> None:
    sink["n_used"] += 1
    sink["sum_p13"] += score.predicted_p13
    sink["sum_covered"] += 1.0 if score.covered13 else 0.0
    sink["sum_best_correct"] += score.best_correct
    sink["sum_rank"] += score.realized_rank


def _finalize(sink: dict[str, Any], *, n_skipped: int) -> dict[str, Any]:
    n_used = sink["n_used"]
    if n_used == 0:
        return {
            "n_used": 0,
            "n_skipped": n_skipped,
            "mean_predicted_p13": None,
            "covered13_rate": None,
            "mean_best_correct": None,
            "frequency_rank_le_n": None,
            "mean_realized_rank": None,
        }
    covered_rate = sink["sum_covered"] / n_used
    return {
        "n_used": n_used,
        "n_skipped": n_skipped,
        "mean_predicted_p13": sink["sum_p13"] / n_used,
        "covered13_rate": covered_rate,
        "mean_best_correct": sink["sum_best_correct"] / n_used,
        "frequency_rank_le_n": covered_rate,
        "mean_realized_rank": sink["sum_rank"] / n_used,
    }


def _empty_sink() -> dict[str, Any]:
    return {
        "n_used": 0,
        "sum_p13": 0.0,
        "sum_covered": 0.0,
        "sum_best_correct": 0.0,
        "sum_rank": 0.0,
    }


def _format_optional(value: float | None) -> str:
    if value is None:
        return f"{'-':>10}"
    return f"{value:10.4f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = list(args.budgets)
    if not budgets or any(budget < 1 for budget in budgets):
        print("Budgets must be positive integers", file=sys.stderr)
        return 1

    try:
        session = _open_session()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        coupons = load_rounds_for_backtest(
            session,
            min_draw_number=args.min_draw,
            max_draw_number=args.max_draw,
        )
        match_ids = [
            match.st_match_id
            for coupon in coupons
            for match in coupon.matches
            if match.st_match_id is not None
        ]
        matches_by_id = STMatchRepository(session).get_by_ids(match_ids)

        print(
            f"Loaded {len(coupons)} settled coupons. "
            "Realized rank = 1-based position in the selected set if covered, "
            "else N+1.",
            flush=True,
        )

        st_extract = resolve_repo_path(ST_EXTRACT_PATH)
        ml_note = "ML-on-coupon skipped"
        if st_extract.is_file():
            extract_rows = load_dataset_rows(st_extract)
            extract_ids = {
                int(row["match_id"])
                for row in extract_rows
                if row.get("match_id") not in (None, "")
            }
            usable, total = _coupon_st_coverage(coupons, extract_ids)
            coverage = usable / total if total else 0.0
            if coverage < 0.5:
                ml_note = (
                    f"ML-on-coupon skipped: dataset.csv covers {usable}/{total} "
                    f"coupons ({coverage:.1%}); join coverage is poor. "
                    "That CSV is a pre-Phase-3 ST extract."
                )
            else:
                ml_note = (
                    f"ML-on-coupon skipped: dataset.csv covers {usable}/{total} "
                    "coupons but scoring that leftover extract is not required "
                    "and is left off."
                )
        else:
            ml_note = "ML-on-coupon skipped: data/residual_ml/dataset.csv is missing"
        print(ml_note, flush=True)

        archive_coupons: list[PreparedCoupon] = []
        archive_skipped = 0
        for coupon in coupons:
            archive_probs = _archive_probs_or_none(session, coupon, matches_by_id)
            if archive_probs is None:
                archive_skipped += 1
                continue
            archive_coupons.append(with_replaced_market_probs(coupon, archive_probs))
        print(
            f"archive_market: using {len(archive_coupons)} coupons, "
            f"skipped {archive_skipped} (incomplete archive odds).",
            flush=True,
        )

        sources: dict[str, list[PreparedCoupon]] = {
            "st_market": coupons,
            "archive_market": archive_coupons,
        }
        skipped = {
            "st_market": 0,
            "archive_market": archive_skipped,
        }

        results: dict[str, dict[int, dict[str, Any]]] = {}
        print(
            f"{'source':<16} {'N':>5} {'n_used':>7} {'n_skip':>7} "
            f"{'p13':>10} {'cov13':>10} {'best':>8} {'rk<=N':>10} {'rank':>8}",
            flush=True,
        )
        for source_name, source_coupons in sources.items():
            results[source_name] = {}
            for budget in budgets:
                sink = _empty_sink()
                scores = score_coupons_max_p13(
                    source_coupons, row_count=budget, workers=args.workers
                )
                for score in scores:
                    _accumulate(score, sink)
                summary = _finalize(sink, n_skipped=skipped[source_name])
                results[source_name][budget] = summary
                print(
                    f"{source_name:<16} {budget:>5} {summary['n_used']:>7} "
                    f"{summary['n_skipped']:>7} "
                    f"{_format_optional(summary['mean_predicted_p13'])} "
                    f"{_format_optional(summary['covered13_rate'])} "
                    f"{_format_optional(summary['mean_best_correct'])} "
                    f"{_format_optional(summary['frequency_rank_le_n'])} "
                    f"{_format_optional(summary['mean_realized_rank'])}",
                    flush=True,
                )
    finally:
        session.close()

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budgets": budgets,
            "ml_note": ml_note,
            "realized_rank_convention": (
                "1-based index of the truth row in optimizer order if covered, "
                "else N+1"
            ),
            "results": {
                source: {str(budget): metrics for budget, metrics in by_budget.items()}
                for source, by_budget in results.items()
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
