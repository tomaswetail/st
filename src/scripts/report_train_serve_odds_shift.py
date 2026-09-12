#!/usr/bin/env python3
"""Quantify ST vig-free vs archive Avg-closing 1X2 shift (eval only).

Limitation, not a ship-gate. Live scoring stays Svenska Spel (DEC-002 / DEC-016).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import text

from src.calc.archive_odds_join import resolve_archive_fixture_for_st_match
from src.calc.market_probabilities import (
    MarketProbabilities,
    load_fixture_market_probabilities,
)
from src.calc.probability_metrics import mean_log_loss, predicted_label
from src.calc.residual_ml.odds_shift import absolute_probability_deltas
from src.objects.repositories.st_match_repository import STMatchRepository
from src.utils.repo_paths import resolve_repo_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
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


def summarize_train_serve_shift(session) -> dict:
    matches = STMatchRepository(session).find_settled_with_odds()
    n_settled = len(matches)
    skip_reasons = {
        "missing_start_time": 0,
        "unresolved_team_or_fixture": 0,
        "missing_archive_odds": 0,
        "missing_st_odds": 0,
    }
    labels: list[str] = []
    st_probs: list[tuple[float, float, float]] = []
    archive_probs: list[tuple[float, float, float]] = []
    abs_deltas: list[tuple[float, float, float]] = []
    argmax_differ = 0
    st_overrounds: list[float] = []
    archive_overrounds: list[float] = []

    for match in matches:
        odds = match.match_odds
        if odds is None:
            skip_reasons["missing_st_odds"] += 1
            continue
        if match.start_time is None:
            skip_reasons["missing_start_time"] += 1
            continue
        try:
            st_breakdown = MarketProbabilities.from_decimal_odds(
                odds.odds_1, odds.odds_X, odds.odds_2
            )
        except (TypeError, ValueError):
            skip_reasons["missing_st_odds"] += 1
            continue
        fixture = resolve_archive_fixture_for_st_match(session, match)
        if fixture is None:
            skip_reasons["unresolved_team_or_fixture"] += 1
            continue
        archive = load_fixture_market_probabilities(session, fixture.id)
        if archive is None:
            skip_reasons["missing_archive_odds"] += 1
            continue
        label = str(match.stryktipset_result).strip().upper()
        st_triple = (st_breakdown.p_home, st_breakdown.p_draw, st_breakdown.p_away)
        archive_triple = (archive.p_home, archive.p_draw, archive.p_away)
        labels.append(label)
        st_probs.append(st_triple)
        archive_probs.append(archive_triple)
        abs_deltas.append(absolute_probability_deltas(st_triple, archive_triple))
        if predicted_label(*st_triple) != predicted_label(*archive_triple):
            argmax_differ += 1
        st_overrounds.append(st_breakdown.overround)
        archive_overrounds.append(archive.overround)

    n_joined = len(labels)
    if n_joined == 0:
        raise ValueError("no settled ST matches joined to archive Avg closing odds")

    mean_abs = (
        sum(delta[0] for delta in abs_deltas) / n_joined,
        sum(delta[1] for delta in abs_deltas) / n_joined,
        sum(delta[2] for delta in abs_deltas) / n_joined,
    )
    return {
        "n_settled": n_settled,
        "n_joined": n_joined,
        "join_rate": n_joined / n_settled if n_settled else 0.0,
        "skip_reasons": skip_reasons,
        "mean_abs_delta_home": mean_abs[0],
        "mean_abs_delta_draw": mean_abs[1],
        "mean_abs_delta_away": mean_abs[2],
        "argmax_differ_share": argmax_differ / n_joined,
        "log_loss_st": mean_log_loss(labels, st_probs),
        "log_loss_archive": mean_log_loss(labels, archive_probs),
        "mean_overround_st": sum(st_overrounds) / n_joined,
        "mean_overround_archive": sum(archive_overrounds) / n_joined,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        session = _open_session()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        summary = summarize_train_serve_shift(session)
    except Exception as exc:
        print(f"Train/serve shift failed: {exc}", file=sys.stderr)
        session.close()
        return 1
    session.close()

    print(
        f"n_settled={summary['n_settled']}  n_joined={summary['n_joined']}  "
        f"join_rate={summary['join_rate']:.1%}",
        flush=True,
    )
    print(f"skip_reasons={summary['skip_reasons']}", flush=True)
    print(
        "mean |Δp| home/draw/away = "
        f"{summary['mean_abs_delta_home']:.4f} / "
        f"{summary['mean_abs_delta_draw']:.4f} / "
        f"{summary['mean_abs_delta_away']:.4f}",
        flush=True,
    )
    print(f"argmax_differ_share={summary['argmax_differ_share']:.4f}", flush=True)
    print(
        f"log_loss ST={summary['log_loss_st']:.4f}  "
        f"archive={summary['log_loss_archive']:.4f}",
        flush=True,
    )
    print(
        f"mean overround ST={summary['mean_overround_st']:.4f}  "
        f"archive={summary['mean_overround_archive']:.4f}",
        flush=True,
    )
    print(
        "Limitation only — live coupon scoring stays Svenska Spel. "
        "Does not move the 518-row ship gate.",
        flush=True,
    )

    if args.json is not None:
        json_path = resolve_repo_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
