#!/usr/bin/env python3
"""Chronological OOS backtest / parameter tuner for the coupon optimizer.

```bash
python -m src.scripts.backtest_stryktipset_optimizer --fixture-dir path/to/rounds/
python -m src.scripts.backtest_stryktipset_optimizer --min-draw 4900 --max-draw 4950
```

Fixture dir: one JSON file per round (same schema as optimize CLI fixture).
Ordering: min(start_time) else draw_number — never shuffled.

Leakage UNKNOWN documented in output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.calc.stryktipset_optimizer.backtester import (
    ParamConfig,
    run_backtest,
)
from src.calc.stryktipset_optimizer.data import (
    CouponMatchInput,
    LEAKAGE_LIMITATIONS,
    prepare_matches,
)
from src.utils.repo_paths import resolve_repo_path


def _parse_start_time(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _coupon_from_payload(payload: dict[str, Any], coupon_size: int) -> Any:
    matches_raw = payload.get("matches") or []
    inputs: list[CouponMatchInput] = []
    for index, item in enumerate(matches_raw):
        inputs.append(
            CouponMatchInput(
                odds_1=float(item["odds_1"]),
                odds_x=float(item.get("odds_x", item.get("odds_X"))),
                odds_2=float(item["odds_2"]),
                public_1=item["public_1"],
                public_x=item.get("public_x", item.get("public_X")),
                public_2=item["public_2"],
                match_index=index,
                external_id=item.get("external_id"),
                label=item.get("label"),
                start_time=_parse_start_time(item.get("start_time")),
                result=item.get("result"),
            )
        )
    return prepare_matches(
        inputs,
        expected_count=coupon_size,
        draw_number=payload.get("draw_number"),
    )


def _default_grid() -> list[ParamConfig]:
    grid: list[ParamConfig] = []
    for beta in (0.5, 1.0, 1.5):
        for lambda_div in (0.0, 0.5, 1.0):
            for cand in (100, 500):
                grid.append(
                    ParamConfig(
                        beta=beta,
                        lambda_diversity=lambda_div,
                        banker_value_weight=1.0,
                        candidate_count=cand,
                        row_count=3,
                    )
                )
    return grid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="Directory of per-round JSON fixtures",
    )
    parser.add_argument("--min-draw", type=int, default=None)
    parser.add_argument("--max-draw", type=int, default=None)
    parser.add_argument("--coupon-size", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-simulations", type=int, default=200)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    coupons = []

    if args.fixture_dir is not None:
        fixture_dir = resolve_repo_path(args.fixture_dir)
        for path in sorted(fixture_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                coupons.append(_coupon_from_payload(payload, args.coupon_size))
            except Exception as exc:
                print(f"skip {path.name}: {exc}", flush=True)
    else:
        from src.calc.stryktipset_optimizer.data import load_rounds_for_backtest
        from src.database import SessionLocal, init_db

        init_db()
        session = SessionLocal()
        try:
            coupons = load_rounds_for_backtest(
                session,
                min_draw_number=args.min_draw,
                max_draw_number=args.max_draw,
                expected_count=args.coupon_size,
            )
        finally:
            session.close()

    print(f"Loaded {len(coupons)} coupons", flush=True)
    print(f"Limitations: {LEAKAGE_LIMITATIONS}", flush=True)

    result = run_backtest(
        coupons,
        _default_grid(),
        seed=args.seed,
        n_simulations=args.n_simulations,
    )

    for item in result.results:
        p = item.params
        print(
            f"beta={p.beta} λ={p.lambda_diversity} C={p.candidate_count} "
            f"n={item.rounds_evaluated} "
            f"mean_portfolio_leverage={item.mean_portfolio_leverage:.4f} "
            f"mean_top_row_score(diag)={item.mean_top_row_score:.4f} "
            f"mean_correct(diag)={item.mean_correct:.4f}",
            flush=True,
        )
    if result.best_by_mean_portfolio_leverage is not None:
        best = result.best_by_mean_portfolio_leverage
        print(
            f"Best by mean_portfolio_leverage: beta={best.beta} "
            f"λ={best.lambda_diversity} C={best.candidate_count}",
            flush=True,
        )

    if args.json is not None:
        out_path = resolve_repo_path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "limitations": result.limitations,
            "best_by_mean_portfolio_leverage": (
                None
                if result.best_by_mean_portfolio_leverage is None
                else result.best_by_mean_portfolio_leverage.__dict__
            ),
            "results": [
                {
                    "params": item.params.__dict__,
                    "rounds_evaluated": item.rounds_evaluated,
                    "rounds_skipped": item.rounds_skipped,
                    "mean_portfolio_leverage": item.mean_portfolio_leverage,
                    "mean_top_row_score_diagnostic": item.mean_top_row_score,
                    "mean_realized_row_score_diagnostic": item.mean_realized_row_score,
                    "mean_correct_diagnostic": item.mean_correct,
                    "mean_tier13_rate_diagnostic": item.mean_tier13_rate,
                }
                for item in result.results
            ],
        }
        out_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
