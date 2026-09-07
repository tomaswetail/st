#!/usr/bin/env python3
"""Optimize Stryktipset coupon rows via market odds + streckprocent.

Uses fair market probabilities as truth and public shares for dilution.
Does NOT call ProbabilityManager / residual ML / Dixon–Coles.

```bash
python -m src.scripts.optimize_stryktipset_coupon --draw-number 4950 --rows 5
python -m src.scripts.optimize_stryktipset_coupon --fixture path/to.json --rows 3 --json out.json
python -m src.scripts.optimize_stryktipset_coupon --fixture path/to.json --simulate --n-simulations 1000 --seed 0
```

Fixture JSON: ``{"draw_number": N, "matches": [{"odds_1", "odds_x", "odds_2",
"public_1", "public_x", "public_2", ...}, ...]}`` (13 matches typical).

Leakage UNKNOWN: stored odds/public % lack timestamps; regCloseTime not on
STRoundModel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.calc.stryktipset_optimizer.data import (
    CouponMatchInput,
    load_coupon_from_session,
    prepare_matches,
)
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.calc.stryktipset_optimizer.simulator import simulate_pool
from src.utils.repo_paths import resolve_repo_path


def _matches_from_fixture(payload: dict[str, Any]) -> list[CouponMatchInput]:
    matches_raw = payload.get("matches")
    if not isinstance(matches_raw, list):
        raise ValueError("fixture JSON must contain a 'matches' list")
    inputs: list[CouponMatchInput] = []
    for index, item in enumerate(matches_raw):
        if not isinstance(item, dict):
            raise ValueError(f"matches[{index}] must be an object")
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
                result=item.get("result"),
            )
        )
    return inputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--draw-number",
        type=int,
        help="Load ST round from database by draw_number",
    )
    source.add_argument(
        "--fixture",
        type=Path,
        help="Load coupon from JSON fixture (DB-less)",
    )
    parser.add_argument("--rows", type=int, default=1, help="Portfolio row count")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-diversity", type=float, default=0.5)
    parser.add_argument("--banker-value-weight", type=float, default=1.0)
    parser.add_argument("--candidate-count", type=int, default=500)
    parser.add_argument("--public-epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--coupon-size",
        type=int,
        default=13,
        help="Expected match count (default 13)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Attach MC relative tier metrics (Phase 2)",
    )
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument(
        "--public-row-count",
        type=int,
        default=1000,
        help="Synthetic public book size for --simulate (default 1000)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    params = OptimizerParams(
        beta=args.beta,
        lambda_diversity=args.lambda_diversity,
        banker_value_weight=args.banker_value_weight,
        candidate_count=args.candidate_count,
        public_epsilon=args.public_epsilon,
        coupon_size=args.coupon_size,
        seed=args.seed,
    )
    optimizer = CouponOptimizer(params)

    if args.fixture is not None:
        fixture_path = resolve_repo_path(args.fixture)
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        draw_number = payload.get("draw_number")
        inputs = _matches_from_fixture(payload)
        coupon = prepare_matches(
            inputs,
            public_epsilon=params.public_epsilon,
            expected_count=params.coupon_size,
            draw_number=draw_number,
        )
    else:
        from src.database import SessionLocal, init_db

        init_db()
        session = SessionLocal()
        try:
            coupon = load_coupon_from_session(
                session,
                args.draw_number,
                public_epsilon=params.public_epsilon,
                expected_count=params.coupon_size,
            )
        finally:
            session.close()

    result = optimizer.optimize_prepared(coupon, row_count=args.rows)

    if args.simulate:
        market_probs = [m.market_probs for m in coupon.matches]
        public_probs = [m.public_probs for m in coupon.matches]
        our_rows = [tuple(row.outcomes) for row in result.rows]
        sim = simulate_pool(
            market_probs,
            public_probs,
            our_rows,
            n_simulations=args.n_simulations,
            seed=args.seed,
            public_row_count=args.public_row_count,
        )
        result = result.model_copy(update={"simulation": sim})

    payload_out = result.model_dump()
    print(
        f"Draw {result.draw_number}: {result.match_count} matches, "
        f"{len(result.rows)} portfolio rows "
        f"(candidates={result.portfolio_analysis.candidate_pool_size})",
        flush=True,
    )
    for row in result.rows:
        print(
            f"  #{row.rank} score={row.row_score:.4f} "
            f"adj={row.adjusted_score:.4f} "
            f"{''.join(row.outcomes)}",
            flush=True,
        )
    if result.simulation is not None:
        print(
            f"MC mean_correct ours={result.simulation.mean_correct_ours:.3f} "
            f"public={result.simulation.mean_correct_public:.3f}",
            flush=True,
        )
    print(f"Limitations: {result.limitations}", flush=True)

    if args.json is not None:
        out_path = resolve_repo_path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload_out, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
