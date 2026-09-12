#!/usr/bin/env python3
"""Print 1X2 probabilities for a Stryktipset draw using the market + ML pipeline.

Usage:
    python -m src.scripts.calculate_probabilities --draw-number 4950
"""

from __future__ import annotations

import argparse

from src.calc.probability_manager import ProbabilityManager
from src.database import SessionLocal, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-number", type=int, required=True)
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        for result in ProbabilityManager(session).process(args.draw_number):
            probs = result.final_probabilities
            print(
                f"{result.event_number:>2}. {result.home_team} vs {result.away_team} "
                f"1={probs['1']:.3f} X={probs['X']:.3f} 2={probs['2']:.3f}",
                flush=True,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
