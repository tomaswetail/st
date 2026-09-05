"""Residual ML validation fraction resolution."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.time_split import DEFAULT_VALIDATION_FRACTION


def resolve_validation_fraction(
    *,
    cli_fraction: float | None,
    model_path: Path,
) -> float:
    """Pick validation fraction from CLI, sweep results, or project default."""
    if cli_fraction is not None:
        return cli_fraction

    sweep_path = model_path.parent / "sweep_results.json"
    if sweep_path.exists():
        try:
            payload = json.loads(sweep_path.read_text(encoding="utf-8"))
            best = payload.get("best", {})
            fraction = best.get("validation_fraction")
            if fraction is not None:
                return float(fraction)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    return DEFAULT_VALIDATION_FRACTION
