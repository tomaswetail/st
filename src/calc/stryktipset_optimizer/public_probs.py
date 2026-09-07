"""Public streckprocent (Pp) normalization with epsilon floor.

Scale rule (STMatchBetModel ints are 0–100 percentages):
- If any value ``> 1``, treat the triple as **percentage scale (0–100)**,
  including components in ``(0, 1]`` (e.g. ``1`` means 1%, ``0.5`` means 0.5%).
- Unit scale ``[0, 1]`` only when **all** values are in ``[0, 1]``.

Rejects negatives and values ``> 100``. After scale conversion, renormalizes
to sum=1 (sum=0 rejected), then applies an always-on epsilon floor and renorms.
"""

from __future__ import annotations

import math
from typing import Mapping

from src.utils.common import OUTCOMES, Outcome


class InvalidPublicShareError(ValueError):
    """Raised when public bet shares cannot form a valid Pp triple."""


def _as_float(label: str, value: float | int | None) -> float:
    if value is None:
        raise InvalidPublicShareError(f"{label} public share is null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidPublicShareError(
            f"{label} public share is not numeric: {value!r}"
        ) from exc
    if math.isnan(number) or math.isinf(number):
        raise InvalidPublicShareError(
            f"{label} public share is not finite: {value!r}"
        )
    return number


def normalize_public_shares(
    share_1: float | int,
    share_x: float | int,
    share_2: float | int,
    *,
    public_epsilon: float = 1e-6,
) -> dict[Outcome, float]:
    """Normalize streckprocent to unit Pp with epsilon floor then renorm."""
    if public_epsilon <= 0:
        raise InvalidPublicShareError(
            f"public_epsilon must be > 0, got {public_epsilon}"
        )

    values = {
        "1": _as_float("1", share_1),
        "X": _as_float("X", share_x),
        "2": _as_float("2", share_2),
    }
    for outcome, value in values.items():
        if value < 0:
            raise InvalidPublicShareError(
                f"{outcome} public share must be >= 0, got {value}"
            )
        if value > 100:
            raise InvalidPublicShareError(
                f"{outcome} public share must be <= 100, got {value}"
            )

    present = list(values.values())
    # Any value > 1 ⇒ percentage scale (0–100), including 1% integers.
    if any(value > 1.0 for value in present):
        scaled = {outcome: value / 100.0 for outcome, value in values.items()}
    else:
        # All in [0, 1] ⇒ already unit scale.
        scaled = dict(values)

    total = sum(scaled.values())
    if total <= 0:
        raise InvalidPublicShareError("public shares sum to 0; cannot normalize")

    # Renorm even when sum ≠ 1 after scale (documented behavior).
    unit = {outcome: value / total for outcome, value in scaled.items()}

    floored = {
        outcome: max(unit[outcome], public_epsilon) for outcome in OUTCOMES
    }
    floor_total = sum(floored.values())
    return {
        outcome: floored[outcome] / floor_total for outcome in OUTCOMES
    }


def normalize_public_shares_from_mapping(
    shares: Mapping[str, float | int],
    *,
    public_epsilon: float = 1e-6,
) -> dict[Outcome, float]:
    try:
        return normalize_public_shares(
            shares["1"],
            shares["X"],
            shares["2"],
            public_epsilon=public_epsilon,
        )
    except KeyError as exc:
        raise InvalidPublicShareError(
            f"public share mapping missing outcome {exc}"
        ) from exc
