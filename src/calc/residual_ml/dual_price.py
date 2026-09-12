"""Dual opening/closing market columns for residual-ML dataset rows."""

from __future__ import annotations

from typing import Any

from src.calc.residual_ml.baseline import market_baseline
from src.objects.schema.data_classes.market_probability import MarketProbabilityBreakdown

PRICE_TYPES = ("opening", "closing")

MARKET_TRIPLE_KEYS = ("p_home_market", "p_draw_market", "p_away_market")
MARKET_NORM_KEYS = (
    "p_home_market_norm",
    "p_draw_market_norm",
    "p_away_market_norm",
)
MARKET_SHAPE_KEYS = (
    "market_overround",
    "market_entropy",
    "market_top_probability",
    "market_second_probability",
    "market_probability_gap",
    "market_balance",
)
MARKET_FAVOURITE_KEYS = ("favourite_strength", "missing_value_x_favourite")

DC_BLEND_COLUMN_PREFIXES = (
    "p_home_dc",
    "p_draw_dc",
    "p_away_dc",
    "market_vs_dc_",
    "p_home_blend",
    "p_draw_blend",
    "p_away_blend",
    "blend_",
    "expected_home_goals",
    "expected_away_goals",
)


def dual_price_column_names() -> list[str]:
    names: list[str] = []
    for price_type in PRICE_TYPES:
        for key in (
            *MARKET_TRIPLE_KEYS,
            *MARKET_NORM_KEYS,
            *MARKET_SHAPE_KEYS,
            *MARKET_FAVOURITE_KEYS,
        ):
            names.append(f"{key}_{price_type}")
    return names


def empty_dual_price_columns() -> dict[str, None]:
    return {name: None for name in dual_price_column_names()}


def _favourite_strength(p_home: float, p_away: float) -> float:
    return max(p_home, p_away)


def rematerialize_favourite_fields(
    p_home: float | None,
    p_away: float | None,
    missing_value_difference: float | None,
) -> tuple[float | None, float | None]:
    """Favourite strength and missing×favourite from a universe triple."""
    if p_home is None or p_away is None:
        return None, None
    favourite = _favourite_strength(float(p_home), float(p_away))
    if missing_value_difference is None:
        return favourite, None
    return favourite, float(missing_value_difference) * favourite


def price_block_from_breakdown(
    breakdown: MarketProbabilityBreakdown | None,
    *,
    price_type: str,
    missing_value_difference: float | None,
) -> dict[str, Any]:
    """Vig-free triple + shape + favourite fields for one price, or nulls."""
    suffix = f"_{price_type}"
    empty = {
        f"{key}{suffix}": None
        for key in (
            *MARKET_TRIPLE_KEYS,
            *MARKET_NORM_KEYS,
            *MARKET_SHAPE_KEYS,
            *MARKET_FAVOURITE_KEYS,
        )
    }
    if breakdown is None:
        return empty
    market = market_baseline(
        {
            "1": breakdown.p_home,
            "X": breakdown.p_draw,
            "2": breakdown.p_away,
        }
    )
    if market is None:
        return empty
    p_home = market["1"]
    p_draw = market["X"]
    p_away = market["2"]
    favourite, missing_x_favourite = rematerialize_favourite_fields(
        p_home, p_away, missing_value_difference
    )
    return {
        f"p_home_market{suffix}": p_home,
        f"p_draw_market{suffix}": p_draw,
        f"p_away_market{suffix}": p_away,
        f"p_home_market_norm{suffix}": p_home,
        f"p_draw_market_norm{suffix}": p_draw,
        f"p_away_market_norm{suffix}": p_away,
        f"market_overround{suffix}": breakdown.overround,
        f"market_entropy{suffix}": breakdown.market_entropy,
        f"market_top_probability{suffix}": breakdown.market_top_probability,
        f"market_second_probability{suffix}": breakdown.market_second_probability,
        f"market_probability_gap{suffix}": breakdown.market_probability_gap,
        f"market_balance{suffix}": abs(p_home - p_away),
        f"favourite_strength{suffix}": favourite,
        f"missing_value_x_favourite{suffix}": missing_x_favourite,
    }


def price_block_is_usable(block: dict[str, Any], price_type: str) -> bool:
    suffix = f"_{price_type}"
    return all(
        block.get(f"{key}{suffix}") is not None for key in MARKET_NORM_KEYS
    )


def remap_universe_row(row: dict[str, Any], universe: str) -> dict[str, Any]:
    """Copy the chosen triple/shape onto unsuffixed names and rematerialize favourite."""
    remapped = dict(row)
    if universe in PRICE_TYPES:
        suffix = f"_{universe}"
        for key in (
            *MARKET_TRIPLE_KEYS,
            *MARKET_NORM_KEYS,
            *MARKET_SHAPE_KEYS,
            *MARKET_FAVOURITE_KEYS,
        ):
            remapped[key] = row.get(f"{key}{suffix}")
    favourite, missing_x_favourite = rematerialize_favourite_fields(
        remapped.get("p_home_market_norm"),
        remapped.get("p_away_market_norm"),
        remapped.get("missing_value_difference"),
    )
    remapped["favourite_strength"] = favourite
    remapped["missing_value_x_favourite"] = missing_x_favourite
    if remapped.get("p_home_market_norm") is not None and remapped.get(
        "p_away_market_norm"
    ) is not None:
        remapped["market_balance"] = abs(
            float(remapped["p_home_market_norm"])
            - float(remapped["p_away_market_norm"])
        )
    return remapped


def has_universe_triple(row: dict[str, Any], universe: str) -> bool:
    if universe in PRICE_TYPES:
        suffix = f"_{universe}"
        return all(row.get(f"{key}{suffix}") is not None for key in MARKET_NORM_KEYS)
    return all(row.get(key) is not None for key in MARKET_NORM_KEYS)


def is_dc_or_blend_column(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in DC_BLEND_COLUMN_PREFIXES)
