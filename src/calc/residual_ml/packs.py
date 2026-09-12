"""Consecutive 13-match packs from sorted fixture rows (eval only)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Sequence


def _match_date_sort_key(row: dict[str, Any]) -> str:
    match_date = row.get("match_date")
    if isinstance(match_date, datetime):
        return match_date.date().isoformat()
    if isinstance(match_date, date):
        return match_date.isoformat()
    if match_date is None:
        return ""
    return str(match_date)[:10]


def sort_rows_for_packs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable chronological order: ``(match_date, match_id)``."""
    return sorted(
        rows,
        key=lambda row: (_match_date_sort_key(row), row.get("match_id", 0)),
    )


def pack_consecutive_coupons(
    rows: list[dict[str, Any]],
    *,
    coupon_size: int = 13,
) -> list[list[dict[str, Any]]]:
    """Sort rows and pack into consecutive groups of ``coupon_size``.

    Leftover rows ``< coupon_size`` are dropped. Order inside each pack
    follows the sorted row order.
    """
    if coupon_size < 1:
        raise ValueError(f"coupon_size must be >= 1, got {coupon_size}")
    sorted_rows = sort_rows_for_packs(rows)
    n_keep = (len(sorted_rows) // coupon_size) * coupon_size
    kept = sorted_rows[:n_keep]
    return [
        kept[start : start + coupon_size]
        for start in range(0, n_keep, coupon_size)
    ]


def pack_consecutive_pairs(
    rows: list[dict[str, Any]],
    values: Sequence[Any],
    *,
    coupon_size: int = 13,
) -> list[tuple[list[dict[str, Any]], list[Any]]]:
    """Sort ``rows`` with aligned ``values`` and pack into coupon groups."""
    if len(rows) != len(values):
        raise ValueError("rows and values length mismatch")
    paired = list(zip(rows, values))
    paired.sort(
        key=lambda item: (_match_date_sort_key(item[0]), item[0].get("match_id", 0))
    )
    n_keep = (len(paired) // coupon_size) * coupon_size
    kept = paired[:n_keep]
    packs: list[tuple[list[dict[str, Any]], list[Any]]] = []
    for start in range(0, n_keep, coupon_size):
        chunk = kept[start : start + coupon_size]
        packs.append(
            ([row for row, _value in chunk], [value for _row, value in chunk])
        )
    return packs
