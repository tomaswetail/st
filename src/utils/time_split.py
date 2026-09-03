"""Shared chronological train/validation split helpers."""

from __future__ import annotations

from typing import Any, TypeVar

DEFAULT_VALIDATION_FRACTION = 0.20

RowT = TypeVar("RowT")


def split_index_for_fraction(row_count: int, validation_fraction: float) -> int:
    """Index separating train prefix from validation suffix (exclusive upper train)."""
    if row_count <= 0:
        return 0
    split_index = max(1, int(row_count * (1.0 - validation_fraction)))
    if split_index >= row_count:
        split_index = row_count - 1
    return split_index


def time_split_rows(
    rows: list[RowT],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    sort_key,
) -> tuple[list[RowT], list[RowT]]:
    """Chronological train/validation split on the last ``validation_fraction`` of rows."""
    ordered = sorted(rows, key=sort_key)
    if not ordered:
        return [], []
    split_index = split_index_for_fraction(len(ordered), validation_fraction)
    return ordered[:split_index], ordered[split_index:]


def time_split_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time-split ML dataset rows by ``(match_date, match_id)``."""
    return time_split_rows(
        rows,
        validation_fraction=validation_fraction,
        sort_key=lambda row: (row.get("match_date", ""), row.get("match_id", 0)),
    )
