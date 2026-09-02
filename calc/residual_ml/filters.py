"""Row filtering helpers for residual ML backtest."""

from __future__ import annotations

from typing import Any

from utils.time_split import DEFAULT_VALIDATION_FRACTION, time_split_dataset_rows


def _draw_number(row: dict[str, Any]) -> int | None:
    value = row.get("draw_number")
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def filter_rows_by_draw(
    rows: list[dict[str, Any]],
    *,
    min_draw: int | None = None,
    max_draw: int | None = None,
) -> list[dict[str, Any]]:
    """Keep rows whose draw_number falls within an inclusive range."""
    if min_draw is None and max_draw is None:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        draw_number = _draw_number(row)
        if draw_number is None:
            continue
        if min_draw is not None and draw_number < min_draw:
            continue
        if max_draw is not None and draw_number > max_draw:
            continue
        filtered.append(row)
    return filtered


def select_backtest_rows(
    rows: list[dict[str, Any]],
    *,
    all_rows: bool = False,
    validation_fraction: float | None = None,
    min_draw: int | None = None,
    max_draw: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return rows to score and a short description of the filter applied."""
    draw_filtered = filter_rows_by_draw(
        rows, min_draw=min_draw, max_draw=max_draw
    )
    draw_note = ""
    if min_draw is not None or max_draw is not None:
        draw_note = (
            f", draws {min_draw if min_draw is not None else '…'}"
            f"–{max_draw if max_draw is not None else '…'}"
        )

    if all_rows:
        return draw_filtered, f"all rows{draw_note} (includes train; optimistic)"

    fraction = (
        DEFAULT_VALIDATION_FRACTION
        if validation_fraction is None
        else validation_fraction
    )
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {fraction}"
        )
    _, validation_rows = time_split_dataset_rows(
        draw_filtered,
        validation_fraction=fraction,
    )
    return (
        validation_rows,
        f"time-split validation fraction={fraction:g}{draw_note}",
    )
