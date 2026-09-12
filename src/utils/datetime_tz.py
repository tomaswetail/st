"""Align naive/aware datetimes for comparison without converting zones."""

from __future__ import annotations

from datetime import datetime


def align_datetime_tzinfo(
    left: datetime, right: datetime
) -> tuple[datetime, datetime]:
    """Copy tzinfo onto the naive side when only one value is aware.

    Does not convert between time zones. Matches the assemble-path pattern
    used in ``archive_odds_join`` and ``fixture_repository``.
    """
    if left.tzinfo is None and right.tzinfo is not None:
        left = left.replace(tzinfo=right.tzinfo)
    elif left.tzinfo is not None and right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    return left, right
