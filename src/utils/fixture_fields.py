"""Helpers for reading FixtureModel / Fixture schema fields consistently."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from objects.repositories.fixture_repository import fixture_result


def fixture_match_date(row: Any) -> date:
    """Return the calendar date of a fixture row."""
    value = getattr(row, "fixture_date", None)
    if value is None:
        value = getattr(row, "match_date", None)
    if isinstance(value, datetime):
        return value.date()
    return value


def fixture_home_name(row: Any) -> str:
    name = getattr(row, "home_team_name", None)
    if name:
        return str(name)
    side = getattr(row, "home_team", None)
    if side is None:
        return ""
    if isinstance(side, str):
        return side
    return str(getattr(side, "name", "") or "")


def fixture_away_name(row: Any) -> str:
    name = getattr(row, "away_team_name", None)
    if name:
        return str(name)
    side = getattr(row, "away_team", None)
    if side is None:
        return ""
    if isinstance(side, str):
        return side
    return str(getattr(side, "name", "") or "")


def fixture_goals_home(row: Any) -> int | None:
    if hasattr(row, "goals_home"):
        return row.goals_home
    return getattr(row, "home_goals", None)


def fixture_goals_away(row: Any) -> int | None:
    if hasattr(row, "goals_away"):
        return row.goals_away
    return getattr(row, "away_goals", None)


def fixture_outcome(row: Any) -> str | None:
    """Derive 1/X/2 from goals (or legacy result if present)."""
    legacy = getattr(row, "result", None)
    if legacy is not None:
        normalized = str(legacy).strip().upper()
        if normalized in {"1", "X", "2"}:
            return normalized
        if normalized == "D":
            return "X"
    return fixture_result(fixture_goals_home(row), fixture_goals_away(row))


def fixture_went_to_extra_time(row: Any) -> bool:
    status = str(getattr(row, "status_short", "") or "").upper()
    if status in {"AET", "PEN"}:
        return True
    if getattr(row, "score_extratime_home", None) is not None:
        return True
    if getattr(row, "score_extratime_away", None) is not None:
        return True
    return False
