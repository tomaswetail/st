from __future__ import annotations

from datetime import datetime
from typing import Any


def to_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def result_by_type(
    results: list[dict[str, Any]] | None, type_id: int
) -> dict[str, Any] | None:
    if not results:
        return None
    for result in results:
        if result.get("type") == type_id:
            return result
    return None


def stryktipset_result(
    home_score: int | None, away_score: int | None
) -> str | None:
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "1"
    if home_score == away_score:
        return "X"
    return "2"


# Alias used by fixture / calc code paths.
fixture_result = stryktipset_result


def distribution_by_outcome(
    bet_metrics: dict[str, Any] | None, outcome: str
) -> int | None:
    if not bet_metrics:
        return None
    for value in bet_metrics.get("values") or []:
        if value.get("outcome") == outcome:
            distribution = value.get("distribution") or {}
            return to_int(distribution.get("distribution"))
    return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)
