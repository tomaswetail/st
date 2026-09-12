"""Shared kickoff/teams view for residual-ML feature calculators.

ST coupon matches use ``teams.id``. Historical fixtures use API-Football
``teams.external_id`` on ``home_team_id`` / ``away_team_id``. Callers must
resolve fixture teams to ``TeamModel`` before building this view so
``home_team.id`` is always the internal PK.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from src.utils.datetime_tz import align_datetime_tzinfo
from src.utils.fixture_fields import fixture_away_name, fixture_home_name


@dataclass(frozen=True)
class MatchFeatureContext:
    """Kickoff, cutoff, and resolved teams for one assemble() call."""

    identity: int
    kickoff: datetime
    cutoff: date
    home_team: Any
    away_team: Any
    league_name: str | None
    league_country: str | None
    league_external_id: int | None
    fixture_pk: int | None
    source: Literal["stryktipset", "fixture"]

    @property
    def home_team_internal_id(self) -> int:
        return int(self.home_team.id)

    @property
    def away_team_internal_id(self) -> int:
        return int(self.away_team.id)

    @property
    def home_team_name(self) -> str:
        return str(self.home_team.name)

    @property
    def away_team_name(self) -> str:
        return str(self.away_team.name)


def is_fixture_shaped(obj: Any) -> bool:
    """True when ``obj`` looks like a historical fixture, not an ST coupon match."""
    return getattr(obj, "fixture_date", None) is not None


def as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def kickoff_datetime(obj: Any) -> datetime:
    if is_fixture_shaped(obj):
        value = getattr(obj, "fixture_date", None)
        if value is None:
            raise ValueError(
                f"Missing fixture_date on fixture id={getattr(obj, 'id', None)}"
            )
        return as_datetime(value)
    value = getattr(obj, "start_time", None)
    if value is None:
        raise ValueError(f"Missing start_time on match id={getattr(obj, 'id', None)}")
    return as_datetime(value)


def resolve_cutoff_date(
    obj: Any,
    *,
    before_date: date | datetime | None = None,
    before: date | datetime | None = None,
) -> date:
    if before_date is not None:
        return as_date(before_date)
    if before is not None:
        return as_date(before)
    return kickoff_datetime(obj).date()


def availability_cutoff_datetime(
    obj: Any,
    *,
    before_date: date | datetime | None = None,
    context: MatchFeatureContext | None = None,
) -> datetime:
    """Use kickoff datetime unless an explicit ``before_date`` differs from it."""
    kickoff = context.kickoff if context is not None else kickoff_datetime(obj)
    cutoff = (
        context.cutoff
        if context is not None and before_date is None
        else resolve_cutoff_date(obj, before_date=before_date)
    )
    if before_date is not None and cutoff != kickoff.date():
        cutoff_dt = datetime.combine(cutoff, datetime.min.time())
        cutoff_dt, _kickoff = align_datetime_tzinfo(cutoff_dt, kickoff)
        return cutoff_dt
    return kickoff


def require_teams(
    obj: Any, *, context: MatchFeatureContext | None = None
) -> tuple[Any, Any]:
    if context is not None:
        if context.home_team is None or context.away_team is None:
            raise ValueError(f"Missing team on match id={context.identity}")
        return context.home_team, context.away_team
    home = getattr(obj, "home_team", None)
    away = getattr(obj, "away_team", None)
    if home is None or away is None:
        raise ValueError(f"Missing team on match id={getattr(obj, 'id', None)}")
    return home, away


def team_side_name(
    obj: Any, *, side: Literal["home", "away"], context: MatchFeatureContext | None
) -> str:
    if context is not None:
        return context.home_team_name if side == "home" else context.away_team_name
    team = getattr(obj, f"{side}_team", None)
    name = getattr(team, "name", None) if team is not None else None
    if name:
        return str(name)
    if side == "home":
        return fixture_home_name(obj)
    return fixture_away_name(obj)
