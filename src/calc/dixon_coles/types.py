"""Shared types for classic Dixon–Coles goals MLE."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from src.utils.fixture_fields import fixture_match_date


@dataclass(frozen=True)
class DixonColesMatch:
    """One finished match used for goals-based Dixon–Coles fitting."""

    match_date: date
    home_team_id: int
    away_team_id: int
    goals_home: int
    goals_away: int

    @classmethod
    def from_fixture(cls, fixture: Any) -> DixonColesMatch:
        goals_home = fixture.goals_home
        goals_away = fixture.goals_away
        if goals_home is None or goals_away is None:
            raise ValueError("Fixture is missing goals")
        return cls(
            match_date=fixture_match_date(fixture),
            home_team_id=int(fixture.home_team_id),
            away_team_id=int(fixture.away_team_id),
            goals_home=int(goals_home),
            goals_away=int(goals_away),
        )
