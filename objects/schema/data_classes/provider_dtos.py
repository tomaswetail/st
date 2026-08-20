"""Normalized provider DTOs independent of SofaScore / API-Football response shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ProviderLeague:
    provider_league_id: str
    name: str
    country: str | None = None
    country_code: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeagueMappingSuggestion:
    internal_league_id: int
    internal_name: str
    candidate: ProviderLeague | None
    confidence: float
    method: str  # exact | alias | fuzzy | unresolved


@dataclass
class LeagueMappingResult:
    internal_league_id: int
    provider: str
    external_entity_id: str | None
    status: Literal["mapped", "already_mapped", "unresolved", "failed"]
    candidates: list[ProviderLeague] = field(default_factory=list)
    error: str | None = None


@dataclass
class ProviderSeason:
    provider_season_id: str
    name: str
    start_year: int | None = None
    end_year: int | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderTeam:
    provider_team_id: str
    name: str
    short_name: str | None = None
    country_name: str | None = None
    country_code: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMatch:
    provider_match_id: str
    provider_league_id: str
    provider_season_id: str | None
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    kickoff_at: datetime
    status: str
    home_score: int | None = None
    away_score: int | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderShot:
    provider_shot_id: str | None
    team_id: str
    player_id: str | None
    minute: int | None
    second: int | None
    xg: float | None
    xgot: float | None
    outcome: str | None
    situation: str | None
    body_part: str | None
    shot_type: str | None
    is_penalty: bool
    is_own_goal: bool
    coordinates: dict[str, Any] | None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMatchDetails:
    match: ProviderMatch
    shots: list[ProviderShot]
    statistics: dict[str, Any]
    lineups: dict[str, Any] | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    # Provider-reported aggregate xG when present (separate from shot-derived).
    home_xg: float | None = None
    away_xg: float | None = None
    home_xgot: float | None = None
    away_xgot: float | None = None
