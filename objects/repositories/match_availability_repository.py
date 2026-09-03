"""Match-level availability snapshot persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.match_availability import MatchAvailabilityModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe


class MatchAvailabilityRepository(BaseRepository[MatchAvailabilityModel]):
    model = MatchAvailabilityModel

    def get_for_fixture(
        self,
        fixture_id: int,
        *,
        provider: str,
        source: str | None = None,
    ) -> MatchAvailabilityModel | None:
        query = select(self.model).where(
            self.model.fixture_id == fixture_id,
            self.model.provider == provider,
        )
        if source is not None:
            query = query.where(self.model.source == source)
        return self.session.scalar(query)

    def list_before_cutoff(
        self,
        fixture_id: int,
        *,
        cutoff: datetime,
        provider: str | None = None,
    ) -> list[MatchAvailabilityModel]:
        query = select(self.model).where(
            self.model.fixture_id == fixture_id,
            self.model.snapshot_at <= cutoff,
        )
        if provider:
            query = query.where(self.model.provider == provider)
        query = query.order_by(self.model.snapshot_at.desc())
        return list(self.session.scalars(query).all())

    def upsert(
        self,
        *,
        fixture_id: int,
        provider: str,
        source: str,
        snapshot_at: datetime,
        kickoff_at: datetime,
        home_team_external_id: str | None,
        away_team_external_id: str | None,
        home_starter_count: int,
        away_starter_count: int,
        home_unavailable_count: int,
        away_unavailable_count: int,
        home_missing_value: float | None,
        away_missing_value: float | None,
        coverage_level: str,
        players_json: list[dict[str, Any]] | None,
        raw_payload: dict[str, Any] | None = None,
    ) -> MatchAvailabilityModel:
        payload = {
            "fixture_id": fixture_id,
            "provider": provider,
            "source": source,
            "snapshot_at": snapshot_at,
            "kickoff_at": kickoff_at,
            "home_team_external_id": home_team_external_id,
            "away_team_external_id": away_team_external_id,
            "home_starter_count": home_starter_count,
            "away_starter_count": away_starter_count,
            "home_unavailable_count": home_unavailable_count,
            "away_unavailable_count": away_unavailable_count,
            "home_missing_value": home_missing_value,
            "away_missing_value": away_missing_value,
            "coverage_level": coverage_level,
            "players_json": json_safe(players_json) if players_json is not None else None,
            "raw_payload": json_safe(raw_payload) if raw_payload is not None else None,
        }
        statement = (
            pg_insert(self.model)
            .values(**payload)
            .on_conflict_do_update(
                constraint="uq_match_availability_fixture_provider_source",
                set_={
                    key: value
                    for key, value in payload.items()
                    if key not in ("fixture_id", "provider", "source")
                },
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()

    def fixture_ids_with_availability(
        self,
        fixture_ids: list[int],
        *,
        provider: str | None = None,
    ) -> set[int]:
        if not fixture_ids:
            return set()
        query = select(self.model.fixture_id).where(
            self.model.fixture_id.in_(fixture_ids)
        )
        if provider:
            query = query.where(self.model.provider == provider)
        return set(self.session.scalars(query).all())
