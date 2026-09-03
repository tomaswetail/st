"""Injury snapshot persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.injury_snapshot import InjurySnapshotModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe


class InjurySnapshotRepository(BaseRepository[InjurySnapshotModel]):
    model = InjurySnapshotModel

    def list_for_fixture(
        self,
        fixture_id: int,
        *,
        provider: str | None = None,
        before: datetime | None = None,
    ) -> list[InjurySnapshotModel]:
        query = select(self.model).where(self.model.fixture_id == fixture_id)
        if provider:
            query = query.where(self.model.provider == provider)
        if before is not None:
            query = query.where(self.model.snapshot_at <= before)
        query = query.order_by(self.model.snapshot_at.desc())
        return list(self.session.scalars(query).all())

    def upsert(
        self,
        *,
        fixture_id: int,
        player_id: int,
        provider: str,
        team_side: str,
        status: str,
        is_starter: bool,
        market_value: float | None,
        snapshot_at: datetime,
        source: str,
        raw_payload: dict[str, Any] | None = None,
    ) -> InjurySnapshotModel:
        payload = {
            "fixture_id": fixture_id,
            "player_id": player_id,
            "provider": provider,
            "team_side": team_side,
            "status": status,
            "is_starter": is_starter,
            "market_value": market_value,
            "snapshot_at": snapshot_at,
            "source": source,
            "raw_payload": json_safe(raw_payload) if raw_payload is not None else None,
        }
        statement = (
            pg_insert(self.model)
            .values(**payload)
            .on_conflict_do_update(
                constraint="uq_injury_snapshots_fixture_player_snapshot",
                set_={
                    key: value
                    for key, value in payload.items()
                    if key
                    not in ("fixture_id", "player_id", "provider", "snapshot_at")
                },
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()

    def delete_for_fixture_provider(
        self,
        fixture_id: int,
        provider: str,
    ) -> int:
        rows = list(
            self.session.scalars(
                select(self.model).where(
                    and_(
                        self.model.fixture_id == fixture_id,
                        self.model.provider == provider,
                    )
                )
            ).all()
        )
        for row in rows:
            self.session.delete(row)
        return len(rows)
