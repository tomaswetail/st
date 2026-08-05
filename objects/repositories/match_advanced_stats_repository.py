"""Persistence for match advanced statistics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.match_advanced_stats import MatchAdvancedStatsModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe


class MatchAdvancedStatsRepository(BaseRepository[MatchAdvancedStatsModel]):
    model = MatchAdvancedStatsModel

    def get_by_match_and_provider(
        self,
        match_id: int,
        provider: str,
    ) -> MatchAdvancedStatsModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.match_id == match_id,
                self.model.provider == provider,
            )
        )

    def list_for_matches(
        self,
        match_ids: list[int],
        provider: str | None = None,
    ) -> list[MatchAdvancedStatsModel]:
        if not match_ids:
            return []
        query = select(self.model).where(self.model.match_id.in_(match_ids))
        if provider:
            query = query.where(self.model.provider == provider)
        return list(self.session.scalars(query).all())

    def upsert(
        self,
        *,
        match_id: int,
        provider: str,
        fields: dict[str, Any],
    ) -> MatchAdvancedStatsModel:
        payload = {
            "match_id": match_id,
            "provider": provider,
            **fields,
        }
        if "raw_payload" in payload and payload["raw_payload"] is not None:
            payload["raw_payload"] = json_safe(payload["raw_payload"])
        if "fetched_at" not in payload or payload["fetched_at"] is None:
            payload["fetched_at"] = datetime.now().astimezone()

        update_fields = {
            key: value
            for key, value in payload.items()
            if key not in ("match_id", "provider")
        }
        statement = (
            pg_insert(self.model)
            .values(**payload)
            .on_conflict_do_update(
                constraint="uq_match_advanced_stats_provider",
                set_=update_fields,
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()
