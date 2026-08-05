"""Persistence for individual match shots."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.match_shot import MatchShotModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe


class MatchShotRepository(BaseRepository[MatchShotModel]):
    model = MatchShotModel

    def list_for_match(
        self,
        match_id: int,
        provider: str | None = None,
    ) -> list[MatchShotModel]:
        query = select(self.model).where(self.model.match_id == match_id)
        if provider:
            query = query.where(self.model.provider == provider)
        return list(self.session.scalars(query).all())

    def list_for_matches(
        self,
        match_ids: list[int],
        provider: str | None = None,
    ) -> list[MatchShotModel]:
        if not match_ids:
            return []
        query = select(self.model).where(self.model.match_id.in_(match_ids))
        if provider:
            query = query.where(self.model.provider == provider)
        return list(self.session.scalars(query).all())

    def upsert_many(
        self,
        *,
        match_id: int,
        provider: str,
        shots: list[dict[str, Any]],
    ) -> int:
        """Upsert shots by fingerprint. Returns number of rows written."""
        if not shots:
            return 0
        written = 0
        for shot in shots:
            payload = {
                "match_id": match_id,
                "provider": provider,
                **shot,
            }
            if payload.get("coordinates") is not None:
                payload["coordinates"] = json_safe(payload["coordinates"])
            if payload.get("raw_payload") is not None:
                payload["raw_payload"] = json_safe(payload["raw_payload"])
            update_fields = {
                key: value
                for key, value in payload.items()
                if key not in ("match_id", "provider", "shot_fingerprint")
            }
            statement = (
                pg_insert(self.model)
                .values(**payload)
                .on_conflict_do_update(
                    constraint="uq_match_shot_fingerprint",
                    set_=update_fields,
                )
            )
            self.session.execute(statement)
            written += 1
        return written
