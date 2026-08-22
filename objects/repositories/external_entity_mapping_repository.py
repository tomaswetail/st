"""Persistence for external provider entity mappings."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe


class ExternalEntityMappingRepository(BaseRepository[ExternalEntityMappingModel]):
    model = ExternalEntityMappingModel

    def get_by_external(
        self,
        *,
        provider: str,
        entity_type: str,
        external_entity_id: str,
    ) -> ExternalEntityMappingModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.provider == provider,
                self.model.entity_type == entity_type,
                self.model.external_entity_id == external_entity_id,
            )
        )

    def get_by_internal(
        self,
        *,
        provider: str,
        entity_type: str,
        internal_entity_id: int,
    ) -> ExternalEntityMappingModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.provider == provider,
                self.model.entity_type == entity_type,
                self.model.internal_entity_id == internal_entity_id,
            )
        )

    def upsert(
        self,
        *,
        provider: str,
        entity_type: str,
        internal_entity_id: int,
        external_entity_id: str,
        external_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExternalEntityMappingModel:
        # Column is named "metadata" in DB; avoid ORM attribute name "metadata"
        # which collides with SQLAlchemy MetaData.
        meta_column = self.model.__table__.c.metadata
        safe_metadata = json_safe(metadata) if metadata is not None else None
        external_id = str(external_entity_id)

        with self.session.no_autoflush:
            existing_by_external = self.get_by_external(
                provider=provider,
                entity_type=entity_type,
                external_entity_id=external_id,
            )
            existing_by_internal = self.get_by_internal(
                provider=provider,
                entity_type=entity_type,
                internal_entity_id=internal_entity_id,
            )

        if existing_by_external is not None:
            if existing_by_external.internal_entity_id == internal_entity_id:
                existing_by_external.external_name = external_name
                existing_by_external.metadata_json = safe_metadata
                return existing_by_external
            # External id already maps elsewhere. Never move it onto an internal
            # team that already has its own mapping (uq_external_entity_by_internal_id).
            if existing_by_internal is not None:
                return existing_by_internal
            existing_by_external.internal_entity_id = internal_entity_id
            existing_by_external.external_name = external_name
            existing_by_external.metadata_json = safe_metadata
            return existing_by_external

        # One internal entity may only have one external id per provider.
        # Alternate spellings that resolve to the same team keep the first mapping.
        if existing_by_internal is not None:
            return existing_by_internal

        values = {
            "provider": provider,
            "entity_type": entity_type,
            "internal_entity_id": internal_entity_id,
            "external_entity_id": external_id,
            "external_name": external_name,
            meta_column: safe_metadata,
        }
        statement = (
            pg_insert(self.model)
            .values(values)
            .on_conflict_do_update(
                constraint="uq_external_entity_by_external_id",
                set_={
                    "internal_entity_id": internal_entity_id,
                    "external_name": external_name,
                    meta_column: safe_metadata,
                },
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()
