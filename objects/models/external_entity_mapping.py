"""Generic mapping between internal entities and external provider IDs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class ExternalEntityMappingModel(Base):
    __tablename__ = "external_entity_mapping"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "entity_type",
            "external_entity_id",
            name="uq_external_entity_by_external_id",
        ),
        UniqueConstraint(
            "provider",
            "entity_type",
            "internal_entity_id",
            name="uq_external_entity_by_internal_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    internal_entity_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    external_entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    external_name: Mapped[str | None] = mapped_column(String(200))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
