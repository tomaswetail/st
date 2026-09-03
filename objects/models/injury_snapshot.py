"""Per-player injury/availability rows at a fixture snapshot."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class InjurySnapshotModel(Base):
    __tablename__ = "injury_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "provider",
            "player_id",
            "snapshot_at",
            name="uq_injury_snapshots_fixture_player_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id"),
        nullable=False,
        index=True,
    )
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    team_side: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    is_starter: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    market_value: Mapped[float | None] = mapped_column(Double)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
