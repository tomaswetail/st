"""Individual shot events for a historical match from a data provider."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Boolean,
    Double,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class MatchShotModel(Base):
    __tablename__ = "match_shots"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "provider",
            "shot_fingerprint",
            name="uq_match_shot_fingerprint",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    provider_shot_id: Mapped[str | None] = mapped_column(String(100))
    shot_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    player_external_id: Mapped[str | None] = mapped_column(String(100))
    minute: Mapped[int | None] = mapped_column(Integer)
    second: Mapped[int | None] = mapped_column(Integer)
    xg: Mapped[float | None] = mapped_column(Double)
    xgot: Mapped[float | None] = mapped_column(Double)
    outcome: Mapped[str | None] = mapped_column(String(50))
    situation: Mapped[str | None] = mapped_column(String(50))
    body_part: Mapped[str | None] = mapped_column(String(50))
    shot_type: Mapped[str | None] = mapped_column(String(50))
    is_penalty: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_own_goal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    coordinates: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
