"""Aggregate match-level availability snapshot."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Double, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MatchAvailabilityModel(Base):
    __tablename__ = "match_availability"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "provider",
            "source",
            name="uq_match_availability_fixture_provider_source",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team_external_id: Mapped[str | None] = mapped_column(String(100))
    away_team_external_id: Mapped[str | None] = mapped_column(String(100))
    home_starter_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_starter_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    home_unavailable_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_unavailable_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    home_missing_value: Mapped[float | None] = mapped_column(Double)
    away_missing_value: Mapped[float | None] = mapped_column(Double)
    coverage_level: Mapped[str] = mapped_column(String(30), nullable=False, default="none")
    players_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
