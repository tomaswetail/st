"""Historical bookmaker 1X2 odds attached to a fixture."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Double, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class FixtureOddsModel(Base):
    __tablename__ = "fixture_odds"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "provider",
            "bookmaker",
            "price_type",
            name="uq_fixture_odds_fixture_provider_bookmaker_price",
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
    bookmaker: Mapped[str] = mapped_column(String(30), nullable=False)
    price_type: Mapped[str] = mapped_column(String(20), nullable=False)
    odds_home: Mapped[float | None] = mapped_column(Double)
    odds_draw: Mapped[float | None] = mapped_column(Double)
    odds_away: Mapped[float | None] = mapped_column(Double)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
