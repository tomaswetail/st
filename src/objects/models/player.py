"""Player registry for injury/availability snapshots."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Double, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PlayerModel(Base):
    __tablename__ = "players"
    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_players_provider_external_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    team_external_id: Mapped[str | None] = mapped_column(String(100))
    market_value: Mapped[float | None] = mapped_column(Double)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
