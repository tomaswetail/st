"""Per-provider advanced match statistics attached to historical_matches."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class MatchAdvancedStatsModel(Base):
    __tablename__ = "match_advanced_stats"
    __table_args__ = (
        UniqueConstraint("match_id", "provider", name="uq_match_advanced_stats_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("historical_matches.id"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)

    home_xg: Mapped[float | None] = mapped_column(Double)
    away_xg: Mapped[float | None] = mapped_column(Double)
    home_non_penalty_xg: Mapped[float | None] = mapped_column(Double)
    away_non_penalty_xg: Mapped[float | None] = mapped_column(Double)
    home_xgot: Mapped[float | None] = mapped_column(Double)
    away_xgot: Mapped[float | None] = mapped_column(Double)
    home_shots: Mapped[int | None] = mapped_column(Integer)
    away_shots: Mapped[int | None] = mapped_column(Integer)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    home_set_piece_xg: Mapped[float | None] = mapped_column(Double)
    away_set_piece_xg: Mapped[float | None] = mapped_column(Double)
    home_open_play_xg: Mapped[float | None] = mapped_column(Double)
    away_open_play_xg: Mapped[float | None] = mapped_column(Double)

    # Shot-derived aggregates kept separate from provider-reported xG.
    home_xg_from_shots: Mapped[float | None] = mapped_column(Double)
    away_xg_from_shots: Mapped[float | None] = mapped_column(Double)
    average_home_shot_xg: Mapped[float | None] = mapped_column(Double)
    average_away_shot_xg: Mapped[float | None] = mapped_column(Double)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload_hash: Mapped[str | None] = mapped_column(String(64))
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
