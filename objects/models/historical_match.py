from datetime import date

from sqlalchemy import UniqueConstraint, Index, Text, Date, DOUBLE_PRECISION, String
from sqlalchemy.dialects.postgresql import JSONB

from database import Base
from typing import Any

from sqlalchemy import (
    Integer,
)
from sqlalchemy.orm import Mapped, mapped_column

class HistoricalMatchModel(Base):
    __tablename__ = "historical_matches"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "league",
            "season",
            "match_date",
            "home_team",
            "away_team",
        ),
        Index("idx_historical_league_season", "league", "season", "match_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    league: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(String(50), nullable=False)
    odds_home: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    odds_draw: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    odds_away: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
