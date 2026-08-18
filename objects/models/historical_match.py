from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Date,
    DOUBLE_PRECISION,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

if TYPE_CHECKING:
    from objects.models.team import TeamModel


# Wipe/reimport required after this schema change (no Alembic):
#   DROP TABLE historical_matches CASCADE;
# then init_db() / create_all and re-run DataCollector.refresh_all_data(...).
class HistoricalMatchModel(Base):
    __tablename__ = "historical_matches"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "league",
            "season",
            "match_date",
            "home_team_id",
            "away_team_id",
        ),
        Index("idx_historical_league_season", "league", "season", "match_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    league: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    home_team: Mapped["TeamModel"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["TeamModel"] = relationship(foreign_keys=[away_team_id])
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(String(50), nullable=False)
    odds_home: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    odds_draw: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    odds_away: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
