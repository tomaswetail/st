from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    String,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

class STMatchModel(Base):
    __tablename__ = "stryktipset_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    stryktipset_round_id: Mapped[int] = mapped_column(
        ForeignKey("stryktipset_rounds.id"),
        nullable=False,
    )

    event: Mapped["STRoundModel"] = relationship(
        back_populates="matches",
    )

    # External API matchId, e.g. 1756264
    external_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)

    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    status: Mapped[Optional[str]] = mapped_column(String(50))
    status_id: Mapped[Optional[int]] = mapped_column(Integer)

    league_name: Mapped[Optional[str]] = mapped_column(String(150))
    league_country_name: Mapped[Optional[str]] = mapped_column(String(100))

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    home_team: Mapped["TeamModel"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["TeamModel"] = relationship(foreign_keys=[away_team_id])

    # Final result
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    stryktipset_result: Mapped[Optional[str]] = mapped_column(String(1))

    # Optional halftime result
    halftime_home_score: Mapped[Optional[int]] = mapped_column(Integer)
    halftime_away_score: Mapped[Optional[int]] = mapped_column(Integer)

    match_bet: Mapped[Optional["STMatchBetModel"]] = relationship(
        back_populates="stryktipset_match",
        uselist=False,
        cascade="all, delete-orphan",
    )
    match_odds: Mapped[Optional["STMatchOddsModel"]] = relationship(
        back_populates="stryktipset_match",
        uselist=False,
        cascade="all, delete-orphan",
    )
