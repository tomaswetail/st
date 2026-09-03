from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    ForeignKey, Double
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

class STMatchOddsModel(Base):
    __tablename__ = "stryktipset_match_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    stryktipset_match_id: Mapped[int] = mapped_column(
        ForeignKey("stryktipset_matches.id"),
        nullable=False,
        unique=True,
    )

    stryktipset_match: Mapped["STMatchModel"] = relationship(
        back_populates="match_odds",
    )

    odds_1: Mapped[int] = mapped_column(Double, nullable=False)
    odds_X: Mapped[int] = mapped_column(Double, nullable=False)
    odds_2: Mapped[int] = mapped_column(Double, nullable=False)

