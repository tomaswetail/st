from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    String,  ForeignKey, DateTime,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

class STMatchBetModel(Base):
    __tablename__ = "stryktipset_match_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    stryktipset_match_id: Mapped[int] = mapped_column(
        ForeignKey("stryktipset_matches.id"),
        nullable=False,
        unique=True,
    )

    stryktipset_match: Mapped["STMatchModel"] = relationship(
        back_populates="match_bet",
    )

    distribution_1: Mapped[int] = mapped_column(Integer, nullable=False)
    distribution_X: Mapped[int] = mapped_column(Integer, nullable=False)
    distribution_2: Mapped[int] = mapped_column(Integer, nullable=False)

