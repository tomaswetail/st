from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class TeamModel(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # External API team id, e.g. 1000512
    external_id: Mapped[Optional[int]] = mapped_column(
        Integer, unique=True, nullable=True
    )
    league_id: Mapped[int] = mapped_column(Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    machine_name: Mapped[str] = mapped_column(String(150), nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(String(50))
    medium_name: Mapped[Optional[str]] = mapped_column(String(100))

    country_name: Mapped[Optional[str]] = mapped_column(String(100))
    iso_code: Mapped[Optional[str]] = mapped_column(String(10))
