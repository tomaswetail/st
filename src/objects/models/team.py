from __future__ import annotations

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class TeamModel(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    code: Mapped[str | None] = mapped_column(String(20))
    country: Mapped[str | None] = mapped_column(String(100))
    national: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
