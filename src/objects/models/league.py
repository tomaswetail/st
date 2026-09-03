from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class LeagueModel(Base):
    __tablename__ = "leagues"

    __table_args__ = (
        UniqueConstraint(
            "external_id",
            name="uq_leagues_external_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(Integer, nullable=False)
    league_name: Mapped[str] = mapped_column(String(100), nullable=False)
    league_type: Mapped[str] = mapped_column(String(50), nullable=False)
    country_name: Mapped[str] = mapped_column(String(100), nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(10))
