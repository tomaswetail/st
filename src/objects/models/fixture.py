from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class FixtureModel(Base):
    __tablename__ = "fixtures"

    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            name="uq_fixtures_fixture_id",
        ),
        Index(
            "idx_fixtures_league_season_date",
            "league_id",
            "league_season",
            "fixture_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # fixture
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fixture_referee: Mapped[str | None] = mapped_column(String(255))
    fixture_timezone: Mapped[str] = mapped_column(String(50), nullable=False)
    fixture_date: Mapped[datetime] = mapped_column(nullable=False)
    fixture_timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)

    period_first: Mapped[int | None] = mapped_column(BigInteger)
    period_second: Mapped[int | None] = mapped_column(BigInteger)

    venue_id: Mapped[int | None] = mapped_column(Integer)
    venue_name: Mapped[str | None] = mapped_column(String(255))
    venue_city: Mapped[str | None] = mapped_column(String(255))

    status_long: Mapped[str] = mapped_column(String(50), nullable=False)
    status_short: Mapped[str] = mapped_column(String(10), nullable=False)

    # league
    league_id: Mapped[int] = mapped_column(Integer, nullable=False)
    league_name: Mapped[str] = mapped_column(String(255), nullable=False)
    league_country: Mapped[str | None] = mapped_column(String(100))
    league_flag: Mapped[str | None] = mapped_column(String(500))
    league_season: Mapped[int] = mapped_column(Integer, nullable=False)
    league_round: Mapped[str | None] = mapped_column(String(100))

    # teams.home
    home_team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    home_team_winner: Mapped[bool | None] = mapped_column(Boolean)

    # teams.away
    away_team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    away_team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    away_team_winner: Mapped[bool | None] = mapped_column(Boolean)

    # goals
    goals_home: Mapped[int | None] = mapped_column(Integer)
    goals_away: Mapped[int | None] = mapped_column(Integer)

    # score
    score_halftime_home: Mapped[int | None] = mapped_column(Integer)
    score_halftime_away: Mapped[int | None] = mapped_column(Integer)
    score_fulltime_home: Mapped[int | None] = mapped_column(Integer)
    score_fulltime_away: Mapped[int | None] = mapped_column(Integer)
    score_extratime_home: Mapped[int | None] = mapped_column(Integer)
    score_extratime_away: Mapped[int | None] = mapped_column(Integer)
    score_penalty_home: Mapped[int | None] = mapped_column(Integer)
    score_penalty_away: Mapped[int | None] = mapped_column(Integer)


# Temporary alias while repositories migrate off HistoricalMatchModel.
HistoricalMatchModel = FixtureModel
