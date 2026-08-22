"""Fixture persistence and query helpers (API-Football shaped rows)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

from sqlalchemy import cast, Date, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from data_sources.api_football_leagues import (
    load_api_football_leagues,
)
from objects.models.fixture import FixtureModel
from objects.models.team import TeamModel
from objects.repositories.base import BaseRepository
from objects.schema.db.fixture import Fixture, FixtureCreate
from utils.seasons import season_code_to_start_year

logger = logging.getLogger(__name__)

# Recent seasons as calendar start years (was YYXX '2425','2526').
DEFAULT_LEAGUE_SEASONS = (2024, 2025)


def fixture_result(
    goals_home: int | None, goals_away: int | None
) -> str | None:
    """Derive 1/X/2 from goals; None if either side is missing."""
    if goals_home is None or goals_away is None:
        return None
    if goals_home > goals_away:
        return "1"
    if goals_home == goals_away:
        return "X"
    return "2"


class FixtureRepository(BaseRepository[FixtureModel]):
    model = FixtureModel

    def __init__(self, session: Session | None = None) -> None:
        super().__init__(session)
        from objects.repositories.league_repository import LeagueRepository
        from objects.repositories.team_repository import TeamRepository

        self.team_repo = TeamRepository(self.session)
        self.league_repo = LeagueRepository(self.session)

    def _team_external_ids_for_names(self, team_names: list[str]) -> list[int]:
        if not team_names:
            return []
        return list(
            self.session.scalars(
                select(TeamModel.external_id).where(TeamModel.name.in_(team_names))
            ).all()
        )

    def _league_api_id(self, league_id: int) -> int | None:
        """Resolve internal leagues.id → API external_id."""
        league = self.league_repo.get(league_id)
        if league is None:
            return None
        return league.external_id

    def _codes_to_league_api_ids(self, codes: list[str]) -> list[int]:
        league_map = load_api_football_leagues()
        ids: list[int] = []
        for code in codes:
            entry = league_map.get(code)
            if entry is not None:
                ids.append(entry.league_id)
                continue
            if str(code).isdigit():
                league = self.league_repo.get_by_external_id(int(code))
                if league is not None:
                    ids.append(league.external_id)
        return ids

    @staticmethod
    def _to_league_season(season: str | int) -> int:
        """Normalize start-year or YYXX season codes to league_season int."""
        if isinstance(season, int):
            return season
        text = str(season).strip()
        if len(text) == 4 and text.isdigit():
            value = int(text)
            if value >= 1900:
                return value
            # YYXX football code (e.g. 2425 → 2024)
            start = int(text[:2])
            end = int(text[2:])
            if end == (start + 1) % 100:
                return season_code_to_start_year(text)
            return value
        try:
            return season_code_to_start_year(text)
        except (KeyError, ValueError, IndexError):
            return int(text) if text.isdigit() else 0

    @staticmethod
    def _fixture_date_col():
        return cast(FixtureModel.fixture_date, Date)

    @staticmethod
    def _venue_filters_by_external_ids(
        team_external_ids: list[int], venue: Literal["home", "away"] | None
    ) -> list:
        if venue == "home":
            return [FixtureModel.home_team_id.in_(team_external_ids)]
        if venue == "away":
            return [FixtureModel.away_team_id.in_(team_external_ids)]
        return [
            or_(
                FixtureModel.home_team_id.in_(team_external_ids),
                FixtureModel.away_team_id.in_(team_external_ids),
            )
        ]

    def find_before_date_by_team(
        self,
        *,
        team_name: str,
        before_date: date,
        venue: Literal["home", "away"] | None = None,
        limit: int,
    ) -> list[FixtureModel]:
        """Newest-first fixtures for a team name before cutoff."""
        team_external_ids = self._team_external_ids_for_names([team_name])
        if not team_external_ids:
            team = self.team_repo.get_by_name(team_name)
            if team is None:
                # Fall back to denormalized fixture names
                query = (
                    select(self.model)
                    .where(
                        self._fixture_date_col() < before_date,
                        or_(
                            self.model.home_team_name == team_name,
                            self.model.away_team_name == team_name,
                        ),
                    )
                    .order_by(self.model.fixture_date.desc())
                    .limit(limit)
                )
                if venue == "home":
                    query = (
                        select(self.model)
                        .where(
                            self._fixture_date_col() < before_date,
                            self.model.home_team_name == team_name,
                        )
                        .order_by(self.model.fixture_date.desc())
                        .limit(limit)
                    )
                elif venue == "away":
                    query = (
                        select(self.model)
                        .where(
                            self._fixture_date_col() < before_date,
                            self.model.away_team_name == team_name,
                        )
                        .order_by(self.model.fixture_date.desc())
                        .limit(limit)
                    )
                return list(self.session.scalars(query).all())
            team_external_ids = [team.external_id]

        query = (
            select(self.model)
            .where(
                self._fixture_date_col() < before_date,
                *self._venue_filters_by_external_ids(team_external_ids, venue),
            )
            .order_by(self.model.fixture_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    def find_before_date_by_team_names(
        self,
        *,
        team_names: list[str],
        before_date: date,
        limit: int = 500,
    ) -> list[FixtureModel]:
        team_external_ids = self._team_external_ids_for_names(team_names)
        if not team_external_ids:
            return []
        query = (
            select(self.model)
            .where(
                self._fixture_date_col() < before_date,
                or_(
                    self.model.home_team_id.in_(team_external_ids),
                    self.model.away_team_id.in_(team_external_ids),
                ),
            )
            .order_by(self.model.fixture_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    def find_before_date_by_league_id(
        self,
        *,
        league_id: int,
        before_date: date,
        season: str | int | None = None,
        limit: int = 500,
    ) -> list[FixtureModel]:
        """Newest-first fixtures; league_id is internal leagues.id."""
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return []
        filters = [
            self._fixture_date_col() < before_date,
            self.model.league_id == league_api_id,
        ]
        if season is not None:
            filters.append(self.model.league_season == self._to_league_season(season))
        query = (
            select(self.model)
            .where(*filters)
            .order_by(self.model.fixture_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    def get_by_fixture_id(self, fixture_id: int) -> FixtureModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.fixture_id == fixture_id)
        )

    def find_team_matches_on_date(
        self, *, team_external_id: int, match_date: date
    ) -> list[FixtureModel]:
        query = select(self.model).where(
            self._fixture_date_col() == match_date,
            or_(
                self.model.home_team_id == team_external_id,
                self.model.away_team_id == team_external_id,
            ),
        )
        return list(self.session.scalars(query).all())

    def get_by_date_and_teams(
        self,
        match_date: date,
        home_team: str,
        away_team: str,
    ) -> Fixture | None:
        home_name = self.team_repo.to_football_data_name(home_team) or home_team
        away_name = self.team_repo.to_football_data_name(away_team) or away_team
        home_ids = self._team_external_ids_for_names([home_name, home_team])
        away_ids = self._team_external_ids_for_names([away_name, away_team])

        filters = [self._fixture_date_col() == match_date]
        if home_ids:
            filters.append(self.model.home_team_id.in_(home_ids))
        else:
            filters.append(
                or_(
                    self.model.home_team_name == home_name,
                    self.model.home_team_name == home_team,
                )
            )
        if away_ids:
            filters.append(self.model.away_team_id.in_(away_ids))
        else:
            filters.append(
                or_(
                    self.model.away_team_name == away_name,
                    self.model.away_team_name == away_team,
                )
            )

        stmt = (
            select(self.model)
            .where(*filters)
            .order_by(self.model.id.desc())
            .limit(1)
        )
        model = self.session.scalar(stmt)
        if model is None:
            return None
        return self.to_schema(model)

    def get_distinct_home_teams(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.home_team_name)
                .distinct()
                .order_by(self.model.home_team_name)
            ).all()
        )

    def get_distinct_away_teams(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.away_team_name)
                .distinct()
                .order_by(self.model.away_team_name)
            ).all()
        )

    def find_by_date_range_and_teams(
        self,
        *,
        date_from: date,
        date_to: date,
        home_names: list[str] | None = None,
        away_names: list[str] | None = None,
        home_team_ids: list[int] | None = None,
        away_team_ids: list[int] | None = None,
        league_external_id: int | None = None,
    ) -> list[FixtureModel]:
        """home_team_ids/away_team_ids are API external ids when provided."""
        resolved_home_ids = home_team_ids or self._team_external_ids_for_names(
            home_names or []
        )

        resolved_away_ids = away_team_ids or self._team_external_ids_for_names(
            away_names or []
        )
        if not resolved_home_ids or not resolved_away_ids:
            return []

        query = select(self.model).where(
            self._fixture_date_col() >= date_from,
            self._fixture_date_col() <= date_to,
            self.model.home_team_id.in_(resolved_home_ids),
            self.model.away_team_id.in_(resolved_away_ids),
        )
        if league_external_id is not None:
            query = query.where(self.model.league_id == league_external_id)
        sql = str(
            query.compile(
                dialect=self.session.get_bind().dialect,
                compile_kwargs={"literal_binds": True},
            )
        )
        return list(self.session.scalars(query).all())

    def find_by_season_and_teams(
        self,
        *,
        league_external_id: int,
        season: str | int,
        home_names: list[str] | None = None,
        away_names: list[str] | None = None,
        home_team_ids: list[int] | None = None,
        away_team_ids: list[int] | None = None,
    ) -> list[FixtureModel]:
        league_season = self._to_league_season(season)
        resolved_home_ids = home_team_ids or self._team_external_ids_for_names(
            home_names or []
        )
        resolved_away_ids = away_team_ids or self._team_external_ids_for_names(
            away_names or []
        )
        if not resolved_home_ids or not resolved_away_ids:
            return []

        query = select(self.model).where(
            self.model.league_season == league_season,
            self.model.home_team_id.in_(resolved_home_ids),
            self.model.away_team_id.in_(resolved_away_ids),
        )
        with_league = query.where(self.model.league_id == league_external_id)
        result = list(self.session.scalars(with_league).all())
        if result:
            return result
        return list(self.session.scalars(query).all())

    def get_filtered(
        self,
        *,
        leagues: list[str] | None = None,
        seasons: list[str | int] | None = None,
        before_date: date | None = None,
        limit: int | None = None,
    ) -> list[FixtureModel]:
        """Return fixtures; ``leagues`` are internal codes, ``seasons`` YYXX or years."""
        query = select(self.model)
        if leagues:
            api_ids = self._codes_to_league_api_ids(leagues)
            if api_ids:
                query = query.where(self.model.league_id.in_(api_ids))
            else:
                return []
        if seasons:
            season_years = [self._to_league_season(s) for s in seasons]
            query = query.where(self.model.league_season.in_(season_years))
        if before_date:
            query = query.where(self._fixture_date_col() < before_date)
        if limit is not None:
            query = query.order_by(self.model.fixture_date.desc()).limit(limit)
        else:
            query = query.order_by(self.model.fixture_date.asc())
        return list(self.session.scalars(query).all())

    def get_matches_by_team(self, team: TeamModel) -> list[Fixture]:
        query = (
            select(self.model)
            .where(
                or_(
                    self.model.home_team_id == team.external_id,
                    self.model.away_team_id == team.external_id,
                )
            )
            .order_by(self.model.fixture_date.asc())
        )
        return [self.to_schema(m) for m in self.session.scalars(query).all()]

    def get_num_teams_by_league(self, league_id: int) -> int:
        """Count distinct API team ids appearing in fixtures for a league PK."""
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return 0
        home_ids = select(self.model.home_team_id).where(
            self.model.league_id == league_api_id
        )
        away_ids = select(self.model.away_team_id).where(
            self.model.league_id == league_api_id
        )
        distinct = self.session.scalar(
            select(func.count()).select_from(
                select(self.model.home_team_id.label("tid"))
                .where(self.model.league_id == league_api_id)
                .union(
                    select(self.model.away_team_id).where(
                        self.model.league_id == league_api_id
                    )
                )
                .subquery()
            )
        )
        del home_ids, away_ids
        return int(distinct or 0)

    def get_home_goals_sum_by_league(
        self, league_id: int, season: str | int, before_date: date
    ) -> tuple[int, int]:
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return 0, 0
        league_season = self._to_league_season(season)
        row = self.session.execute(
            select(
                func.coalesce(func.sum(self.model.goals_home), 0),
                func.count(),
            ).where(
                self.model.league_id == league_api_id,
                self.model.league_season == league_season,
                self._fixture_date_col() < before_date,
            )
        ).one()
        return int(row[0] or 0), int(row[1] or 0)

    def get_away_goals_sum_by_league(
        self, league_id: int, season: str | int, before_date: date
    ) -> tuple[int, int]:
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return 0, 0
        league_season = self._to_league_season(season)
        row = self.session.execute(
            select(
                func.coalesce(func.sum(self.model.goals_away), 0),
                func.count(),
            ).where(
                self.model.league_id == league_api_id,
                self.model.league_season == league_season,
                self._fixture_date_col() < before_date,
            )
        ).one()
        return int(row[0] or 0), int(row[1] or 0)

    def get_goal_sums_by_league_before_date(
        self, before_date: date
    ) -> dict[int, tuple[int, int, int, int]]:
        """Aggregate home/away goals by API league id before cutoff."""

        rows = self.session.execute(
            select(
                self.model.league_id,
                func.coalesce(func.sum(self.model.goals_home), 0),
                func.count(),
                func.coalesce(func.sum(self.model.goals_away), 0),
                func.count(),
            )
            .where(self._fixture_date_col() < before_date)
            .group_by(self.model.league_id)
        ).all()
        result: dict[int, tuple[int, int, int, int]] = {}
        for league_api_id, sum_home, home_count, sum_away, away_count in rows:
            result[int(league_api_id)] = (
                int(sum_home),
                int(home_count),
                int(sum_away),
                int(away_count),
            )
        return result

    def get_home_goal_average_by_league_before_date(
        self, league_id: int, before_date: date
    ) -> float:
        num_teams = self.get_num_teams_by_league(league_id)
        if not num_teams:
            return 0.0
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return 0.0
        total = self.session.scalar(
            select(func.coalesce(func.sum(self.model.goals_home), 0)).where(
                self.model.league_id == league_api_id,
                self._fixture_date_col() <= before_date,
            )
        )
        return float(total or 0) / num_teams

    def get_away_goal_average_by_league(
        self, league_id: int, before_date: date | None = None
    ) -> float:
        num_teams = self.get_num_teams_by_league(league_id)
        if not num_teams:
            return 0.0
        league_api_id = self._league_api_id(league_id)
        if league_api_id is None:
            return 0.0
        filters = [self.model.league_id == league_api_id]
        if before_date is not None:
            filters.append(self._fixture_date_col() <= before_date)
        else:
            filters.append(self.model.league_season.in_(DEFAULT_LEAGUE_SEASONS))
        total = self.session.scalar(
            select(func.coalesce(func.sum(self.model.goals_away), 0)).where(*filters)
        )
        return float(total or 0) / num_teams

    def resolve_internal_league_id_for_team(
        self, team: TeamModel
    ) -> int | None:
        """Map a team to internal leagues.id via its most recent fixture."""
        league_api_id = self.session.scalar(
            select(self.model.league_id)
            .where(
                or_(
                    self.model.home_team_id == team.external_id,
                    self.model.away_team_id == team.external_id,
                )
            )
            .order_by(self.model.fixture_date.desc())
            .limit(1)
        )
        if league_api_id is None:
            return None
        league = self.league_repo.get_by_external_id(int(league_api_id))
        return league.id if league is not None else None

    def upsert_many(self, fixtures: list[FixtureCreate]) -> int:
        written = 0
        for fixture in fixtures:
            values = fixture.model_dump()
            stmt = pg_insert(self.model).values(**values)
            update_cols = {
                key: getattr(stmt.excluded, key)
                for key in values
                if key != "fixture_id"
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["fixture_id"],
                set_=update_cols,
            )
            self.session.execute(stmt)
            written += 1
        self.session.commit()
        return written

    def to_schema(self, model: FixtureModel) -> Fixture:
        return Fixture.model_validate(model)


# Temporary aliases while call sites migrate off HistoricalMatch* names.
HistoricalMatchRepository = FixtureRepository
HistoricalMatchModel = FixtureModel
HistoricalMatch = Fixture
HistoricalMatchCreate = FixtureCreate
