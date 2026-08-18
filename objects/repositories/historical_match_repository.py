from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Literal

from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, aliased

from objects.models.historical_match import HistoricalMatchModel
from objects.models.team import TeamModel
from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe
from objects.schema.db.historical_match import HistoricalMatch, HistoricalMatchCreate
from utils.common import LEAGUE_NAMES_REV, get_season, odds_to_probabilities

logger = logging.getLogger(__name__)

SEASONS = "('2425','2526')"


class HistoricalMatchRepository(BaseRepository[HistoricalMatchModel]):
    model = HistoricalMatchModel

    def __init__(self, session: Session | None = None) -> None:
        super().__init__(session)
        from objects.repositories.league_repository import LeagueRepository
        from objects.repositories.team_repository import TeamRepository

        self.team_repo = TeamRepository(self.session)
        self.league_repo = LeagueRepository(self.session)

    def _team_ids_for_names(self, team_names: list[str]) -> list[int]:
        if not team_names:
            return []
        ids = list(
            self.session.scalars(
                select(TeamModel.id).where(TeamModel.name.in_(team_names))
            ).all()
        )
        return ids

    def find_before_date_by_team(
        self,
        *,
        team_name: str,
        before_date: date,
        venue: Literal["home", "away"] | None = None,
        limit: int,
    ) -> list[HistoricalMatchModel]:
        """Newest-first matches for a team name before cutoff, optionally venue-filtered."""
        team_ids = self._team_ids_for_names([team_name])
        if not team_ids:
            team = self.team_repo.get_by_name(team_name)
            if team is None:
                return []
            team_ids = [team.id]
        query = (
            select(self.model)
            .where(
                self.model.match_date < before_date,
                *self._venue_filters_by_ids(team_ids, venue),
            )
            .order_by(self.model.match_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    def find_before_date_by_team_names(
        self,
        *,
        team_names: list[str],
        before_date: date,
        limit: int = 500,
    ) -> list[HistoricalMatchModel]:
        """Newest-first matches involving any of the given team names before cutoff."""
        team_ids = self._team_ids_for_names(team_names)
        if not team_ids:
            return []
        query = (
            select(self.model)
            .where(
                self.model.match_date < before_date,
                or_(
                    self.model.home_team_id.in_(team_ids),
                    self.model.away_team_id.in_(team_ids),
                ),
            )
            .order_by(self.model.match_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    def find_before_date_by_league_id(
        self,
        *,
        league_id: int,
        before_date: date,
        season: str | None = None,
        limit: int = 500,
    ) -> list[HistoricalMatchModel]:
        """Newest-first league matches with match_date < before_date, then limit."""
        league = self.league_repo.get(league_id)
        if league is None:
            return []
        league_code = LEAGUE_NAMES_REV.get(league.name, league.name)
        filters = [
            self.model.match_date < before_date,
            self.model.league == league_code,
        ]
        if season is not None:
            filters.append(self.model.season == self._to_season_code(season))
        query = (
            select(self.model)
            .where(*filters)
            .order_by(self.model.match_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).all())

    @staticmethod
    def _to_season_code(season: str) -> str:
        """Normalize start-year or football-data season codes to DB season codes."""
        text_value = str(season).strip()
        if len(text_value) == 4 and text_value.isdigit():
            start = int(text_value[:2])
            end = int(text_value[2:])
            if end == (start + 1) % 100:
                return text_value
            if int(text_value) >= 1900:
                try:
                    return get_season(text_value)
                except KeyError:
                    return text_value
        try:
            return get_season(text_value)
        except KeyError:
            return text_value

    @staticmethod
    def _venue_filters_by_ids(
        team_ids: list[int], venue: Literal["home", "away"] | None
    ) -> list:
        """SQLAlchemy filters for home-only, away-only, or either venue."""
        if venue == "home":
            return [HistoricalMatchModel.home_team_id.in_(team_ids)]
        if venue == "away":
            return [HistoricalMatchModel.away_team_id.in_(team_ids)]
        return [
            or_(
                HistoricalMatchModel.home_team_id.in_(team_ids),
                HistoricalMatchModel.away_team_id.in_(team_ids),
            )
        ]

    def get_by_match_key(
        self,
        *,
        source: str,
        league: str,
        season: str,
        match_date: date,
        home_team_id: int,
        away_team_id: int,
    ) -> HistoricalMatchModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.source == source,
                self.model.league == league,
                self.model.season == season,
                self.model.match_date == match_date,
                self.model.home_team_id == home_team_id,
                self.model.away_team_id == away_team_id,
            )
        )

    def find_team_matches_on_date(
        self, *, team_id: int, match_date: date
    ) -> list[HistoricalMatchModel]:
        query = select(self.model).where(
            self.model.match_date == match_date,
            or_(
                self.model.home_team_id == team_id,
                self.model.away_team_id == team_id,
            ),
        )
        return list(self.session.scalars(query).all())

    @staticmethod
    def _same_unique_key(
        existing: HistoricalMatchModel, match: HistoricalMatchCreate
    ) -> bool:
        return (
            existing.source == match.source
            and existing.league == match.league
            and existing.season == match.season
            and existing.match_date == match.match_date
            and existing.home_team_id == match.home_team_id
            and existing.away_team_id == match.away_team_id
        )

    @staticmethod
    def _same_fixture(
        existing: HistoricalMatchModel,
        match: HistoricalMatchCreate,
        team_id: int,
    ) -> bool:
        return (
            existing.match_date == match.match_date
            and team_id in (existing.home_team_id, existing.away_team_id)
            and team_id in (match.home_team_id, match.away_team_id)
        )

    def team_has_other_match_on_date(
        self, match: HistoricalMatchCreate
    ) -> HistoricalMatchModel | None:
        for team_id in (match.home_team_id, match.away_team_id):
            for existing in self.find_team_matches_on_date(
                team_id=team_id, match_date=match.match_date
            ):
                if self._same_unique_key(existing, match):
                    continue
                if self._same_fixture(existing, match, team_id):
                    return existing
        return None

    def get_by_date_and_teams(
        self,
        match_date: date,
        home_team: str,
        away_team: str,
    ) -> HistoricalMatch | None:
        home_name = self.team_repo.to_football_data_name(home_team) or home_team
        away_name = self.team_repo.to_football_data_name(away_team) or away_team
        home_ids = self._team_ids_for_names([home_name, home_team])
        away_ids = self._team_ids_for_names([away_name, away_team])
        if not home_ids or not away_ids:
            return None

        stmt = (
            select(self.model)
            .where(
                self.model.match_date == match_date,
                self.model.home_team_id.in_(home_ids),
                self.model.away_team_id.in_(away_ids),
            )
            .order_by(self.model.id.desc())
            .limit(1)
        )
        model = self.session.scalar(stmt)
        if model is None:
            return None
        return self.to_schema(model)

    def get_distinct_home_teams(self) -> list[str]:
        home = aliased(TeamModel)
        return list(
            self.session.scalars(
                select(home.name)
                .select_from(self.model)
                .join(home, home.id == self.model.home_team_id)
                .distinct()
                .order_by(home.name)
            ).all()
        )

    def get_distinct_away_teams(self) -> list[str]:
        away = aliased(TeamModel)
        return list(
            self.session.scalars(
                select(away.name)
                .select_from(self.model)
                .join(away, away.id == self.model.away_team_id)
                .distinct()
                .order_by(away.name)
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
        league_code: str | None = None,
    ) -> list[HistoricalMatchModel]:
        resolved_home_ids = home_team_ids or self._team_ids_for_names(home_names or [])
        resolved_away_ids = away_team_ids or self._team_ids_for_names(away_names or [])
        if not resolved_home_ids or not resolved_away_ids:
            return []

        if home_names:
            l=1
        query = select(self.model).where(
            self.model.match_date >= date_from,
            self.model.match_date <= date_to,
            self.model.home_team_id.in_(resolved_home_ids),
            self.model.away_team_id.in_(resolved_away_ids),
        )
        if league_code:
            query = query.where(self.model.league == league_code)
        return list(self.session.scalars(query).all())

    def find_by_season_and_teams(
        self,
        *,
        league_code: str,
        season: str,
        home_names: list[str] | None = None,
        away_names: list[str] | None = None,
        home_team_ids: list[int] | None = None,
        away_team_ids: list[int] | None = None,
    ) -> list[HistoricalMatchModel]:
        season = get_season(season)
        resolved_home_ids = home_team_ids or self._team_ids_for_names(home_names or [])
        resolved_away_ids = away_team_ids or self._team_ids_for_names(away_names or [])
        if not resolved_home_ids or not resolved_away_ids:
            return []

        query = select(self.model).where(
            self.model.league == league_code,
            self.model.season == season,
            self.model.home_team_id.in_(resolved_home_ids),
            self.model.away_team_id.in_(resolved_away_ids),
        )
        result = list(self.session.scalars(query).all())
        if result:
            return result

        query = select(self.model).where(
            self.model.season == season,
            self.model.home_team_id.in_(resolved_home_ids),
            self.model.away_team_id.in_(resolved_away_ids),
        )
        return list(self.session.scalars(query).all())

    def get_filtered(
        self,
        *,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        before_date: date | None = None,
        limit: int | None = None,
    ) -> list[HistoricalMatchModel]:
        """Return historical matches, optionally capped to newest ``limit`` rows."""
        query = select(self.model)
        if leagues:
            query = query.where(self.model.league.in_(leagues))
        if seasons:
            query = query.where(self.model.season.in_(seasons))
        if before_date:
            query = query.where(self.model.match_date < before_date)
        if limit is not None:
            query = query.order_by(self.model.match_date.desc()).limit(limit)
        else:
            query = query.order_by(self.model.match_date.asc())
        return list(self.session.scalars(query).all())

    def get_matches_by_team(self, team: TeamModel) -> list[HistoricalMatch]:
        query = (
            select(self.model)
            .where(
                or_(
                    self.model.home_team_id == team.id,
                    self.model.away_team_id == team.id,
                )
            )
            .order_by(self.model.match_date.asc())
        )
        return [self.to_schema(m) for m in self.session.scalars(query).all()]

    def get_cross_league_matches(self):
        home = aliased(TeamModel)
        away = aliased(TeamModel)
        query = (
            select(self.model)
            .join(home, home.id == self.model.home_team_id)
            .join(away, away.id == self.model.away_team_id)
            .where(home.league_id != away.league_id)
        )
        return list(self.session.scalars(query).all())

    def get_num_teams_by_league(self, league_id: int):
        stmt = text(f"SELECT COUNT(*) FROM teams WHERE league_id={league_id};")
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_home_goals_sum_by_league(
        self, league_id: int, season: str, before_date: date
    ) -> tuple[int, int]:
        league = self.league_repo.get(league_id)
        season = get_season(season)
        stmt = text(
            f"SELECT SUM(home_goals), COUNT(*) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season='{season}' AND match_date < '{before_date}';"
        )
        result = self.session.execute(stmt)
        goals_sum, match_count = result.one()
        return (goals_sum or 0, match_count or 0)

    def get_away_goals_sum_by_league(
        self, league_id: int, season: str, before_date: date
    ) -> tuple[int, int]:
        league = self.league_repo.get(league_id)
        season = get_season(season)
        stmt = text(
            f"SELECT SUM(away_goals), COUNT(*) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season='{season}' AND match_date < '{before_date}';"
        )
        result = self.session.execute(stmt)
        goals_sum, match_count = result.one()
        return (goals_sum or 0, match_count or 0)

    def get_goal_sums_by_league_before_date(
        self, before_date: date
    ) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate home/away goals and counts by league code before cutoff."""
        rows = self.session.execute(
            select(
                self.model.league,
                func.coalesce(func.sum(self.model.home_goals), 0),
                func.count(),
                func.coalesce(func.sum(self.model.away_goals), 0),
                func.count(),
            )
            .where(self.model.match_date < before_date)
            .group_by(self.model.league)
        ).all()
        return {
            league: (int(sum_home), int(home_count), int(sum_away), int(away_count))
            for league, sum_home, home_count, sum_away, away_count in rows
        }

    def get_goal_average_by_league(self, league_id: int):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(
            f"SELECT SUM(home_goals), SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        totals = result.one()
        return (totals[0] + totals[1]) / num_teams

    def get_home_goal_average_by_league(self, league_id: int):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0] / num_teams

    def get_home_goal_average_by_league_before_date(self, league_id: int, before_date: date):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND match_date <= '{before_date}';"
        )
        result = self.session.execute(stmt)
        return result.one()[0] / num_teams

    def get_away_goal_average_by_league(self, league_id: int, before_date: date | None = None):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        if before_date is not None:
            stmt = text(
                f"SELECT SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND match_date <= '{before_date}';"
            )
        else:
            stmt = text(
                f"SELECT SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
            )
        result = self.session.execute(stmt)
        return result.one()[0] / num_teams

    def get_scored_home_goals_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE home_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_scored_away_goals_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE away_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_lost_home_goals_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE home_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_number_of_home_matches_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT COUNT(*) FROM historical_matches WHERE home_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_number_of_away_matches_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT COUNT(*) FROM historical_matches WHERE away_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_lost_away_goals_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE away_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        result = self.session.execute(stmt)
        return result.one()[0]

    def get_scored_goal_average_by_team(self, team_name: str):
        team = self.team_repo.get_by_name(team_name)
        if not team:
            return

        league = self.league_repo.get(team.league_id)
        average_scored_goals = self.get_goal_average_by_league(team.league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE home_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        home_goals = self.session.execute(stmt).one()[0]
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE away_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        away_goals = self.session.execute(stmt).one()[0]
        return (home_goals + away_goals) / average_scored_goals

    def get_lost_goal_average_by_team(self, team_name: str):
        team = self.team_repo.get_by_name(team_name)
        if not team:
            return
        league = self.league_repo.get(team.league_id)
        average_lost_goals = self.get_goal_average_by_league(team.league_id)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE home_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        away_goals = self.session.execute(stmt).one()[0]
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE away_team_id={team.id} AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};"
        )
        home_goals = self.session.execute(stmt).one()[0]
        return (home_goals + away_goals) / average_lost_goals

    _CONFLICTING_CSV_FIELDS = (
        "home_name", "home_id", "away_name", "away_id", "match_date"
    )

    def _append_conflicting_match(
        self, match: HistoricalMatchCreate, csv_path: Path
    ) -> None:
        """Append a conflicting match row to the CSV for manual review."""
        home_team = self.team_repo.get(match.home_team_id)
        away_team = self.team_repo.get(match.away_team_id)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        row = {
            "home_name": home_team.name if home_team else "",
            "home_id": match.home_team_id,
            "away_name": away_team.name if away_team else "",
            "away_id": match.away_team_id,
            "match_date": match.match_date.isoformat(),
        }
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._CONFLICTING_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def upsert_many(
        self,
        matches: list[HistoricalMatchCreate],
        conflicting_csv_path: Path | None = None,
    ) -> int:
        from objects.schema.data_classes.data_sources import DataSourceConfig

        csv_path = conflicting_csv_path or DataSourceConfig().conflicting_matches_csv_path
        written = 0
        seen_team_dates: set[tuple[int, date]] = set()
        for match in matches:
            home_key = (match.home_team_id, match.match_date)
            away_key = (match.away_team_id, match.match_date)
            if home_key in seen_team_dates or away_key in seen_team_dates:
                logger.warning(
                    "Skipping historical match on %s: team already in this batch "
                    "(home_team_id=%s away_team_id=%s)",
                    match.match_date,
                    match.home_team_id,
                    match.away_team_id,
                )
                continue
            conflicting = self.team_has_other_match_on_date(match)
            if conflicting is not None:
                logger.warning(
                    "Skipping historical match on %s: team already plays that day "
                    "(home_team_id=%s away_team_id=%s conflicting_id=%s)",
                    match.match_date,
                    match.home_team_id,
                    match.away_team_id,
                    conflicting.id,
                )
                self._append_conflicting_match(match, csv_path)
                continue
            stmt = pg_insert(self.model).values(
                source=match.source,
                league=match.league,
                season=match.season,
                match_date=match.match_date,
                home_team_id=match.home_team_id,
                away_team_id=match.away_team_id,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                result=match.result,
                odds_home=match.odds_home,
                odds_draw=match.odds_draw,
                odds_away=match.odds_away,
                raw_data=json_safe(match.raw_data) if match.raw_data else None,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    "source",
                    "league",
                    "season",
                    "match_date",
                    "home_team_id",
                    "away_team_id",
                ],
                set_={
                    "home_goals": stmt.excluded.home_goals,
                    "away_goals": stmt.excluded.away_goals,
                    "result": stmt.excluded.result,
                    "odds_home": stmt.excluded.odds_home,
                    "odds_draw": stmt.excluded.odds_draw,
                    "odds_away": stmt.excluded.odds_away,
                    "raw_data": stmt.excluded.raw_data,
                },
            )
            self.session.execute(stmt)
            seen_team_dates.add(home_key)
            seen_team_dates.add(away_key)
            written += 1
        self.session.commit()
        return written

    def to_schema(self, model: HistoricalMatchModel) -> HistoricalMatch:
        market_probabilities = None
        if all(
            odds is not None and odds > 0
            for odds in (model.odds_home, model.odds_draw, model.odds_away)
        ):
            market_probabilities = odds_to_probabilities(
                model.odds_home,
                model.odds_draw,
                model.odds_away,
            )
        return HistoricalMatch(
            id=model.id,
            source=model.source,
            league=model.league,
            season=model.season,
            match_date=model.match_date,
            home_team_id=model.home_team_id,
            away_team_id=model.away_team_id,
            home_team=model.home_team.name if model.home_team is not None else "",
            away_team=model.away_team.name if model.away_team is not None else "",
            home_goals=model.home_goals,
            away_goals=model.away_goals,
            result=model.result,
            odds_home=model.odds_home,
            odds_draw=model.odds_draw,
            odds_away=model.odds_away,
            raw_data=model.raw_data,
            league_id=model.league,
            actual_outcome=model.result,
            market_home_probability=(
                market_probabilities["1"] if market_probabilities else None
            ),
            market_draw_probability=(
                market_probabilities["X"] if market_probabilities else None
            ),
            market_away_probability=(
                market_probabilities["2"] if market_probabilities else None
            ),
        )
