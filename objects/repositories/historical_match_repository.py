from __future__ import annotations

from datetime import date
from typing import Literal

from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from objects.models.historical_match import HistoricalMatchModel
from objects.models.team import TeamModel

from objects.schema.db.historical_match import HistoricalMatch, HistoricalMatchCreate

from objects.repositories.base import BaseRepository
from objects.repositories.utils import json_safe
from utils.common import LEAGUE_NAMES_REV, fix_swedish_name, odds_to_probabilities, get_season

SEASONS = "('2425','2526')"

class HistoricalMatchRepository(BaseRepository[HistoricalMatchModel]):
    model = HistoricalMatchModel

    def __init__(self, session: Session | None = None) -> None:
        super().__init__(session)
        from objects.repositories.league_repository import LeagueRepository
        from objects.repositories.team_repository import TeamRepository

        self.team_repo = TeamRepository(self.session)
        self.league_repo = LeagueRepository(self.session)

    def find_before_date_by_team(
        self,
        *,
        team_name: str,
        before_date: date,
        venue: Literal["home", "away"] | None = None,
        limit: int,
    ) -> list[HistoricalMatchModel]:
        """Newest-first matches for a team name before cutoff, optionally venue-filtered."""
        query = (
            select(self.model)
            .where(
                self.model.match_date < before_date,
                *self._venue_filters(team_name, venue),
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
        if not team_names:
            return []
        query = (
            select(self.model)
            .where(
                self.model.match_date < before_date,
                or_(
                    self.model.home_team.in_(team_names),
                    self.model.away_team.in_(team_names),
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
        """Newest-first league matches before cutoff (excludes cups/Europe)."""
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
        text = str(season).strip()
        if len(text) == 4 and text.isdigit():
            start = int(text[:2])
            end = int(text[2:])
            if end == (start + 1) % 100:
                return text
            if int(text) >= 1900:
                try:
                    return get_season(text)
                except KeyError:
                    return text
        try:
            return get_season(text)
        except KeyError:
            return text

    @staticmethod
    def _venue_filters(
        team_name: str, venue: Literal["home", "away"] | None
    ) -> list:
        """SQLAlchemy filters for home-only, away-only, or either venue."""
        if venue == "home":
            return [HistoricalMatchModel.home_team == team_name]
        if venue == "away":
            return [HistoricalMatchModel.away_team == team_name]
        return [
            or_(
                HistoricalMatchModel.home_team == team_name,
                HistoricalMatchModel.away_team == team_name,
            )
        ]

    def get_by_match_key(
        self,
        *,
        source: str,
        league: str,
        season: str,
        match_date: date,
        home_team: str,
        away_team: str,
    ) -> HistoricalMatchModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.source == source,
                self.model.league == league,
                self.model.season == season,
                self.model.match_date == match_date,
                self.model.home_team == home_team,
                self.model.away_team == away_team,
            )
        )

    def get_by_date_and_teams(
        self,
        match_date: date,
        home_team: str,
        away_team: str,
    ) -> HistoricalMatch | None:

        home_team = self.team_repo.to_football_data_name(home_team) or home_team
        away_team = self.team_repo.to_football_data_name(away_team) or away_team

        stmt = (
            select(self.model)
            .where(
                self.model.match_date == match_date,
                self.model.home_team == home_team,
                self.model.away_team == away_team,
            )
            .order_by(self.model.id.desc())
            .limit(1)
        )
        model = self.session.scalar(stmt)

        query = stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
        if model is None:
            return None
        return self.to_schema(model)

    def get_distinct_home_teams(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.home_team)
                .distinct()
                .order_by(self.model.home_team)
            ).all()
        )

    def get_distinct_away_teams(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.away_team)
                .distinct()
                .order_by(self.model.away_team)
            ).all()
        )

    def find_by_date_range_and_teams(
        self,
        *,
        date_from: date,
        date_to: date,
        home_names: list[str],
        away_names: list[str],
        league_code: str | None = None,
    ) -> list[HistoricalMatchModel]:
        query = select(self.model).where(
            self.model.match_date >= date_from,
            self.model.match_date <= date_to,
            self.model.home_team.in_(home_names),
            self.model.away_team.in_(away_names),
        )
        #if league_code:
        #    query = query.where(self.model.league == league_code)
        return list(self.session.scalars(query).all())

    def find_by_season_and_teams(
        self,
        *,
        league_code: str,
        season: str,
        home_names: list[str],
        away_names: list[str],
    ) -> list[HistoricalMatchModel]:

        season = get_season(season)
        query = select(self.model).where(
            self.model.league == league_code,
            self.model.season == season,
            self.model.home_team.in_(home_names),
            self.model.away_team.in_(away_names),
        )

        result = list(self.session.scalars(query).all())

        if not result:
            query = select(self.model).where(
                self.model.season == season,
                self.model.home_team.in_(home_names),
                self.model.away_team.in_(away_names),
            )
            return list(self.session.scalars(query).all())

    def get_filtered(
        self,
        *,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        before_date: date | None = None,
    ) -> list[HistoricalMatchModel]:
        print(f"Pre get_filtered")
        query = select(self.model)
        if leagues:
            query = query.where(self.model.league.in_(leagues))
        if seasons:
            query = query.where(self.model.season.in_(seasons))
        if before_date:
            query = query.where(self.model.match_date < before_date)
        query = query.order_by(self.model.match_date.asc())
        return list(self.session.scalars(query).all())

    def get_matches_by_team(self, team: TeamModel) -> list[HistoricalMatch]:
        query = (
            select(self.model)
            .where(
                or_(
                    self.model.home_team == team.name,
                    self.model.away_team == team.name,
                )
            )
            .order_by(self.model.match_date.asc())
        )
        return [self.to_schema(m) for m in self.session.scalars(query).all()]

    def get_cross_league_matches(self):
        query = select(self.model).join(
            TeamModel, TeamModel.name == self.model.home_team
        )
        query = query.join(
            TeamModel, TeamModel.name == self.model.away_team
        )
        return list(self.session.scalars(query).all())

    def get_num_teams_by_league(self, league_id: int):
        stmt = text(f"SELECT COUNT(*) FROM teams WHERE league_id={league_id};")
        result = self.session.execute(stmt)
        t = result.one()[0]
        return t

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

    def get_goal_average_by_league(self, league_id: int):

        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(f"SELECT SUM(home_goals), SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};") #TODO
        result = self.session.execute(stmt)
        t = result.one()
        return (t[0] + t[1]) / num_teams

    def get_home_goal_average_by_league(self, league_id: int):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(f"SELECT SUM(home_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};") #TODO
        result = self.session.execute(stmt)
        t = result.one()
        return t[0] / num_teams

    def get_home_goal_average_by_league_before_date(self, league_id: int, before_date: date):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(f"SELECT SUM(home_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND match_date <= '{before_date}';") #TODO
        result = self.session.execute(stmt)
        t = result.one()
        return t[0] / num_teams

    def get_away_goal_average_by_league(self, league_id: int):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(f"SELECT SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};") #TODO
        result = self.session.execute(stmt)
        t = result.one()
        return t[0] / num_teams

    def get_away_goal_average_by_league(self, league_id: int, before_date: date):
        num_teams = self.get_num_teams_by_league(league_id)
        league = self.league_repo.get(league_id)
        stmt = text(f"SELECT SUM(away_goals) FROM historical_matches WHERE league='{LEAGUE_NAMES_REV[league.name]}' AND match_date <= '{before_date}';") #TODO
        result = self.session.execute(stmt)
        t = result.one()
        return t[0] / num_teams

    def get_scored_home_goals_by_team(self, team: TeamModel):
        team_name = fix_swedish_name(team.name)
        league = self.league_repo.get(team.league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE home_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        home_goals = result.one()[0]
        return home_goals

    def get_scored_away_goals_by_team(self, team: TeamModel):

        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team.name)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE away_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        away_goals = result.one()[0]
        return away_goals

    def get_lost_home_goals_by_team(self, team: TeamModel):

        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team.name)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE home_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        lost_goals = result.one()[0]
        return lost_goals

    def get_number_of_home_matches_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team.name)
        stmt = text(
            f"SELECT COUNT(*) FROM historical_matches WHERE home_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        num_matches = result.one()[0]
        return num_matches

    def get_number_of_away_matches_by_team(self, team: TeamModel):
        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team.name)
        stmt = text(
            f"SELECT COUNT(*) FROM historical_matches WHERE away_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        num_matches = result.one()[0]
        return num_matches

    def get_lost_away_goals_by_team(self, team: TeamModel):

        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team.name)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE away_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        lost_goals = result.one()[0]
        return lost_goals

    def get_scored_goal_average_by_team(self, team_name: str):
        team = self.team_repo.get_by_name(team_name)
        if not team:
            return

        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team_name)
        average_scored_goals = self.get_goal_average_by_league(team.league_id)
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE home_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        home_goals = result.one()[0]
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE away_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        away_goals = result.one()[0]
        return (home_goals + away_goals) / average_scored_goals


    def get_lost_goal_average_by_team(self, team_name: str):
        team = self.team_repo.get_by_name(team_name)
        if not team:
            return
        league = self.league_repo.get(team.league_id)
        team_name = fix_swedish_name(team_name)
        average_lost_goals = self.get_goal_average_by_league(team.league_id)
        stmt = text(
            f"SELECT SUM(away_goals) FROM historical_matches WHERE home_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        away_goals = result.one()[0]
        stmt = text(
            f"SELECT SUM(home_goals) FROM historical_matches WHERE away_team='{team_name}' AND league='{LEAGUE_NAMES_REV[league.name]}' AND season IN {SEASONS};")  # TODO
        result = self.session.execute(stmt)
        home_goals = result.one()[0]
        return (home_goals + away_goals) / average_lost_goals


    def upsert_many(self, matches: list[HistoricalMatchCreate]) -> int:
        for match in matches:
            stmt = pg_insert(self.model).values(
                source=match.source,
                league=match.league,
                season=match.season,
                match_date=match.match_date,
                home_team=match.home_team,
                away_team=match.away_team,
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
                    "home_team",
                    "away_team",
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
        self.session.commit()
        return len(matches)

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
            home_team=model.home_team,
            away_team=model.away_team,
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