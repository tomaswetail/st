import logging
from typing import Any

from sqlalchemy import select

from objects.models.league import LeagueModel
from utils.common import LEAGUE_COUNTRIES, LEAGUE_NAMES

from objects.repositories.base import BaseRepository


class LeagueRepository(BaseRepository[LeagueModel]):
    model = LeagueModel

    def get_by_name(self, name: str) -> LeagueModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.name == name)
        )

    def get_by_name_and_country(
        self, name: str, country: str
    ) -> LeagueModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name == name,
                self.model.country == country,
            )
        )

    def get_by_likely_name_and_country(
        self, name: str, country: str
    ) -> LeagueModel | None:
        matches = self.session.scalars(
            select(self.model).where(
                self.model.name.ilike(f"%{name}%"),
                self.model.country == country,
            )
        ).all()

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            return matches[1]
        return None#TODO

    def upsert_from_match(self, league: dict[str, Any]) -> LeagueModel:
        name = league["name"]
        _league = self.get_by_name(name)
        if _league is None:
            _league = self.create(name=name)

        _league.country = league.get("country")
        return _league

    def get_by_code(self, league_code: str) -> LeagueModel | None:
        name = LEAGUE_NAMES.get(league_code)
        if not name:
            return None
        return self.get_by_name(name)

    def ensure_from_code(self, league_code: str) -> LeagueModel | None:
        name = LEAGUE_NAMES.get(league_code)
        country = LEAGUE_COUNTRIES.get(league_code)
        if not name or not country:
            logging.warning("LEAGUE NOT FOUND")
            return None

        league = self.get_by_name(name)
        if league is None:
            league = self.create(name=name, country=country)
        return league
