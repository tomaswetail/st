from typing import Any

from sqlalchemy import delete, select, text, update

from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.models.fixture import FixtureModel
from objects.models.match_shot import MatchShotModel
from objects.models.st_match import STMatchModel
from objects.models.team import TeamModel
from objects.repositories.base import BaseRepository
from utils.common import sanitize_string
from utils.team_name_matcher import to_football_data_name


class TeamRepository(BaseRepository[TeamModel]):
    model = TeamModel

    def get_by_external_id(self, external_id: int) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.external_id == external_id)
        )

    def get_by_likely_name(self, name: str) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.name.ilike(f"%{name}%"))
        )

    def get_by_name(self, name: str) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.name == name)
        )

    def get_by_name_and_league(
        self, name: str, league_id: int | None = None
    ) -> TeamModel | None:
        """Lookup by name; league_id ignored (teams are not league-scoped)."""
        del league_id
        return self.get_by_name(name)

    def find_exact_normalized(self, name: str) -> TeamModel | None:
        """Match an incoming name to a team ignoring case, accents, and punctuation."""
        sql = """
            WITH incoming AS (
                SELECT trim(
                    regexp_replace(lower(unaccent(:name)), '[^a-z0-9]+', ' ', 'g')
                ) AS n
            ),
            normalized AS (
                SELECT
                    id,
                    trim(
                        regexp_replace(lower(unaccent(name)), '[^a-z0-9]+', ' ', 'g')
                    ) AS n
                FROM teams
            )
            SELECT normalized.id
            FROM normalized
            JOIN incoming ON incoming.n = normalized.n
            WHERE incoming.n <> ''
            ORDER BY normalized.id
            LIMIT 1
        """
        return self._first_matching_team(sql, {"name": name})

    def find_by_club_affix(self, name: str) -> TeamModel | None:
        """Match after stripping common club prefixes/suffixes (fc, ik, gif, ...)."""
        sql = r"""
            WITH incoming AS (
                SELECT trim(
                    regexp_replace(
                        regexp_replace(
                            lower(unaccent(:name)),
                            '\m(fc|fk|ik|if|bk|sk|aif|gif)\M',
                            '',
                            'g'
                        ),
                        '\s+',
                        ' ',
                        'g'
                    )
                ) AS n
            ),
            normalized AS (
                SELECT
                    id,
                    trim(
                        regexp_replace(
                            regexp_replace(
                                lower(unaccent(name)),
                                '\m(fc|fk|ik|if|bk|sk|aif|gif)\M',
                                '',
                                'g'
                            ),
                            '\s+',
                            ' ',
                            'g'
                        )
                    ) AS n
                FROM teams
            )
            SELECT normalized.id
            FROM normalized
            JOIN incoming ON similarity(incoming.n, normalized.n) >= 0.75
            WHERE incoming.n <> ''
            ORDER BY similarity(incoming.n, normalized.n) DESC, normalized.id
            LIMIT 1
        """
        return self._first_matching_team(sql, {"name": name})

    def find_fuzzy_duplicate(
        self, name: str, min_similarity: float = 0.80
    ) -> TeamModel | None:
        """Match an incoming name with pg_trgm similarity against existing teams."""
        sql = """
            WITH incoming AS (
                SELECT trim(
                    regexp_replace(lower(unaccent(:name)), '[^a-z0-9]+', ' ', 'g')
                ) AS n
            ),
            normalized AS (
                SELECT
                    id,
                    trim(
                        regexp_replace(lower(unaccent(name)), '[^a-z0-9]+', ' ', 'g')
                    ) AS n
                FROM teams
            )
            SELECT normalized.id
            FROM normalized
            JOIN incoming ON similarity(incoming.n, normalized.n) >= :min_similarity
            WHERE incoming.n <> ''
            ORDER BY similarity(incoming.n, normalized.n) DESC, normalized.id
            LIMIT 1
        """
        return self._first_matching_team(
            sql, {"name": name, "min_similarity": min_similarity}
        )

    def find_substring_duplicate(
        self, name: str, min_length: int = 5
    ) -> TeamModel | None:
        """Match when one normalized name contains the other (short names skipped)."""
        sql = """
            WITH incoming AS (
                SELECT trim(
                    regexp_replace(lower(unaccent(:name)), '[^a-z0-9]+', ' ', 'g')
                ) AS n
            ),
            normalized AS (
                SELECT
                    id,
                    trim(
                        regexp_replace(lower(unaccent(name)), '[^a-z0-9]+', ' ', 'g')
                    ) AS n
                FROM teams
            )
            SELECT normalized.id
            FROM normalized
            JOIN incoming ON (
                incoming.n LIKE '%' || normalized.n || '%'
                OR normalized.n LIKE '%' || incoming.n || '%'
            )
            WHERE incoming.n <> ''
              AND length(LEAST(incoming.n, normalized.n)) >= :min_length
            ORDER BY normalized.id
            LIMIT 1
        """
        return self._first_matching_team(
            sql, {"name": name, "min_length": min_length}
        )

    def _first_matching_team(
        self, sql: str, params: dict[str, Any]
    ) -> TeamModel | None:
        name = params.get("name")
        if not name or not str(name).strip():
            return None
        team_id = self.session.execute(text(sql), params).scalar()
        if team_id is None:
            return None
        return self.get(int(team_id))

    def get_all_names(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.name).distinct().order_by(self.model.name)
            ).all()
        )

    def get_names_by_league_external_id(self, league_external_id: int) -> list[str]:
        """Return distinct team names that appear in fixtures for an API league id."""
        return list(
            self.session.scalars(
                select(FixtureModel.home_team_name)
                .where(FixtureModel.league_id == league_external_id)
                .distinct()
            ).all()
        )

    def to_football_data_name(self, svenska_spel_name: str) -> str | None:
        if not hasattr(self, "_football_data_names"):
            from objects.repositories.fixture_repository import FixtureRepository

            fixture_repo = FixtureRepository(self.session)
            self._football_data_names = fixture_repo.get_distinct_home_teams()
        return to_football_data_name(svenska_spel_name, self._football_data_names)

    def team_name_wide_search(self, team_name: str) -> TeamModel | None:
        if not team_name:
            return None
        if len(team_name.split(" ")) > 1:
            return self.get_by_likely_name(team_name.split(" ")[0])
        return self.get_by_likely_name(team_name)

    def team_likely_name_wide_search(self, team_name: str) -> TeamModel | None:
        return self.team_name_wide_search(sanitize_string(team_name) if team_name else "")

    def create_from_provider_team(
        self,
        *,
        external_id: int,
        name: str,
        code: str | None = None,
        country: str | None = None,
        national: bool = False,
    ) -> TeamModel:
        """Create a team from provider data, or return existing by external_id."""
        existing = self.get_by_external_id(external_id)
        if existing is not None:
            existing.name = name
            if code is not None:
                existing.code = code
            if country is not None:
                existing.country = country
            existing.national = national
            return existing
        return self.create(
            external_id=external_id,
            name=name,
            code=code,
            country=country,
            national=national,
        )

    def upsert_from_participant(self, participant: dict[str, Any]) -> TeamModel:
        external_id = int(participant["id"])
        return self.create_from_provider_team(
            external_id=external_id,
            name=participant["name"],
            code=participant.get("code") or participant.get("isoCode"),
            country=participant.get("country") or participant.get("countryName"),
            national=bool(participant.get("national", False)),
        )

    def merge_duplicate(
        self, keep_team_id: int, remove_team_id: int
    ) -> dict[str, int]:
        """Reassign FKs from remove_team_id onto keep_team_id, then delete remove."""
        if keep_team_id == remove_team_id:
            raise ValueError("keep_team_id and remove_team_id must differ")
        keep_team = self.get(keep_team_id)
        remove_team = self.get(remove_team_id)
        if keep_team is None:
            raise ValueError(f"keep team id={keep_team_id} not found")
        if remove_team is None:
            raise ValueError(f"remove team id={remove_team_id} not found")

        fixtures_updated = self._reassign_fixtures(
            keep_external_id=keep_team.external_id,
            remove_external_id=remove_team.external_id,
        )
        st_updated = self._reassign_st_matches(
            keep_team_id=keep_team_id,
            remove_team_id=remove_team_id,
        )
        shots_updated = self.session.execute(
            update(MatchShotModel)
            .where(MatchShotModel.team_id == remove_team_id)
            .values(team_id=keep_team_id)
        ).rowcount or 0
        mappings_deleted = self.session.execute(
            delete(ExternalEntityMappingModel).where(
                ExternalEntityMappingModel.entity_type == "team",
                ExternalEntityMappingModel.internal_entity_id == remove_team_id,
            )
        ).rowcount or 0

        self.delete(remove_team)
        self.session.commit()

        return {
            "fixtures_updated": int(fixtures_updated),
            "st_updated": int(st_updated),
            "shots_updated": int(shots_updated),
            "mappings_deleted": int(mappings_deleted),
            "removed_team_id": remove_team_id,
        }

    def _reassign_fixtures(
        self, *, keep_external_id: int, remove_external_id: int
    ) -> int:
        home_count = self.session.execute(
            update(FixtureModel)
            .where(FixtureModel.home_team_id == remove_external_id)
            .values(home_team_id=keep_external_id)
        ).rowcount or 0
        away_count = self.session.execute(
            update(FixtureModel)
            .where(FixtureModel.away_team_id == remove_external_id)
            .values(away_team_id=keep_external_id)
        ).rowcount or 0
        return int(home_count) + int(away_count)

    def _reassign_st_matches(
        self, *, keep_team_id: int, remove_team_id: int
    ) -> int:
        home_count = self.session.execute(
            update(STMatchModel)
            .where(STMatchModel.home_team_id == remove_team_id)
            .values(home_team_id=keep_team_id)
        ).rowcount or 0
        away_count = self.session.execute(
            update(STMatchModel)
            .where(STMatchModel.away_team_id == remove_team_id)
            .values(away_team_id=keep_team_id)
        ).rowcount or 0
        return int(home_count) + int(away_count)
