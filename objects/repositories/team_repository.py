from typing import Any

from sqlalchemy import case, delete, exists, or_, select, text, update
from sqlalchemy.exc import ProgrammingError

from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.models.historical_match import HistoricalMatchModel
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

    def get_by_name_and_league(
        self, name: str, league_id: int
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name == name,
                self.model.league_id == league_id,
            )
        )

    def get_by_likely_name(
        self, name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name.ilike(f'%{name}%')
            )
        )


    def get_by_name(
        self, name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name == name
            )
        )

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

    def get_by_likely_machine_name(
        self, machine_name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.machine_name.ilike(f'%{machine_name}%')
            )
        )


    def get_by_machine_name(
        self, machine_name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.machine_name == machine_name
            )
        )

    def get_all_names(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.name).distinct().order_by(self.model.name)
            ).all()
        )

    def get_names_by_league_id(self, league_id: int) -> list[str]:
        """Return team names belonging to a league."""
        return list(
            self.session.scalars(
                select(self.model.name).where(self.model.league_id == league_id)
            ).all()
        )

    def to_football_data_name(self, svenska_spel_name: str) -> str | None:
        if not hasattr(self, "_football_data_names"):
            from objects.repositories.historical_match_repository import HistoricalMatchRepository

            historical_repo = HistoricalMatchRepository(self.session)
            self._football_data_names = historical_repo.get_distinct_home_teams()
        return to_football_data_name(svenska_spel_name, self._football_data_names)

    def team_name_wide_search(self, team_name):
        if not team_name:
            return
        if len(team_name.split(' ')) > 1:
            _team_name = team_name.split(' ')[0]
            team = self.get_by_likely_name(_team_name)
            return team
        return self.get_by_likely_name(team_name)

    def team_likely_name_wide_search(self, team_name):
        if not team_name:
            return
        if len(team_name.split(' ')) > 1:
            team_name = team_name.split(' ')[0]
            team_name = sanitize_string(team_name)
            team = self.get_by_likely_machine_name(team_name)
            return team
        return self.get_by_likely_machine_name(sanitize_string(team_name))


    def ensure_from_historical(self, name: str, league_id: int) -> TeamModel:
        team = self.get_by_name_and_league(name, league_id)
        if team is None:
            team = self.get_by_likely_name(name)
        if team is None:
            if len(name.split(' ')) > 1:
                name = name.split(' ')[0]
                team = self.get_by_likely_name(name)
        if team is None:
            team = self.get_by_machine_name(sanitize_string(name))
        if team is None:
            if len(name.split(' ')) > 1:
                name = name.split(' ')[0]
                team = self.get_by_likely_machine_name(name)
        if team is None:
            team = self.create(name=name, machine_name=sanitize_string(name), league_id=league_id)
        return team

    def create_from_provider_team(
        self,
        *,
        name: str,
        league_id: int,
        short_name: str | None = None,
        country_name: str | None = None,
        iso_code: str | None = None,
    ) -> TeamModel:
        """Create a team from provider profile data, or return existing name+league row."""
        existing = self.get_by_name_and_league(name, league_id)
        if existing is not None:
            return existing
        return self.create(
            name=name,
            machine_name=sanitize_string(name),
            short_name=short_name,
            league_id=league_id,
            country_name=country_name,
            iso_code=iso_code,
        )

    def upsert_from_participant(self, participant: dict[str, Any]) -> TeamModel:
        external_id = participant["id"]
        team = self.get_by_external_id(external_id)
        if team is None:
            team = self.create(external_id=external_id)

        team.name = participant["name"]
        team.machine_name = sanitize_string(participant["name"])
        team.short_name = participant.get("shortName")
        team.medium_name = participant.get("mediumName")
        team.country_name = participant.get("countryName")
        team.iso_code = participant.get("isoCode")
        if participant.get("league_id") is not None:
            team.league_id = participant["league_id"]
        return team

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

        historical_deleted = self._delete_historical_merge_conflicts(
            keep_team_id=keep_team_id,
            remove_team_id=remove_team_id,
        )
        historical_updated = self._reassign_historical_matches(
            keep_team_id=keep_team_id,
            remove_team_id=remove_team_id,
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
            "historical_updated": int(historical_updated),
            "historical_deleted_conflicts": int(historical_deleted),
            "st_updated": int(st_updated),
            "shots_updated": int(shots_updated),
            "mappings_deleted": int(mappings_deleted),
            "removed_team_id": remove_team_id,
        }

    def _delete_historical_merge_conflicts(
        self, *, keep_team_id: int, remove_team_id: int
    ) -> int:
        """Delete remove-team matches that would violate the unique key after remapping."""
        remove_match = HistoricalMatchModel
        other = HistoricalMatchModel.__table__.alias("other_hm")

        remapped_home = case(
            (remove_match.home_team_id == remove_team_id, keep_team_id),
            else_=remove_match.home_team_id,
        )
        remapped_away = case(
            (remove_match.away_team_id == remove_team_id, keep_team_id),
            else_=remove_match.away_team_id,
        )

        conflict_ids = (
            select(remove_match.id)
            .where(
                or_(
                    remove_match.home_team_id == remove_team_id,
                    remove_match.away_team_id == remove_team_id,
                ),
                exists(
                    select(1)
                    .select_from(other)
                    .where(
                        other.c.id != remove_match.id,
                        other.c.source == remove_match.source,
                        other.c.league == remove_match.league,
                        other.c.season == remove_match.season,
                        other.c.match_date == remove_match.match_date,
                        other.c.home_team_id == remapped_home,
                        other.c.away_team_id == remapped_away,
                    )
                ),
            )
        )
        result = self.session.execute(
            delete(HistoricalMatchModel).where(
                HistoricalMatchModel.id.in_(conflict_ids)
            )
        )
        return result.rowcount or 0

    def _reassign_historical_matches(
        self, *, keep_team_id: int, remove_team_id: int
    ) -> int:
        home_count = self.session.execute(
            update(HistoricalMatchModel)
            .where(HistoricalMatchModel.home_team_id == remove_team_id)
            .values(home_team_id=keep_team_id)
        ).rowcount or 0
        away_count = self.session.execute(
            update(HistoricalMatchModel)
            .where(HistoricalMatchModel.away_team_id == remove_team_id)
            .values(away_team_id=keep_team_id)
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
