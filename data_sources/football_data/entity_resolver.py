"""Resolve provider entities to internal leagues, teams, and historical matches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from objects.models.historical_match import HistoricalMatchModel
from objects.models.league import LeagueModel
from objects.models.team import TeamModel
from objects.repositories.external_entity_mapping_repository import (
    ExternalEntityMappingRepository,
)
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import ProviderMatch
from utils.common import LEAGUE_NAMES_REV, sanitize_string
from utils.team_name_matcher import _load_aliases, normalize_team_name

logger = logging.getLogger(__name__)


@dataclass
class TeamResolution:
    """Result of resolving a provider team to an internal team."""
    team: TeamModel | None
    confidence: float
    method: str
    unresolved_name: str | None = None


@dataclass
class MatchResolution:
    """Result of resolving a provider fixture to a historical match."""
    match: HistoricalMatchModel | None
    method: str
    warnings: list[str]


class EntityResolver:
    """Map provider leagues, teams, and matches onto internal entities."""
    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        provider: str = "fotmob",
    ) -> None:
        """Wire repositories and team-name aliases for a provider."""
        self.session = session
        self.config = DataSourceConfig()
        self.provider = provider
        self.mapping_repo = ExternalEntityMappingRepository(session)
        self.team_repo = TeamRepository(session)
        self.league_repo = LeagueRepository(session)
        self.historical_repo = HistoricalMatchRepository(session)
        self._aliases = _load_aliases()
        self._team_name_cache: list[str] | None = None

    def resolve_provider_league_id(self, league_id: int) -> str | None:
        """Look up the external league id for an internal league."""
        mapping = self.mapping_repo.get_by_internal(
            provider=self.provider,
            entity_type="league",
            internal_entity_id=league_id,
        )
        if mapping is not None:
            return mapping.external_entity_id
        logger.warning(
            "No %s league mapping for internal league_id=%s",
            self.provider,
            league_id,
        )
        return None

    def resolve_league(self, league_id: int) -> LeagueModel | None:
        """Load an internal LeagueModel by id."""
        return self.league_repo.get(league_id)

    def football_data_league_code(self, league: LeagueModel) -> str | None:
        """Map an internal league name to its football-data code."""
        return LEAGUE_NAMES_REV.get(league.name)



    def resolve_team(
        self,
        *,
        provider_team_id: str,
        provider_team_name: str,
        league_id: int | None = None,
    ) -> TeamResolution:
        """Resolve a provider team via mapping, exact name, alias, or fuzzy match."""
        mapping = self.mapping_repo.get_by_external(
            provider=self.provider,
            entity_type="team",
            external_entity_id=str(provider_team_id),
        )
        if mapping is not None:
            team = self.team_repo.get(mapping.internal_entity_id)
            if team is not None:
                return TeamResolution(team=team, confidence=1.0, method="mapping")

        candidates = self._candidate_team_names()
        exact = self._find_by_normalized_name(provider_team_name, candidates)
        if exact is not None:
            team = self._get_team_by_name(exact, league_id)
            if team is not None:
                return TeamResolution(team=team, confidence=1.0, method="exact_name")

        alias = self._aliases.get(provider_team_name)
        if alias:
            team = self._get_team_by_name(alias, league_id)
            if team is not None:
                return TeamResolution(team=team, confidence=0.98, method="alias")
            aliased_exact = self._find_by_normalized_name(alias, candidates)
            if aliased_exact is not None:
                team = self._get_team_by_name(aliased_exact, league_id)
                if team is not None:
                    return TeamResolution(team=team, confidence=0.98, method="alias")
        else:
            team = self._get_team_by_name_fuzzy(provider_team_name)
            if team is not None:
                return TeamResolution(team=team, confidence=0.75, method="alias")
            team = self._get_team_by_machine_name(provider_team_name)
            if team is not None:
                return TeamResolution(team=team, confidence=1.0, method="exact_name")
            team = self._get_team_by_machine_name_fuzzy(provider_team_name)
            if team is not None:
                return TeamResolution(team=team, confidence=0.9, method="alias")

        threshold = self.config.fuzzy_match_threshold / 100
        best_name = ""
        best_score = 0.0
        target = normalize_team_name(provider_team_name)
        for candidate in candidates:
            score = SequenceMatcher(
                None, target, normalize_team_name(candidate)
            ).ratio()
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_name and best_score >= threshold:
            team = self._get_team_by_name(best_name, league_id)
            if team is not None:
                return TeamResolution(
                    team=team,
                    confidence=best_score,
                    method="fuzzy",
                )

        logger.warning(
            "Unresolved team provider=%s id=%s name=%s best_score=%.3f",
            self.provider,
            provider_team_id,
            provider_team_name,
            best_score,
        )
        return TeamResolution(
            team=None,
            confidence=best_score,
            method="unresolved",
            unresolved_name=provider_team_name,
        )

    def resolve_match(
        self,
        provider_match: ProviderMatch,
        *,
        league_code: str | None,
        league_id: int | None,
        home_team: TeamModel | None,
        away_team: TeamModel | None,
        season: str | None = None,
    ) -> MatchResolution:
        """Resolve a provider fixture to a historical match by mapping or date/teams."""
        warnings: list[str] = []
        mapping = self.mapping_repo.get_by_external(
            provider=self.provider,
            entity_type="match",
            external_entity_id=str(provider_match.provider_match_id),
        )
        if mapping is not None:
            match = self.historical_repo.get(mapping.internal_entity_id)
            if match is not None:
                return MatchResolution(match=match, method="mapping", warnings=warnings)

        home_names = self._names_for_team(home_team, provider_match.home_team_name, league_id)
        away_names = self._names_for_team(away_team, provider_match.away_team_name, league_id)
        kickoff = provider_match.kickoff_at
        tolerance = timedelta(minutes=self.config.kickoff_match_tolerance_minutes)
        date_from = (kickoff - tolerance).date()
        date_to = (kickoff + tolerance).date()

        candidates = self.historical_repo.find_by_date_range_and_teams(
            date_from=date_from,
            date_to=date_to,
            home_names=home_names,
            away_names=away_names,
            league_code=league_code,
        )
        if len(candidates) == 1:
            return MatchResolution(
                match=candidates[0],
                method="date_teams",
                warnings=warnings,
            )
        if len(candidates) > 1:
            # Prefer closest date; handle postponed fixtures with same teams.
            candidates.sort(
                key=lambda match: abs((match.match_date - kickoff.date()).days)
            )
            warnings.append(
                f"Multiple historical matches for {provider_match.home_team_name} vs "
                f"{provider_match.away_team_name}; chose id={candidates[0].id}"
            )
            return MatchResolution(
                match=candidates[0],
                method="date_teams_ambiguous",
                warnings=warnings,
            )

        # Postponed / rescheduled: widen to season + teams without strict date.
        if season and league_code and home_names and away_names:
            season_candidates = self.historical_repo.find_by_season_and_teams(
                league_code=league_code,
                season=season,
                home_names=home_names,
                away_names=away_names,
            )
            if len(season_candidates) == 1:
                warnings.append("Matched postponed/rescheduled fixture via season+teams")
                return MatchResolution(
                    match=season_candidates[0],
                    method="season_teams",
                    warnings=warnings,
                )
            if len(season_candidates) > 1:
                season_candidates.sort(
                    key=lambda match: abs((match.match_date - kickoff.date()).days)
                )
                warnings.append(
                    "Matched postponed fixture among season candidates "
                    f"id={season_candidates[0].id}"
                )
                return MatchResolution(
                    match=season_candidates[0],
                    method="season_teams_ambiguous",
                    warnings=warnings,
                )

        logger.warning(
            "Unresolved match provider=%s id=%s %s vs %s at %s",
            self.provider,
            provider_match.provider_match_id,
            provider_match.home_team_name,
            provider_match.away_team_name,
            provider_match.kickoff_at,
        )
        return MatchResolution(match=None, method="unresolved", warnings=warnings)

    def ensure_mapping(
        self,
        *,
        entity_type: str,
        internal_entity_id: int,
        external_entity_id: str,
        external_name: str | None = None,
        dry_run: bool = False,
    ) -> None:
        """Upsert an external_entity_mapping row unless dry_run."""
        if dry_run:
            return
        self.mapping_repo.upsert(
            provider=self.provider,
            entity_type=entity_type,
            internal_entity_id=internal_entity_id,
            external_entity_id=str(external_entity_id),
            external_name=external_name,
        )

    def _candidate_team_names(self) -> list[str]:
        """Cached union of team and historical match names."""
        if self._team_name_cache is None:
            team_names = self.team_repo.get_all_names()
            historical_names = self.historical_repo.get_distinct_home_teams()
            away_names = self.historical_repo.get_distinct_away_teams()
            self._team_name_cache = sorted(
                set(team_names) | set(historical_names) | set(away_names)
            )
        return self._team_name_cache

    def _find_by_normalized_name(
        self, name: str, candidates: list[str]
    ) -> str | None:
        """Find a candidate matching the normalized team name."""
        lookup = {normalize_team_name(candidate): candidate for candidate in candidates}
        return lookup.get(normalize_team_name(name))

    def _get_team_by_name(
        self, name: str, league_id: int | None
    ) -> TeamModel | None:
        """Load a team by name, preferring a league-scoped lookup."""
        if league_id is not None:
            team = self.team_repo.get_by_name_and_league(name, league_id)
            if team is not None:
                return team

    def _get_team_by_name_fuzzy(
            self, name: str
    ) -> TeamModel | None:
        return self.team_repo.team_name_wide_search(name)

    def _get_team_by_machine_name(
        self, name: str
    ) -> TeamModel | None:
        """Load a team by name, preferring a league-scoped lookup."""

        team = self.team_repo.get_by_machine_name(sanitize_string(name))
        if team is not None:
            return team

    def _get_team_by_machine_name_fuzzy(
            self, name: str
    ) -> TeamModel | None:
        return self.team_repo.team_likely_name_wide_search(name)

    def _names_for_team(
        self, team: TeamModel | None, provider_name: str, league_id: int
    ) -> list[str]:
        """Collect provider, alias, and internal name variants for matching."""
        names = {provider_name}
        if team is not None:
            names.add(team.name)
            if team.short_name:
                names.add(team.short_name)
            if team.medium_name:
                names.add(team.medium_name)
            mapped = self.team_repo.to_football_data_name(team.name)
            if mapped:
                names.add(mapped)
        alias = self._aliases.get(provider_name)
        if alias:
            names.add(alias)
        # Include normalized historical matches via football-data name matcher.
        fd_name = self.team_repo.to_football_data_name(provider_name)
        if fd_name:
            names.add(fd_name)
        team = self.team_repo.team_name_wide_search(provider_name)
        if team:
            names.add(team.name)
        return [name for name in names if name]
