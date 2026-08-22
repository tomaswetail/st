"""Resolve provider entities to internal leagues, teams, and historical matches."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy.orm import Session

from objects.models.fixture import FixtureModel
from objects.models.league import LeagueModel
from objects.models.team import TeamModel
from objects.repositories.external_entity_mapping_repository import (
    ExternalEntityMappingRepository,
)
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import ProviderMatch
from utils.common import FOTMOB_TO_API_FOOTBALL_TEAM_MAPPING
from utils.team_name_matcher import _load_aliases, normalize_team_name

# Temporary alias for call sites / type hints still using the old name.

logger = logging.getLogger(__name__)


def _as_fixture_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


UNRESOLVED_MATCH_CSV_FIELDS = (
    "provider",
    "provider_match_id",
    "provider_league_id",
    "provider_season_id",
    "league_external_id",
    "league_id",
    "season",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "internal_home_team",
    "internal_away_team",
    "kickoff_at",
    "home_score",
    "away_score",
)


@dataclass
class TeamResolution:
    """Result of resolving a provider team to an internal team."""
    team: TeamModel | None
    confidence: float
    method: str
    unresolved_name: str | None = None


@dataclass
class MatchResolution:
    """Result of resolving a provider fixture to a fixture."""
    match: FixtureModel | None
    method: str
    warnings: list[str]


class EntityResolver:
    """Map provider leagues, teams, and matches onto internal entities."""
    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        provider: str = "api-football",
    ) -> None:
        """Wire repositories and team-name aliases for a provider."""
        self.session = session
        self.config = config or DataSourceConfig()
        self.provider = provider
        self.mapping_repo = ExternalEntityMappingRepository(session)
        self.team_repo = TeamRepository(session)
        self.league_repo = LeagueRepository(session)
        self.fixture_repo = FixtureRepository(session)
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

    def resolve_team(
        self,
        *,
        provider_team_id: str,
        provider_team_name: str,
        league_id: int | None = None,
        create_if_missing: bool = False,
    ) -> TeamResolution:
        """Resolve a provider team via mapping, exact name, alias, fuzzy, or Postgres duplicate lookup.

        When ``create_if_missing`` is True (historical upsert), create a TeamModel
        on miss and store an external mapping for reimports.
        """

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
            spellings_differ = (
                provider_team_name.strip().lower() != exact.strip().lower()
            )
            if not spellings_differ or self._is_safe_team_match(
                provider_team_name, exact
            ):
                team = self._get_team_by_name(exact, league_id)
                if team is not None:
                    if create_if_missing:
                        self._map_resolved_team(team, provider_team_id, provider_team_name)
                    return TeamResolution(team=team, confidence=1.0, method="exact_name")

        alias = self._aliases.get(provider_team_name)
        if alias:
            team = self._get_team_by_name(alias, league_id)
            if team is not None and self._is_safe_team_match(
                provider_team_name, team.name
            ):
                if create_if_missing:
                    self._map_resolved_team(team, provider_team_id, provider_team_name)
                return TeamResolution(team=team, confidence=0.98, method="alias")
            aliased_exact = self._find_by_normalized_name(alias, candidates)
            if aliased_exact is not None and self._is_safe_team_match(
                provider_team_name, aliased_exact
            ):
                team = self._get_team_by_name(aliased_exact, league_id)
                if team is not None and self._is_safe_team_match(
                    provider_team_name, team.name
                ):
                    if create_if_missing:
                        self._map_resolved_team(team, provider_team_id, provider_team_name)
                    return TeamResolution(team=team, confidence=0.98, method="alias")

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
            if self._is_safe_team_match(provider_team_name, best_name):
                team = self._get_team_by_name(best_name, league_id)
                if team is not None:
                    if create_if_missing:
                        self._map_resolved_team(team, provider_team_id, provider_team_name)
                    return TeamResolution(
                        team=team,
                        confidence=best_score,
                        method="fuzzy",
                    )

        postgres_lookups = (
            (self.team_repo.find_exact_normalized, "normalized", 1.0),
            (self.team_repo.find_by_club_affix, "club_affix", 0.95),
            (self.team_repo.find_fuzzy_duplicate, "pg_fuzzy", 0.80),
            (self.team_repo.find_substring_duplicate, "substring", 0.90),
        )
        for lookup, method, confidence in postgres_lookups:
            team = lookup(provider_team_name)
            if team is not None:
                if not self._is_safe_team_match(provider_team_name, team.name):
                    continue
                if create_if_missing:
                    self._map_resolved_team(team, provider_team_id, provider_team_name)
                return TeamResolution(
                    team=team, confidence=confidence, method=method
                )

        if create_if_missing:
            try:
                external_id = int(provider_team_id)
            except (TypeError, ValueError):
                logger.warning(
                    "Cannot create team without numeric external_id (got %r)",
                    provider_team_id,
                )
                return TeamResolution(
                    team=None,
                    confidence=best_score,
                    method="unresolved",
                    unresolved_name=provider_team_name,
                )
            team = self.team_repo.create_from_provider_team(
                external_id=external_id,
                name=provider_team_name,
            )
            self.team_repo.flush()
            self._team_name_cache = None
            self._map_resolved_team(team, provider_team_id, provider_team_name)
            return TeamResolution(team=team, confidence=1.0, method="created")

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

    def _is_safe_team_match(self, provider_team_name: str, candidate_name: str) -> bool:
        """Block known false positives in non-exact matching paths."""
        blocked_pairs = {
            ("angers", "rangers"),
            ("villarreal", "villarreal b"),
            ("aris", "paris fc"),
            ("manchester city", "manchester united"),
            ("manchester city", "man united"),
            ("bristol rovers", "bristol"),
            ("bristol city", "bristol rovers"),
            ("oxford united", "oxford city"),
            ("oxford city", "oxford"),
            ("new york red bulls", "york"),
            ("new york city", "york"),
            ("arsenal", "arsenal sarandi"),
            ("inter", "inter turku"),
            ("lille", "lillestrom"),
            ("atalanta", "atlanta utd"),
        }
        provider_norm = normalize_team_name(provider_team_name)
        candidate_norm = normalize_team_name(candidate_name)
        provider_raw = provider_team_name.strip().lower()
        candidate_raw = candidate_name.strip().lower()
        return (
            (provider_norm, candidate_norm) not in blocked_pairs
            and (provider_raw, candidate_raw) not in blocked_pairs
        )

    def _map_resolved_team(
        self,
        team: TeamModel,
        provider_team_id: str,
        provider_team_name: str,
    ) -> None:
        self.ensure_mapping(
            entity_type="team",
            internal_entity_id=team.id,
            external_entity_id=str(provider_team_id),
            external_name=provider_team_name,
        )

    def resolve_match(
        self,
        provider_match: ProviderMatch,
        *,
        league_external_id: int | None,
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
            match = self.fixture_repo.get(mapping.internal_entity_id)
            if match is not None:
                return MatchResolution(match=match, method="mapping", warnings=warnings)

        home_team_ids = (
            [home_team.external_id] if home_team is not None else None
        )
        away_team_ids = (
            [away_team.external_id] if away_team is not None else None
        )
        home_names = self._names_for_team(
            home_team, provider_match.home_team_name, league_id
        )
        away_names = self._names_for_team(
            away_team, provider_match.away_team_name, league_id
        )
        kickoff = provider_match.kickoff_at
        tolerance = timedelta(minutes=self.config.kickoff_match_tolerance_minutes)
        date_from = (kickoff - tolerance).date()
        date_to = (kickoff + tolerance).date()

        candidates = self.fixture_repo.find_by_date_range_and_teams(
            date_from=date_from,
            date_to=date_to,
            home_team_ids=home_team_ids,
            away_team_ids=away_team_ids,
            home_names=None if home_team_ids else home_names,
            away_names=None if away_team_ids else away_names,
            league_external_id=league_external_id,
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
                key=lambda match: abs(
                    (_as_fixture_date(match.fixture_date) - kickoff.date()).days
                )
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
        if season and league_external_id is not None and (
            home_team_ids or home_names
        ) and (away_team_ids or away_names):
            season_candidates = self.fixture_repo.find_by_season_and_teams(
                league_external_id=league_external_id,
                season=season,
                home_team_ids=home_team_ids,
                away_team_ids=away_team_ids,
                home_names=None if home_team_ids else home_names,
                away_names=None if away_team_ids else away_names,
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
                    key=lambda match: abs(
                        (_as_fixture_date(match.fixture_date) - kickoff.date()).days
                    )
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
            "Unresolved match provider=%s id=%s %s vs %s league=%s at %s",
            self.provider,
            provider_match.provider_match_id,
            provider_match.home_team_name,
            provider_match.away_team_name,
            provider_match.provider_league_id,
            provider_match.kickoff_at,
        )
        self._append_unresolved_match(
            provider_match,
            league_external_id=league_external_id,
            league_id=league_id,
            home_team=home_team,
            away_team=away_team,
            season=season,
        )
        return MatchResolution(match=None, method="unresolved", warnings=warnings)

    def _append_unresolved_match(
        self,
        provider_match: ProviderMatch,
        *,
        league_external_id: int | None,
        league_id: int | None,
        home_team: TeamModel | None,
        away_team: TeamModel | None,
        season: str | None,
    ) -> None:
        """Append an unresolved fixture to CSV, creating the file with a header if needed."""
        csv_path = Path(self.config.unresolved_matches_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        row = {
            "provider": self.provider,
            "provider_match_id": provider_match.provider_match_id,
            "provider_league_id": provider_match.provider_league_id,
            "provider_season_id": provider_match.provider_season_id or "",
            "league_external_id": (
                league_external_id if league_external_id is not None else ""
            ),
            "league_id": league_id if league_id is not None else "",
            "season": season or "",
            "home_team_id": provider_match.home_team_id,
            "home_team_name": provider_match.home_team_name,
            "away_team_id": provider_match.away_team_id,
            "away_team_name": provider_match.away_team_name,
            "internal_home_team": home_team.name if home_team is not None else "",
            "internal_away_team": away_team.name if away_team is not None else "",
            "kickoff_at": provider_match.kickoff_at.isoformat(),
            "home_score": provider_match.home_score if provider_match.home_score is not None else "",
            "away_score": provider_match.away_score if provider_match.away_score is not None else "",
        }
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNRESOLVED_MATCH_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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
            historical_names = self.fixture_repo.get_distinct_home_teams()
            away_names = self.fixture_repo.get_distinct_away_teams()
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
        """Load a team by name (league_id ignored; teams are global)."""
        del league_id
        return self.team_repo.get_by_name(name)

    def _get_team_by_name_fuzzy(self, name: str) -> TeamModel | None:
        return self.team_repo.team_name_wide_search(name)

    def _names_for_team(
        self, team: TeamModel | None, provider_name: str, league_id: int | None
    ) -> list[str]:
        """Collect provider, alias, and internal name variants for matching."""
        del league_id
        names = {provider_name}
        if team is not None:
            names.add(team.name)
            if team.code:
                names.add(team.code)
            mapped = self.team_repo.to_football_data_name(team.name)
            if mapped:
                names.add(mapped)
        alias = self._aliases.get(provider_name)
        if alias:
            names.add(alias)
        fd_name = self.team_repo.to_football_data_name(provider_name)
        if fd_name:
            names.add(fd_name)
        wide = self.team_repo.team_name_wide_search(provider_name)
        if wide:
            names.add(wide.name)
        return [name for name in names if name]
