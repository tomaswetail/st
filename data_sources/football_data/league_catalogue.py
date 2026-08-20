"""Discover provider leagues and map them to internal leagues.id."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from data_sources.football_data.protocol import FootballDataProvider
from data_sources.football_data.service import ProviderName, build_provider
from database import SessionLocal
from objects.repositories.external_entity_mapping_repository import (
    ExternalEntityMappingRepository,
)
from objects.repositories.league_repository import LeagueRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    LeagueMappingResult,
    LeagueMappingSuggestion,
    ProviderLeague,
)
from utils.team_name_matcher import normalize_team_name

logger = logging.getLogger(__name__)


class LeagueCatalogueService:
    """List/search provider league catalogues and map them to internal leagues."""

    def __init__(
        self,
        session: Session | None = None,
        provider: ProviderName | FootballDataProvider = "sofascore",
    ) -> None:
        """Build provider client and DB session for catalogue operations."""
        self.config = DataSourceConfig()
        self._owns_session = session is None
        self.session = session or SessionLocal()

        if isinstance(provider, str):
            self.provider_name: str = provider
            self.provider = build_provider(provider, config=self.config)
        else:
            self.provider = provider
            self.provider_name = provider.name

        self.league_repo = LeagueRepository(self.session)
        self.mapping_repo = ExternalEntityMappingRepository(self.session)
        self._catalogue_cache: list[ProviderLeague] | None = None

    def close(self) -> None:
        """Close owned provider client and session."""
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
        if self._owns_session:
            self.session.close()

    def list_available_leagues(self) -> list[ProviderLeague]:
        """Full provider catalogue (id, name, country)."""
        if self._catalogue_cache is None:
            self._catalogue_cache = self.provider.fetch_available_leagues()
        return list(self._catalogue_cache)

    def find_leagues(
        self,
        query: str | None = None,
        country: str | None = None,
    ) -> list[ProviderLeague]:
        """Filter/rank catalogue by normalized name and optional country."""
        leagues = self.list_available_leagues()
        if country:
            country_norm = normalize_team_name(country)
            leagues = [
                league
                for league in leagues
                if (
                    league.country
                    and normalize_team_name(league.country) == country_norm
                )
                or (
                    league.country_code
                    and league.country_code.lower() == country.strip().lower()
                )
                or (
                    league.country
                    and country_norm in normalize_team_name(league.country)
                )
            ]
        if not query:
            return leagues

        query_norm = normalize_team_name(query)
        scored: list[tuple[float, ProviderLeague]] = []
        for league in leagues:
            name_norm = normalize_team_name(league.name)
            score = self._name_similarity(query_norm, name_norm)
            if score >= 0.5:
                scored.append((score, league))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [league for _, league in scored]

    def suggest_mappings(
        self,
        league_ids: list[int] | None = None,
    ) -> list[LeagueMappingSuggestion]:
        """For each internal LeagueModel, best provider candidate + confidence.

        Never writes to the database.
        """
        if league_ids is not None:
            internal = [
                league
                for league_id in league_ids
                if (league := self.league_repo.get(league_id)) is not None
            ]
        else:
            internal = self.league_repo.get_all()

        catalogue = self.list_available_leagues()
        suggestions: list[LeagueMappingSuggestion] = []
        for league in internal:
            existing = self.mapping_repo.get_by_internal(
                provider=self.provider_name,
                entity_type="league",
                internal_entity_id=league.id,
            )
            if existing is not None:
                candidate = ProviderLeague(
                    provider_league_id=existing.external_entity_id,
                    name=existing.external_name or league.league_name,
                    country=league.country_name,
                )
                suggestions.append(
                    LeagueMappingSuggestion(
                        internal_league_id=league.id,
                        internal_name=league.league_name,
                        candidate=candidate,
                        confidence=1.0,
                        method="mapping",
                    )
                )
                continue

            method, confidence, candidate = self._best_candidate(
                league.league_name, league.country_name, catalogue
            )
            suggestions.append(
                LeagueMappingSuggestion(
                    internal_league_id=league.id,
                    internal_name=league.league_name,
                    candidate=candidate,
                    confidence=confidence,
                    method=method,
                )
            )
        return suggestions

    def map_league(
        self,
        league_id: int,
        *,
        external_entity_id: str | None = None,
        query: str | None = None,
        dry_run: bool = False,
    ) -> LeagueMappingResult:
        """Upsert external_entity_mapping for entity_type=league."""
        league = self.league_repo.get(league_id)
        if league is None:
            return LeagueMappingResult(
                internal_league_id=league_id,
                provider=self.provider_name,
                external_entity_id=None,
                status="failed",
                error=f"Internal league {league_id} not found",
            )

        existing = self.mapping_repo.get_by_internal(
            provider=self.provider_name,
            entity_type="league",
            internal_entity_id=league_id,
        )
        if existing is not None and external_entity_id is None and query is None:
            return LeagueMappingResult(
                internal_league_id=league_id,
                provider=self.provider_name,
                external_entity_id=existing.external_entity_id,
                status="already_mapped",
            )

        chosen: ProviderLeague | None = None
        candidates: list[ProviderLeague] = []

        if external_entity_id is not None:
            catalogue = self.list_available_leagues()
            chosen = next(
                (
                    item
                    for item in catalogue
                    if item.provider_league_id == str(external_entity_id)
                ),
                ProviderLeague(
                    provider_league_id=str(external_entity_id),
                    name=league.league_name,
                    country=league.country_name,
                ),
            )
        else:
            search_query = query or league.league_name
            candidates = self.find_leagues(
                query=search_query, country=league.country_name
            )
            threshold = self.config.fuzzy_match_threshold / 100
            query_norm = normalize_team_name(search_query)
            strong = [
                item
                for item in candidates
                if self._name_similarity(query_norm, normalize_team_name(item.name))
                >= threshold
            ]
            if len(strong) == 1:
                chosen = strong[0]
            elif len(strong) > 1:
                exact = [
                    item
                    for item in strong
                    if normalize_team_name(item.name) == query_norm
                    or query_norm in normalize_team_name(item.name)
                    or normalize_team_name(item.name) in query_norm
                ]
                if len(exact) == 1:
                    chosen = exact[0]
                elif strong:
                    # Prefer highest-ranked find_leagues hit when unique top score.
                    chosen = strong[0]
                    if len(strong) > 1 and normalize_team_name(strong[0].name) != normalize_team_name(strong[1].name):
                        # Keep first if clearly best; otherwise unresolved.
                        top = self._name_similarity(
                            query_norm, normalize_team_name(strong[0].name)
                        )
                        second = self._name_similarity(
                            query_norm, normalize_team_name(strong[1].name)
                        )
                        if top - second < 0.05:
                            chosen = None
                            candidates = strong
                else:
                    candidates = strong
            else:
                candidates = strong or candidates[:5]

        if chosen is None:
            return LeagueMappingResult(
                internal_league_id=league_id,
                provider=self.provider_name,
                external_entity_id=None,
                status="unresolved",
                candidates=candidates,
                error="No unique high-confidence provider league match",
            )

        if dry_run:
            return LeagueMappingResult(
                internal_league_id=league_id,
                provider=self.provider_name,
                external_entity_id=chosen.provider_league_id,
                status="mapped",
                candidates=[chosen],
            )

        try:
            self.mapping_repo.upsert(
                provider=self.provider_name,
                entity_type="league",
                internal_entity_id=league_id,
                external_entity_id=chosen.provider_league_id,
                external_name=chosen.name,
                metadata={
                    "country": chosen.country,
                    "country_code": chosen.country_code,
                },
            )
            self.session.commit()
        except Exception as exc:  # noqa: BLE001
            self.session.rollback()
            logger.exception("Failed mapping league %s", league_id)
            return LeagueMappingResult(
                internal_league_id=league_id,
                provider=self.provider_name,
                external_entity_id=chosen.provider_league_id,
                status="failed",
                candidates=[chosen],
                error=str(exc),
            )

        return LeagueMappingResult(
            internal_league_id=league_id,
            provider=self.provider_name,
            external_entity_id=chosen.provider_league_id,
            status="mapped",
            candidates=[chosen],
        )

    def _best_candidate(
        self,
        internal_name: str,
        country: str | None,
        catalogue: list[ProviderLeague],
    ) -> tuple[str, float, ProviderLeague | None]:
        """Pick the best catalogue match for an internal league name."""
        target = normalize_team_name(internal_name)
        country_norm = normalize_team_name(country) if country else None

        scoped = catalogue
        if country_norm:
            country_matches = [
                league
                for league in catalogue
                if league.country and country_norm in normalize_team_name(league.country)
            ]
            if country_matches:
                scoped = country_matches

        best: ProviderLeague | None = None
        best_score = 0.0
        for league in scoped:
            name_norm = normalize_team_name(league.name)
            score = self._name_similarity(target, name_norm)
            if score > best_score:
                best_score = score
                best = league

        if best is not None and best_score >= 0.99:
            return "exact", best_score, best
        threshold = self.config.fuzzy_match_threshold / 100
        if best is not None and best_score >= threshold:
            return "fuzzy", best_score, best
        return "unresolved", best_score, None

    @staticmethod
    def _name_similarity(left: str, right: str) -> float:
        """Score similarity between two normalized league names."""
        if left == right:
            return 1.0
        if left in right or right in left:
            return 0.96
        # Compare token overlap for names like "England Premier League" vs "Premier League".
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if left_tokens and right_tokens:
            overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if overlap >= 0.5:
                return max(overlap, SequenceMatcher(None, left, right).ratio())
        return SequenceMatcher(None, left, right).ratio()
