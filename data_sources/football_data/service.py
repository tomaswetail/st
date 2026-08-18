"""Public historical football data ingestion service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy.orm import Session

from data_sources.entity_resolver import EntityResolver, TeamResolution
from data_sources.football_data.http_client import (
    FootballDataHttpError,
    NotFoundError,
)
from data_sources.football_data.metrics import (
    calculate_derived_metrics,
    shot_fingerprint,
)
from data_sources.football_data.protocol import FootballDataProvider
from data_sources.football_data.providers.fotmob import FotMobProvider
from data_sources.football_data.providers.sofascore import SofaScoreProvider
from data_sources.football_data.results import BatchImportResult, MatchImportResult
from database import SessionLocal
from objects.models.historical_match import HistoricalMatchModel
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.match_advanced_stats_repository import (
    MatchAdvancedStatsRepository,
)
from objects.repositories.match_shot_repository import MatchShotRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    ProviderMatch,
    ProviderMatchDetails,
)
from utils.seasons import last_n_season_codes

logger = logging.getLogger(__name__)

ProviderName = Literal["fotmob", "sofascore"]

# FotMob season discovery is broken — use fixed season labels only.
FIXED_IMPORT_SEASONS = ["2024/2025", "2025/2026"]


def build_provider(
    name: ProviderName,
    *,
    config: DataSourceConfig,
    client=None,
) -> FootballDataProvider:
    """Construct a FotMob or SofaScore provider adapter."""
    if name == "fotmob":
        return FotMobProvider(client=client, config=config)
    if name == "sofascore":
        return SofaScoreProvider(client=client, config=config)
    raise ValueError(f"Unsupported provider: {name}")


class ExtendedMatchDataService:
    """Fetch and persist historical xG / shot data for existing historical matches."""

    def __init__(
        self,
        provider: ProviderName | FootballDataProvider = "fotmob",
        session: Session | None = None,
        *,
        fallback_provider: ProviderName | FootballDataProvider | None = None,
        config: DataSourceConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        """Wire primary/fallback providers, resolver, and repositories."""
        self.config = DataSourceConfig()
        self.dry_run = dry_run
        self._owns_session = session is None
        self.session = session or SessionLocal()

        if isinstance(provider, str):
            self.provider_name: str = provider
            self.provider = build_provider(provider, config=self.config)
        else:
            self.provider = provider
            self.provider_name = provider.name

        self.fallback_provider: FootballDataProvider | None = None
        self.fallback_provider_name: str | None = None
        if isinstance(fallback_provider, str):
            self.fallback_provider_name = fallback_provider
            self.fallback_provider = build_provider(
                fallback_provider, config=self.config
            )
        elif fallback_provider is not None:
            self.fallback_provider = fallback_provider
            self.fallback_provider_name = fallback_provider.name

        self.resolver = EntityResolver(
            self.session, config=self.config, provider=self.provider_name
        )
        self.historical_repo = HistoricalMatchRepository(self.session)
        self.stats_repo = MatchAdvancedStatsRepository(self.session)
        self.shot_repo = MatchShotRepository(self.session)

    def close(self) -> None:
        """Close owned provider clients and DB session."""
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
        if self.fallback_provider is not None:
            close_fb = getattr(self.fallback_provider, "close", None)
            if callable(close_fb):
                close_fb()
        if self._owns_session:
            self.session.close()

    def fetch_and_store_match(
        self,
        match_id: int,
        force_refresh: bool = False,
    ) -> MatchImportResult:
        """Import xG/shots for one historical match by id."""
        historical = self.historical_repo.get(match_id)
        if historical is None:
            return MatchImportResult(
                internal_match_id=match_id,
                provider_match_id=None,
                status="failed",
                error=f"historical match {match_id} not found",
            )

        if not force_refresh:
            existing = self.stats_repo.get_by_match_and_provider(
                match_id, self.provider_name
            )
            if existing is not None:
                return MatchImportResult(
                    internal_match_id=match_id,
                    provider_match_id=None,
                    status="skipped",
                    warnings=["Advanced stats already present; use force_refresh"],
                )

        provider_match_id = self._provider_match_id_for_internal(match_id)
        try:
            details, used_provider = self._fetch_details_with_fallback(
                provider_match_id, historical
            )
        except Exception as exc:  # noqa: BLE001 — per-match isolation
            logger.exception("Failed fetching match %s", match_id)
            return MatchImportResult(
                internal_match_id=match_id,
                provider_match_id=provider_match_id,
                status="failed",
                error=str(exc),
            )

        if details is None:
            return MatchImportResult(
                internal_match_id=match_id,
                provider_match_id=provider_match_id,
                status="unresolved",
                error="No provider match details available",
            )

        return self._persist_match_details(
            historical=historical,
            details=details,
            provider_name=used_provider,
            force_refresh=force_refresh,
        )

    def fetch_and_store_matches(
        self,
        match_ids: list[int],
        force_refresh: bool = False,
    ) -> BatchImportResult:
        """Import xG/shots for many historical match ids."""
        batch = BatchImportResult(requested=len(match_ids))
        for match_id in match_ids:
            try:
                with self.session.begin_nested():
                    result = self.fetch_and_store_match(
                        match_id, force_refresh=force_refresh
                    )
                if not self.dry_run:
                    self.session.commit()
                batch.add(result)
            except Exception as exc:  # noqa: BLE001
                self.session.rollback()
                batch.add(
                    MatchImportResult(
                        internal_match_id=match_id,
                        provider_match_id=None,
                        status="failed",
                        error=str(exc),
                    )
                )
        return batch

    def fetch_and_store_league_history(
        self,
        league_id: int,
        season: str | None = None,
        force_refresh: bool = False,
        limit: int | None = None,
        min_season_year: int | None = None,
    ) -> BatchImportResult:
        """Import fixtures for fixed seasons of a mapped league."""
        league = self.resolver.resolve_league(league_id)
        if league is None:
            return BatchImportResult(requested=0)

        provider_league_id = self.resolver.resolve_provider_league_id(league_id)
        if provider_league_id is None:
            logger.error(
                "Cannot import league_id=%s without external mapping for %s",
                league_id,
                self.provider_name,
            )
            return BatchImportResult(requested=0)

        league_code = self.resolver.football_data_league_code(league)
        country_code = self._resolve_provider_country_code(
            league_id, provider_league_id
        )
        if self.provider_name == "fotmob" and not country_code:
            logger.error(
                "Cannot import league_id=%s: missing ccode3 for FotMob id=%s",
                league_id,
                provider_league_id,
            )
            return BatchImportResult(requested=0)

        seasons = [season] if season else list(FIXED_IMPORT_SEASONS)
        logger.info(
            "league_id=%s provider=%s seasons=%s ccode3=%s",
            league_id,
            self.provider_name,
            seasons,
            country_code,
        )
        batch = BatchImportResult(requested=0)
        imported_count = 0
        min_kickoff = (
            datetime(min_season_year, 1, 1, tzinfo=timezone.utc)
            if min_season_year is not None
            else None
        )

        for season_id in seasons:
            try:
                fixtures = self.provider.fetch_season_matches(
                    provider_league_id,
                    season_id,
                    country_code=country_code,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed fetching season %s for league %s: %s",
                    season_id,
                    provider_league_id,
                    exc,
                )
                continue

            for fixture in fixtures:
                if min_kickoff is not None and fixture.kickoff_at < min_kickoff:
                    continue
                if limit is not None and imported_count >= limit:
                    return batch
                batch.requested += 1
                try:
                    with self.session.begin_nested():
                        result = self._import_provider_fixture(
                            fixture=fixture,
                            league_id=league_id,
                            league_code=league_code,
                            season=season or self._map_season_label(season_id),
                            force_refresh=force_refresh,
                        )
                    if not self.dry_run:
                        self.session.commit()
                    batch.add(result)
                    if result.status in {"imported", "updated"}:
                        imported_count += 1
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    s = trace = traceback.format_exc()
                    logger.exception(
                        "Failed fetching season %s for league %s: %s",
                        season_id,
                        provider_league_id,
                        exc,
                    )
                    self.session.rollback()
                    batch.add(
                        MatchImportResult(
                            internal_match_id=0,
                            provider_match_id=fixture.provider_match_id,
                            status="failed",
                            error=str(exc),
                        )
                    )
        return batch

    def _import_provider_fixture(
        self,
        *,
        fixture: ProviderMatch,
        league_id: int,
        league_code: str | None,
        season: str | None,
        force_refresh: bool,
    ) -> MatchImportResult:

        if fixture.provider_match_id == '4506273':
            p=1
        """Resolve teams/match and persist details for one fixture."""
        home = self.resolver.resolve_team(
            provider_team_id=fixture.home_team_id,
            provider_team_name=fixture.home_team_name,
            league_id=league_id,
        )

        away = self.resolver.resolve_team(
            provider_team_id=fixture.away_team_id,
            provider_team_name=fixture.away_team_name,
            league_id=league_id,
        )
        warnings: list[str] = []
        if home.team is None:
            home = self._create_team_from_fotmob(
                fixture.home_team_id, league_id
            ) or home
            if home.team is None:
                warnings.append(f"Unresolved home team: {fixture.home_team_name}")
        if away.team is None:
            away = self._create_team_from_fotmob(
                fixture.away_team_id, league_id
            ) or away
            if away.team is None:
                warnings.append(f"Unresolved away team: {fixture.away_team_name}")

        match_resolution = self.resolver.resolve_match(
            fixture,
            league_code=league_code,
            league_id=league_id,
            home_team=home.team,
            away_team=away.team,
            season=season,
        )
        warnings.extend(match_resolution.warnings)

        if match_resolution.match is None:
            if self.config.create_missing_historical_matches and not self.dry_run:
                # Intentionally disabled by default; reserved for future use.
                pass
            return MatchImportResult(
                internal_match_id=0,
                provider_match_id=fixture.provider_match_id,
                status="unresolved",
                warnings=warnings,
                error="No matching historical match",
            )

        historical = match_resolution.match
        if home.team is not None:
            self.resolver.ensure_mapping(
                entity_type="team",
                internal_entity_id=home.team.id,
                external_entity_id=fixture.home_team_id,
                external_name=fixture.home_team_name,
                dry_run=self.dry_run,
            )
        if away.team is not None:
            self.resolver.ensure_mapping(
                entity_type="team",
                internal_entity_id=away.team.id,
                external_entity_id=fixture.away_team_id,
                external_name=fixture.away_team_name,
                dry_run=self.dry_run,
            )
        self.resolver.ensure_mapping(
            entity_type="match",
            internal_entity_id=historical.id,
            external_entity_id=fixture.provider_match_id,
            external_name=f"{fixture.home_team_name} vs {fixture.away_team_name}",
            dry_run=self.dry_run,
        )

        try:
            details, used_provider = self._fetch_details_with_fallback(
                fixture.provider_match_id, historical, seed_match=fixture
            )
        except Exception as exc:  # noqa: BLE001
            return MatchImportResult(
                internal_match_id=historical.id,
                provider_match_id=fixture.provider_match_id,
                status="failed",
                warnings=warnings,
                error=str(exc),
            )

        if details is None:
            return MatchImportResult(
                internal_match_id=historical.id,
                provider_match_id=fixture.provider_match_id,
                status="unresolved",
                warnings=warnings,
                error="Provider details unavailable",
            )

        result = self._persist_match_details(
            historical=historical,
            details=details,
            provider_name=used_provider,
            force_refresh=force_refresh,
            home_team_id=home.team.id if home.team else None,
            away_team_id=away.team.id if away.team else None,
        )
        result.warnings = warnings + result.warnings
        return result

    def _persist_match_details(
        self,
        *,
        historical: HistoricalMatchModel,
        details: ProviderMatchDetails,
        provider_name: str,
        force_refresh: bool,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
    ) -> MatchImportResult:
        """Upsert advanced stats and shots for a historical match."""
        warnings: list[str] = []
        existing = self.stats_repo.get_by_match_and_provider(
            historical.id, provider_name
        )
        if existing is not None and not force_refresh:
            return MatchImportResult(
                internal_match_id=historical.id,
                provider_match_id=details.match.provider_match_id,
                status="skipped",
                warnings=["Stats already present"],
            )

        metrics = calculate_derived_metrics(
            details,
            home_team_external_id=details.match.home_team_id,
            away_team_external_id=details.match.away_team_id,
            xg_aggregate_tolerance=self.config.xg_aggregate_tolerance,
        )
        warnings.extend(metrics.warnings)

        if self.dry_run:
            return MatchImportResult(
                internal_match_id=historical.id,
                provider_match_id=details.match.provider_match_id,
                status="imported" if existing is None else "updated",
                shots_imported=len(details.shots),
                warnings=warnings + ["dry_run"],
            )

        self.resolver.provider = provider_name
        self.resolver.ensure_mapping(
            entity_type="match",
            internal_entity_id=historical.id,
            external_entity_id=details.match.provider_match_id,
            external_name=(
                f"{details.match.home_team_name} vs {details.match.away_team_name}"
            ),
            dry_run=False,
        )

        team_id_by_external = {
            details.match.home_team_id: home_team_id,
            details.match.away_team_id: away_team_id,
        }
        shot_rows = []
        for shot in details.shots:
            team_internal = team_id_by_external.get(shot.team_id)
            fingerprint = shot_fingerprint(
                match_id=historical.id,
                provider=provider_name,
                shot=shot,
                team_internal_id=team_internal,
            )
            shot_rows.append(
                {
                    "provider_shot_id": shot.provider_shot_id,
                    "shot_fingerprint": fingerprint,
                    "team_id": team_internal,
                    "player_external_id": shot.player_id,
                    "minute": shot.minute,
                    "second": shot.second,
                    "xg": shot.xg,
                    "xgot": shot.xgot,
                    "outcome": shot.outcome,
                    "situation": shot.situation,
                    "body_part": shot.body_part,
                    "shot_type": shot.shot_type,
                    "is_penalty": shot.is_penalty,
                    "is_own_goal": shot.is_own_goal,
                    "coordinates": shot.coordinates,
                    "raw_payload": shot.raw_payload,
                }
            )

        self.stats_repo.upsert(
            match_id=historical.id,
            provider=provider_name,
            fields={
                "home_xg": metrics.home_xg,
                "away_xg": metrics.away_xg,
                "home_non_penalty_xg": metrics.home_non_penalty_xg,
                "away_non_penalty_xg": metrics.away_non_penalty_xg,
                "home_xgot": metrics.home_xgot,
                "away_xgot": metrics.away_xgot,
                "home_shots": metrics.home_shots,
                "away_shots": metrics.away_shots,
                "home_shots_on_target": metrics.home_shots_on_target,
                "away_shots_on_target": metrics.away_shots_on_target,
                "home_set_piece_xg": metrics.home_set_piece_xg,
                "away_set_piece_xg": metrics.away_set_piece_xg,
                "home_open_play_xg": metrics.home_open_play_xg,
                "away_open_play_xg": metrics.away_open_play_xg,
                "home_xg_from_shots": metrics.home_xg_from_shots,
                "away_xg_from_shots": metrics.away_xg_from_shots,
                "average_home_shot_xg": metrics.average_home_shot_xg,
                "average_away_shot_xg": metrics.average_away_shot_xg,
                "fetched_at": datetime.now(tz=timezone.utc),
                "raw_payload_hash": metrics.raw_payload_hash,
                "raw_payload": details.raw_payload,
            },
        )
        shots_imported = self.shot_repo.upsert_many(
            match_id=historical.id,
            provider=provider_name,
            shots=shot_rows,
        )
        return MatchImportResult(
            internal_match_id=historical.id,
            provider_match_id=details.match.provider_match_id,
            status="updated" if existing is not None else "imported",
            shots_imported=shots_imported,
            warnings=warnings,
        )

    def _fetch_details_with_fallback(
        self,
        provider_match_id: str | None,
        historical: HistoricalMatchModel,
        seed_match: ProviderMatch | None = None,
    ) -> tuple[ProviderMatchDetails | None, str]:
        """Fetch match details, falling back to the secondary provider if needed."""
        details: ProviderMatchDetails | None = None
        used = self.provider_name
        primary_error: str | None = None

        if provider_match_id:
            try:
                details = self.provider.fetch_match_details(provider_match_id)
            except NotFoundError as exc:
                primary_error = str(exc)
            except FootballDataHttpError as exc:
                primary_error = str(exc)
                if not exc.retryable and self.fallback_provider is None:
                    raise

        needs_fallback = (
            details is None
            or not details.shots
            or (details.home_xg is None and details.away_xg is None and not details.shots)
        )
        if needs_fallback and self.fallback_provider is not None:
            fallback_id = self._provider_match_id_for_internal(
                historical.id, provider=self.fallback_provider_name or "sofascore"
            )
            if fallback_id:
                try:
                    fb_details = self.fallback_provider.fetch_match_details(fallback_id)
                    if details is None:
                        details = fb_details
                        used = self.fallback_provider_name or self.fallback_provider.name
                    elif (not details.shots and fb_details.shots) or (
                        details.home_xg is None
                        and details.away_xg is None
                        and (fb_details.home_xg is not None or fb_details.shots)
                    ):
                        # Do not merge silently — replace whole record with fallback.
                        details = fb_details
                        used = self.fallback_provider_name or self.fallback_provider.name
                        if primary_error:
                            logger.warning(
                                "Using fallback provider %s after primary failure: %s",
                                used,
                                primary_error,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Fallback provider failed: %s", exc)

        if details is None and seed_match is not None and provider_match_id:
            # Minimal shell when only fixture list data exists.
            details = ProviderMatchDetails(
                match=seed_match,
                shots=[],
                statistics={},
                raw_payload=seed_match.raw_payload,
            )
        return details, used

    def _create_team_from_fotmob(
        self,
        provider_team_id: str,
        league_id: int,
    ) -> TeamResolution | None:
        """Fetch FotMob team profile and create an internal team row."""
        if self.dry_run or self.provider_name != "fotmob":
            return None
        fetch_team = getattr(self.provider, "fetch_team", None)
        if not callable(fetch_team):
            return None
        try:
            provider_team = fetch_team(provider_team_id)
            team = self.resolver.team_repo.create_from_provider_team(
                name=provider_team.name,
                league_id=league_id,
                short_name=provider_team.short_name,
                country_name=provider_team.country_name,
                iso_code=provider_team.country_code,
            )
            return TeamResolution(team=team, confidence=1.0, method="created")
        except Exception as exc:  # noqa: BLE001 — keep fixture import going
            logger.warning(
                "Failed creating team from FotMob id=%s: %s",
                provider_team_id,
                exc,
            )
            return TeamResolution(
                team=None,
                confidence=0.0,
                method="unresolved",
                unresolved_name=provider_team_id,
            )

    def _provider_match_id_for_internal(
        self,
        match_id: int,
        provider: str | None = None,
    ) -> str | None:
        """Look up a stored provider match id for an internal match."""
        mapping = self.resolver.mapping_repo.get_by_internal(
            provider=provider or self.provider_name,
            entity_type="match",
            internal_entity_id=match_id,
        )
        return mapping.external_entity_id if mapping else None

    def _resolve_provider_country_code(
        self,
        league_id: int,
        provider_league_id: str,
    ) -> str | None:
        """Resolve FotMob ccode3 from mapping metadata or CSV."""
        mapping = self.resolver.mapping_repo.get_by_internal(
            provider=self.provider_name,
            entity_type="league",
            internal_entity_id=league_id,
        )
        if mapping is not None and mapping.metadata_json:
            ccode = mapping.metadata_json.get("ccode")
            if ccode:
                return str(ccode)
        return _ccode_from_all_leagues_csv(provider_league_id)

    @staticmethod
    def _map_season_label(provider_season_name: str) -> str | None:
        """Normalize a provider season label to an internal season code."""
        text = provider_season_name.strip()
        if len(text) == 4 and text.isdigit():
            return text
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 4:
            return digits[:4]
        codes = last_n_season_codes(1)
        return codes[0] if codes else None


_CSV_CCODE_CACHE: dict[str, str] | None = None


def _ccode_from_all_leagues_csv(provider_league_id: str) -> str | None:
    """Look up FotMob ccode3 for a league id in all_leagues.csv."""
    global _CSV_CCODE_CACHE
    if _CSV_CCODE_CACHE is None:
        import csv
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "all_leagues.csv"
        cache: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="latin-1", newline="") as handle:
                for row in csv.DictReader(handle):
                    cache[str(row["id"])] = str(row["ccode"])
        _CSV_CCODE_CACHE = cache
    return _CSV_CCODE_CACHE.get(str(provider_league_id))
