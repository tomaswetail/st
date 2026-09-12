"""Ingest historical 1X2 odds from football-data.co.uk onto existing fixtures."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from sqlalchemy.orm import Session

from src.data_sources.entity_resolver import EntityResolver
from src.data_sources.football_data.http_client import (
    FootballDataHttpError,
    NotFoundError,
    ThrottledHttpClient,
)
from src.data_sources.football_data_odds.constants import (
    DEFAULT_DATE_FROM,
    DEFAULT_DATE_TO,
    DEFAULT_SEASONS,
    EXTRA_BASE_PATH,
    EXTRA_CODES,
    ENGLISH_TIER_CODES,
    MAIN_BASE_PATH,
    PROVIDER,
)
from src.data_sources.football_data_odds.http import make_odds_http_client
from src.data_sources.football_data_odds.leagues import (
    load_covered_leagues,
    resolve_league_codes,
)
from src.data_sources.football_data_odds.parser import (
    ParsedMatchOdds,
    odds_payload,
    parse_odds_csv,
)
from src.data_sources.football_data_odds.team_names import canonical_team_name
from src.data_sources.football_data_odds.unresolved import (
    UnresolvedOddsRow,
    append_unresolved_odds_row,
    load_unresolved_odds_rows,
    write_unresolved_odds_rows,
)
from src.objects.models.fixture import FixtureModel
from src.objects.models.team import TeamModel
from src.objects.repositories.fixture_odds_repository import FixtureOddsRepository
from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.provider_dtos import ProviderMatch

logger = logging.getLogger(__name__)


@dataclass
class LeagueSeasonStats:
    source: str
    season: str
    csv_rows: int = 0
    resolved: int = 0
    unresolved: int = 0
    upserted: int = 0
    skipped: int = 0
    fetch_failed: bool = False


@dataclass
class IngestTotals:
    summaries: list[LeagueSeasonStats] = field(default_factory=list)

    @property
    def csv_rows(self) -> int:
        return sum(item.csv_rows for item in self.summaries)

    @property
    def resolved(self) -> int:
        return sum(item.resolved for item in self.summaries)

    @property
    def unresolved(self) -> int:
        return sum(item.unresolved for item in self.summaries)

    @property
    def upserted(self) -> int:
        return sum(item.upserted for item in self.summaries)

    @property
    def skipped(self) -> int:
        return sum(item.skipped for item in self.summaries)


@dataclass
class RetryTotals:
    attempted: int = 0
    newly_resolved: int = 0
    upserted: int = 0
    skipped: int = 0
    still_unresolved: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    by_league: dict[str, int] = field(default_factory=dict)


class FootballDataOddsIngestService:
    """Fetch CSVs, resolve onto fixtures, upsert complete 1X2 triples."""

    def __init__(
        self,
        session: Session,
        *,
        config: DataSourceConfig | None = None,
        http_client: ThrottledHttpClient | None = None,
        resolver: EntityResolver | None = None,
        odds_repo: FixtureOddsRepository | None = None,
        fixture_repo: FixtureRepository | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.http = http_client or make_odds_http_client(self.config)
        self.resolver = resolver or EntityResolver(
            session, config=self.config, provider=PROVIDER
        )
        self.odds_repo = odds_repo or FixtureOddsRepository(session)
        self.fixture_repo = fixture_repo or FixtureRepository(session)
        self._owns_http = http_client is None
        self._league_map = load_covered_leagues(self.config.api_football_leagues_path)
        if resolver is None:
            self.resolver._append_unresolved_match = _ignore_resolver_unresolved  # type: ignore[method-assign]

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def ingest(
        self,
        *,
        league: str | None = None,
        season: str | None = None,
        dry_run: bool = False,
        force: bool = False,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> IngestTotals:
        window_from = date_from or DEFAULT_DATE_FROM
        window_to = date_to or DEFAULT_DATE_TO
        codes = resolve_league_codes(
            league, leagues_path=self.config.api_football_leagues_path
        )
        seasons = (season,) if season else DEFAULT_SEASONS
        totals = IngestTotals()
        for code in codes:
            if code in EXTRA_CODES:
                stats = self._ingest_extra(
                    code,
                    dry_run=dry_run,
                    force=force,
                    date_from=window_from,
                    date_to=window_to,
                )
                totals.summaries.append(stats)
                if not dry_run:
                    self.session.commit()
                continue
            for season_code in seasons:
                stats = self._ingest_main(
                    code,
                    season_code,
                    dry_run=dry_run,
                    force=force,
                    date_from=window_from,
                    date_to=window_to,
                )
                totals.summaries.append(stats)
                if not dry_run:
                    self.session.commit()
        return totals

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        season: str,
        closing_only: bool = False,
        dry_run: bool = False,
        force: bool = False,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> LeagueSeasonStats:
        """Parse already-fetched CSV text and resolve/upsert. Used by tests."""
        matches = parse_odds_csv(
            text,
            source=source,
            season=season,
            closing_only=closing_only,
            date_from=date_from,
            date_to=date_to,
        )
        return self._ingest_parsed(
            matches,
            source=source,
            season=season,
            dry_run=dry_run,
            force=force,
        )

    def retry_unresolved(
        self,
        *,
        input_path: Path,
        remaining_path: Path,
        dry_run: bool = False,
        force: bool = False,
        league: str | None = None,
        rewrite_input: bool = False,
    ) -> RetryTotals:
        """Retry odds ingest only for matches listed in the Phase 1 miss log."""
        listed = _unique_listed_rows(
            load_unresolved_odds_rows(input_path),
            league=league,
            leagues_path=self.config.api_football_leagues_path,
        )
        listed_by_key = {row.match_key: row for row in listed}
        found_keys: set[tuple[str, str, str, str]] = set()
        unresolved_reasons: dict[tuple[str, str, str, str], str] = {}
        totals = RetryTotals(attempted=len(listed))

        def on_unresolved(match: ParsedMatchOdds, reason: str) -> None:
            unresolved_reasons[_parsed_match_key(match)] = reason

        for source, season, closing_only in _csv_groups_for_listed(listed):
            path = _odds_csv_path(source, season)
            text = self._fetch_csv(path, required=True)
            if text is None:
                raise FootballDataHttpError(
                    f"Odds CSV fetch returned no body: {path}",
                    retryable=False,
                )
            parsed = parse_odds_csv(
                text,
                source=source,
                season=season,
                closing_only=closing_only,
            )
            selected: list[ParsedMatchOdds] = []
            seen_keys: set[tuple[str, str, str, str]] = set()
            for match in parsed:
                key = _parsed_match_key(match)
                if key not in listed_by_key or key in seen_keys:
                    continue
                seen_keys.add(key)
                found_keys.add(key)
                selected.append(match)
            stats = self._ingest_parsed(
                selected,
                source=source,
                season=season,
                dry_run=dry_run,
                force=force,
                on_unresolved=on_unresolved,
            )
            totals.newly_resolved += stats.resolved
            totals.upserted += stats.upserted
            totals.skipped += stats.skipped

        remaining: list[UnresolvedOddsRow] = []
        for row in listed:
            if row.match_key not in found_keys:
                remaining.append(replace(row, reason="not_in_source_csv"))
            elif row.match_key in unresolved_reasons:
                remaining.append(
                    replace(row, reason=unresolved_reasons[row.match_key])
                )
        totals.still_unresolved = len(remaining)
        totals.by_reason = dict(Counter(item.reason for item in remaining))
        totals.by_league = dict(Counter(item.league_code for item in remaining))
        write_unresolved_odds_rows(remaining_path, remaining)
        if rewrite_input:
            write_unresolved_odds_rows(input_path, remaining)
        if not dry_run:
            self.session.commit()
        return totals

    def _ingest_main(
        self,
        code: str,
        season: str,
        *,
        dry_run: bool,
        force: bool,
        date_from: date,
        date_to: date,
    ) -> LeagueSeasonStats:
        path = f"{MAIN_BASE_PATH}/{season}/{code}.csv"
        text = self._fetch_csv(path)
        if text is None:
            return LeagueSeasonStats(source=code, season=season, fetch_failed=True)
        return self.ingest_text(
            text,
            source=code,
            season=season,
            closing_only=False,
            dry_run=dry_run,
            force=force,
            date_from=date_from,
            date_to=date_to,
        )

    def _ingest_extra(
        self,
        code: str,
        *,
        dry_run: bool,
        force: bool,
        date_from: date,
        date_to: date,
    ) -> LeagueSeasonStats:
        path = f"{EXTRA_BASE_PATH}/{code}.csv"
        text = self._fetch_csv(path)
        if text is None:
            return LeagueSeasonStats(source=code, season="all", fetch_failed=True)
        return self.ingest_text(
            text,
            source=code,
            season="all",
            closing_only=True,
            dry_run=dry_run,
            force=force,
            date_from=date_from,
            date_to=date_to,
        )

    def _fetch_csv(self, path: str, *, required: bool = False) -> str | None:
        try:
            return self.http.get_text(path)
        except NotFoundError:
            if required:
                raise
            logger.warning("Odds CSV not found: %s", path)
            return None
        except FootballDataHttpError as exc:
            if required:
                raise
            logger.warning("Odds CSV fetch failed for %s: %s", path, exc)
            return None

    def _ingest_parsed(
        self,
        matches: list[ParsedMatchOdds],
        *,
        source: str,
        season: str,
        dry_run: bool,
        force: bool,
        on_unresolved: Callable[[ParsedMatchOdds, str], None] | None = None,
    ) -> LeagueSeasonStats:
        stats = LeagueSeasonStats(source=source, season=season, csv_rows=len(matches))
        league_external_id = self._league_external_id(source)
        resolutions: list[tuple[ParsedMatchOdds, FixtureModel | None, str]] = []
        for match in matches:

            if match.kickoff_at.date() < date(2022, 6, 30):
                continue
            elif match.source == 'FIN' and match.kickoff_at.date() < date(2022, 12, 30):
                continue
            elif match.source == 'FIN' and match.kickoff_at.date() > date(2026, 4, 3):
                continue
            elif match.source == 'SWE' and match.kickoff_at.date() > date(2026, 4, 3):
                continue
            elif match.source == 'NOR' and match.kickoff_at.date() > date(2026, 3, 13):
                continue
            fixture, reason = self._resolve_fixture(
                match, league_external_id=league_external_id
            )
            resolutions.append((match, fixture, reason))

        resolved_ids = [
            fixture.id for _, fixture, _ in resolutions if fixture is not None
        ]
        existing: set[int] = set()
        if not force:
            existing = self.odds_repo.fixture_ids_with_odds(
                resolved_ids, provider=PROVIDER
            )

        for match, fixture, reason in resolutions:
            if fixture is None:
                stats.unresolved += 1
                if on_unresolved is not None:
                    on_unresolved(match, reason)
                else:
                    append_unresolved_odds_row(
                        self.config.unresolved_football_data_odds_csv_path,
                        home_team=match.home_team,
                        away_team=match.away_team,
                        match_date=match.match_date.isoformat(),
                        league_code=source,
                        season=match.season or season,
                        reason=reason,
                    )
                continue
            stats.resolved += 1
            if fixture.id in existing:
                stats.skipped += 1
                continue
            if dry_run:
                continue
            kickoff_at = _as_aware(fixture.fixture_date)
            for triple in match.triples:
                self.odds_repo.upsert(
                    fixture_id=fixture.id,
                    provider=PROVIDER,
                    source=source,
                    bookmaker=triple.bookmaker,
                    price_type=triple.price_type,
                    odds_home=triple.odds_home,
                    odds_draw=triple.odds_draw,
                    odds_away=triple.odds_away,
                    snapshot_at=kickoff_at,
                    kickoff_at=kickoff_at,
                    raw_payload=odds_payload(match.raw_row),
                )
                stats.upserted += 1
        return stats

    def _resolve_fixture(
        self,
        match: ParsedMatchOdds,
        *,
        league_external_id: int | None,
    ) -> tuple[FixtureModel | None, str]:
        home_team = self._resolve_team(match.home_team, league_external_id)
        away_team = self._resolve_team(match.away_team, league_external_id)
        if home_team is None and away_team is None:
            reason = "unresolved_teams"
        elif home_team is None:
            reason = "unresolved_home_team"
        elif away_team is None:
            reason = "unresolved_away_team"
        else:
            reason = "unresolved_fixture"

        provider_match = ProviderMatch(
            provider_match_id=f"{match.source}:{match.match_date.isoformat()}:{match.home_team}:{match.away_team}",
            provider_league_id=match.source,
            provider_season_id=match.season,
            home_team_id="",
            away_team_id="",
            home_team_name=match.home_team,
            away_team_name=match.away_team,
            kickoff_at=match.kickoff_at,
            status="finished",
        )
        resolution = self.resolver.resolve_match(
            provider_match,
            league_external_id=league_external_id,
            league_id=None,
            home_team=home_team,
            away_team=away_team,
            season=match.season,
        )
        if resolution.match is not None:
            return resolution.match, ""
        fallback = self._resolve_fixture_by_exact_names(
            match, league_external_id=league_external_id
        )
        if fallback is not None:
            return fallback, ""
        return None, reason

    def _resolve_fixture_by_exact_names(
        self,
        match: ParsedMatchOdds,
        *,
        league_external_id: int | None,
    ) -> FixtureModel | None:
        """Match on exact TeamModel.name (archive + canonical) within ±1 day.

        EntityResolver normalized lookup can bind 'Norwich' to 'Norwich United'
        because club suffixes are stripped. Exact names avoid that collision
        without changing EntityResolver.
        """
        tolerance = timedelta(minutes=self.config.kickoff_match_tolerance_minutes)
        date_from = (match.kickoff_at - tolerance).date()
        date_to = (match.kickoff_at + tolerance).date()
        home_names = _lookup_team_names(match.home_team)
        away_names = _lookup_team_names(match.away_team)
        candidates = self.fixture_repo.find_by_date_range_and_teams(
            date_from=date_from,
            date_to=date_to,
            home_names=home_names,
            away_names=away_names,
            league_external_id=league_external_id,
        )
        if not candidates and league_external_id is not None:
            candidates = self.fixture_repo.find_by_date_range_and_teams(
                date_from=date_from,
                date_to=date_to,
                home_names=home_names,
                away_names=away_names,
                league_external_id=None,
            )
        if not candidates:
            return None
        if len(candidates) > 1:
            candidates.sort(
                key=lambda fixture: abs(
                    (_as_fixture_date(fixture.fixture_date) - match.match_date).days
                )
            )
        return candidates[0]

    def _resolve_team(
        self, archive_name: str, league_external_id: int | None
    ) -> TeamModel | None:
        if archive_name == 'Odd':
            l=1
        lookup_name = canonical_team_name(archive_name)
        for candidate in (lookup_name, archive_name):
            result = self.resolver.resolve_team(
                provider_team_id="",
                provider_team_name=candidate,
                league_id=league_external_id,
            )
            if result.team is not None:
                return result.team
        return None

    def _league_external_id(self, code: str) -> int | None:
        meta = self._league_map.get(code)
        if meta is None:
            return None
        league_id = meta.get("league_id")
        return int(league_id) if league_id is not None else None


def format_retry_totals(totals: RetryTotals) -> str:
    reason_part = " ".join(
        f"{name}={count}" for name, count in sorted(totals.by_reason.items())
    )
    league_part = " ".join(
        f"{name}={count}" for name, count in sorted(totals.by_league.items())
    )
    return (
        f"attempted={totals.attempted} newly_resolved={totals.newly_resolved} "
        f"upserted={totals.upserted} skipped={totals.skipped} "
        f"still_unresolved={totals.still_unresolved}\n"
        f"by_reason: {reason_part or '(none)'}\n"
        f"by_league: {league_part or '(none)'}"
    )


def format_stats_line(stats: LeagueSeasonStats) -> str:
    failed = " fetch_failed=1" if stats.fetch_failed else ""
    return (
        f"{stats.source} {stats.season}: csv_rows={stats.csv_rows} "
        f"resolved={stats.resolved} unresolved={stats.unresolved} "
        f"upserted={stats.upserted} skipped={stats.skipped}{failed}"
    )


def english_tier_resolution_pct(summaries: list[LeagueSeasonStats]) -> float | None:
    english = [item for item in summaries if item.source in ENGLISH_TIER_CODES]
    csv_rows = sum(item.csv_rows for item in english)
    if csv_rows == 0:
        return None
    resolved = sum(item.resolved for item in english)
    return 100.0 * resolved / csv_rows


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _as_fixture_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _lookup_team_names(archive_name: str) -> list[str]:
    names = [archive_name]
    canonical = canonical_team_name(archive_name)
    if canonical not in names:
        names.append(canonical)
    return names


def _ignore_resolver_unresolved(*args: object, **kwargs: object) -> None:
    """Odds ingest logs unresolved rows itself; skip EntityResolver's CSV."""
    return None


def _parsed_match_key(match: ParsedMatchOdds) -> tuple[str, str, str, str]:
    return (
        match.home_team,
        match.away_team,
        match.match_date.isoformat(),
        match.source,
    )


def _odds_csv_path(code: str, season: str) -> str:
    if code in EXTRA_CODES:
        return f"{EXTRA_BASE_PATH}/{code}.csv"
    return f"{MAIN_BASE_PATH}/{season}/{code}.csv"


def _csv_groups_for_listed(
    listed: list[UnresolvedOddsRow],
) -> list[tuple[str, str, bool]]:
    extra_codes: set[str] = set()
    main_keys: set[tuple[str, str]] = set()
    for row in listed:
        if row.league_code in EXTRA_CODES:
            extra_codes.add(row.league_code)
        else:
            main_keys.add((row.league_code, row.season))
    groups: list[tuple[str, str, bool]] = []
    for code, season in sorted(main_keys):
        groups.append((code, season, False))
    for code in sorted(extra_codes):
        groups.append((code, "all", True))
    return groups


def _unique_listed_rows(
    rows: list[UnresolvedOddsRow],
    *,
    league: str | None,
    leagues_path: Path,
) -> list[UnresolvedOddsRow]:
    if league:
        allowed = set(resolve_league_codes(league, leagues_path=leagues_path))
        rows = [row for row in rows if row.league_code in allowed]
    unique: dict[tuple[str, str, str, str], UnresolvedOddsRow] = {}
    ordered: list[UnresolvedOddsRow] = []
    for row in rows:
        if row.match_key in unique:
            continue
        unique[row.match_key] = row
        ordered.append(row)
    return ordered
