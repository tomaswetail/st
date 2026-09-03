"""Backfill injury/availability snapshots from API-Football only."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from data_sources.api_football_client import APIFootballClient
from data_sources.injuries.dtos import MatchAvailabilitySnapshot, PlayerAvailabilityRecord
from data_sources.injuries.parsers.api_football import parse_api_football_availability
from objects.models.fixture import FixtureModel
from objects.models.st_match import STMatchModel
from objects.models.st_round import STRoundModel
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.injury_snapshot_repository import InjurySnapshotRepository
from objects.repositories.match_availability_repository import MatchAvailabilityRepository
from objects.repositories.player_repository import PlayerRepository
from objects.schema.data_classes.data_sources import DataSourceConfig

logger = logging.getLogger(__name__)

PROVIDER = "api-football"


@dataclass
class InjuryBackfillResult:
    requested: int = 0
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    no_data: int = 0
    failed: int = 0
    fixture_ids: list[int] = field(default_factory=list)


def _serialize_player(player: PlayerAvailabilityRecord) -> dict[str, Any]:
    return {
        "external_id": player.external_id,
        "provider": player.provider,
        "team_side": player.team_side,
        "status": player.status,
        "is_starter": player.is_starter,
        "name": player.name,
        "team_external_id": player.team_external_id,
        "market_value": player.market_value,
        "shirt_number": player.shirt_number,
        "position_id": player.position_id,
    }


class InjuryBackfillService:
    """Fetch API-Football lineups + injuries and persist availability snapshots.

    Sole injury/lineup source for Phase 2. FotMob/SofaScore are not used.
    """

    def __init__(
        self,
        session: Session,
        *,
        client: APIFootballClient | None = None,
        config: DataSourceConfig | None = None,
        request_delay_sec: float = 0.25,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.client = client or APIFootballClient(config=self.config)
        self.request_delay_sec = request_delay_sec
        self.fixture_repo = FixtureRepository(session)
        self.player_repo = PlayerRepository(session)
        self.injury_repo = InjurySnapshotRepository(session)
        self.availability_repo = MatchAvailabilityRepository(session)

    def backfill(
        self,
        *,
        draw_min: int | None = None,
        draw_max: int | None = None,
        after_date: date | None = None,
        before_date: date | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
        dry_run: bool = False,
        skip_http: bool = False,
    ) -> InjuryBackfillResult:
        candidates = self._load_candidate_fixtures(
            draw_min=draw_min,
            draw_max=draw_max,
            after_date=after_date,
            before_date=before_date,
            limit=limit,
        )
        result = InjuryBackfillResult(requested=len(candidates))
        existing = self.availability_repo.fixture_ids_with_availability(
            [fixture.id for fixture in candidates],
            provider=PROVIDER,
        )

        for fixture in candidates:
            if (
                not force_refresh
                and fixture.id in existing
                and self.availability_repo.get_for_fixture(
                    fixture.id,
                    provider=PROVIDER,
                )
                is not None
            ):
                result.skipped += 1
                continue

            if skip_http:
                result.imported += 1
                result.fixture_ids.append(fixture.id)
                continue

            try:
                lineups_payload, injuries_payload = self._fetch_payloads(
                    int(fixture.fixture_id)
                )
            except Exception:
                logger.exception(
                    "API-Football fetch failed fixture_pk=%s api_fixture_id=%s",
                    fixture.id,
                    fixture.fixture_id,
                )
                result.failed += 1
                continue

            kickoff = fixture.fixture_date
            if not isinstance(kickoff, datetime):
                kickoff = datetime.combine(kickoff, datetime.min.time())

            try:
                snapshot = parse_api_football_availability(
                    fixture_id=fixture.id,
                    home_team_external_id=fixture.home_team_id,
                    away_team_external_id=fixture.away_team_id,
                    kickoff_at=kickoff,
                    lineups_payload=lineups_payload,
                    injuries_payload=injuries_payload,
                )
            except Exception:
                logger.exception(
                    "Parse failed fixture_pk=%s api_fixture_id=%s",
                    fixture.id,
                    fixture.fixture_id,
                )
                result.failed += 1
                continue

            if snapshot is None:
                result.no_data += 1
                continue

            if dry_run:
                result.imported += 1
                result.fixture_ids.append(fixture.id)
                continue

            already_exists = self.availability_repo.get_for_fixture(
                fixture.id,
                provider=PROVIDER,
                source=snapshot.source,
            )
            self._persist_snapshot(snapshot)
            if already_exists is None:
                result.imported += 1
            else:
                result.updated += 1
            result.fixture_ids.append(fixture.id)
            existing.add(fixture.id)

        if not dry_run:
            self.session.commit()
        return result

    def _fetch_payloads(
        self, api_fixture_id: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        lineups = self.client.get_fixture_lineups(api_fixture_id)
        if self.request_delay_sec > 0:
            time.sleep(self.request_delay_sec)
        injuries = self.client.get_fixture_injuries(api_fixture_id)
        if self.request_delay_sec > 0:
            time.sleep(self.request_delay_sec)
        return lineups, injuries

    def _persist_snapshot(self, snapshot: MatchAvailabilitySnapshot) -> None:
        self.injury_repo.delete_for_fixture_provider(
            snapshot.fixture_id,
            snapshot.provider,
        )
        for player in snapshot.players:
            player_row = self.player_repo.upsert(
                provider=player.provider,
                external_id=player.external_id,
                name=player.name,
                team_external_id=player.team_external_id,
                market_value=player.market_value,
            )
            self.injury_repo.upsert(
                fixture_id=snapshot.fixture_id,
                player_id=player_row.id,
                provider=snapshot.provider,
                team_side=player.team_side,
                status=player.status,
                is_starter=player.is_starter,
                market_value=player.market_value,
                snapshot_at=snapshot.snapshot_at,
                source=snapshot.source,
                raw_payload=player.raw_payload,
            )

        self.availability_repo.upsert(
            fixture_id=snapshot.fixture_id,
            provider=snapshot.provider,
            source=snapshot.source,
            snapshot_at=snapshot.snapshot_at,
            kickoff_at=snapshot.kickoff_at,
            home_team_external_id=snapshot.home_team_external_id,
            away_team_external_id=snapshot.away_team_external_id,
            home_starter_count=snapshot.home_starter_count,
            away_starter_count=snapshot.away_starter_count,
            home_unavailable_count=snapshot.home_unavailable_count,
            away_unavailable_count=snapshot.away_unavailable_count,
            home_missing_value=snapshot.home_missing_value,
            away_missing_value=snapshot.away_missing_value,
            coverage_level=snapshot.coverage_level,
            players_json=[_serialize_player(player) for player in snapshot.players],
            raw_payload=snapshot.raw_payload,
        )

    def _load_candidate_fixtures(
        self,
        *,
        draw_min: int | None,
        draw_max: int | None,
        after_date: date | None,
        before_date: date | None,
        limit: int | None,
    ) -> list[FixtureModel]:
        """Resolve ST matches in the draw window to internal FixtureModel rows."""
        query = (
            select(STMatchModel)
            .join(STRoundModel, STMatchModel.stryktipset_round_id == STRoundModel.id)
            .options(
                selectinload(STMatchModel.home_team),
                selectinload(STMatchModel.away_team),
            )
            .where(STMatchModel.start_time.is_not(None))
            .order_by(STMatchModel.start_time.asc())
        )
        if draw_min is not None:
            query = query.where(STRoundModel.draw_number >= draw_min)
        if draw_max is not None:
            query = query.where(STRoundModel.draw_number <= draw_max)

        matches = list(self.session.scalars(query).all())
        fixtures: list[FixtureModel] = []
        seen: set[int] = set()
        tolerance_days = max(
            1,
            int(self.config.kickoff_match_tolerance_minutes / (24 * 60)) or 1,
        )

        for match in matches:
            if match.home_team is None or match.away_team is None:
                continue
            home_external_id = getattr(match.home_team, "external_id", None)
            away_external_id = getattr(match.away_team, "external_id", None)
            if home_external_id is None or away_external_id is None:
                continue
            kickoff = match.start_time
            kickoff_date = kickoff.date() if isinstance(kickoff, datetime) else kickoff
            if after_date is not None and kickoff_date < after_date:
                continue
            if before_date is not None and kickoff_date > before_date:
                continue

            candidates = self.fixture_repo.find_by_date_range_and_teams(
                date_from=kickoff_date - timedelta(days=tolerance_days),
                date_to=kickoff_date + timedelta(days=tolerance_days),
                home_team_ids=[int(home_external_id)],
                away_team_ids=[int(away_external_id)],
            )
            if not candidates:
                continue
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = min(
                    candidates,
                    key=lambda row: abs(
                        (
                            (
                                row.fixture_date.date()
                                if isinstance(row.fixture_date, datetime)
                                else row.fixture_date
                            )
                            - kickoff_date
                        ).days
                    ),
                )
            if chosen.id in seen:
                continue
            seen.add(chosen.id)
            fixtures.append(chosen)
            if limit is not None and len(fixtures) >= limit:
                break

        return fixtures
