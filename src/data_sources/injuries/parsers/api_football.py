"""Parse API-Football injuries + lineups into availability snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.data_sources.injuries.dtos import MatchAvailabilitySnapshot, PlayerAvailabilityRecord

PROVIDER = "api-football"
SOURCE = "api_football_injuries_lineups"


def _parse_kickoff(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return fallback or datetime.now(tz=timezone.utc)


def _injury_status(player_block: dict[str, Any]) -> str:
    type_text = str(player_block.get("type") or "").lower()
    reason_text = str(player_block.get("reason") or "").lower()
    combined = f"{type_text} {reason_text}"
    if "suspend" in combined:
        return "suspended"
    if "doubt" in combined or "question" in combined:
        return "doubtful"
    if "injur" in combined or "missing" in combined:
        return "injured"
    return "unavailable"


def _parse_lineup_side(
    team_block: dict[str, Any],
    *,
    team_side: str,
    home_team_id: str | None,
    away_team_id: str | None,
) -> tuple[str | None, list[PlayerAvailabilityRecord]]:
    team = team_block.get("team") if isinstance(team_block.get("team"), dict) else {}
    team_external_id = str(team.get("id") or "") or None
    resolved_side = team_side
    if team_external_id and home_team_id and team_external_id == home_team_id:
        resolved_side = "home"
    elif team_external_id and away_team_id and team_external_id == away_team_id:
        resolved_side = "away"

    players: list[PlayerAvailabilityRecord] = []
    for role, status, is_starter in (
        ("startXI", "starter", True),
        ("substitutes", "bench", False),
    ):
        roster = team_block.get(role) or []
        if not isinstance(roster, list):
            continue
        for item in roster:
            if not isinstance(item, dict):
                continue
            player = item.get("player") if isinstance(item.get("player"), dict) else item
            external_id = player.get("id")
            if external_id is None:
                continue
            players.append(
                PlayerAvailabilityRecord(
                    external_id=str(external_id),
                    provider=PROVIDER,
                    team_side=resolved_side,  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
                    is_starter=is_starter,
                    name=player.get("name"),
                    team_external_id=team_external_id,
                    shirt_number=(
                        str(player.get("number"))
                        if player.get("number") is not None
                        else None
                    ),
                    raw_payload=item,
                )
            )
    return team_external_id, players


def _parse_injuries(
    injuries_response: list[dict[str, Any]],
    *,
    home_team_id: str | None,
    away_team_id: str | None,
) -> list[PlayerAvailabilityRecord]:
    players: list[PlayerAvailabilityRecord] = []
    for row in injuries_response:
        if not isinstance(row, dict):
            continue
        player = row.get("player") if isinstance(row.get("player"), dict) else {}
        team = row.get("team") if isinstance(row.get("team"), dict) else {}
        external_id = player.get("id")
        if external_id is None:
            continue
        team_external_id = str(team.get("id") or "") or None
        if team_external_id and home_team_id and team_external_id == home_team_id:
            team_side = "home"
        elif team_external_id and away_team_id and team_external_id == away_team_id:
            team_side = "away"
        else:
            # Unknown side — skip rather than mis-attribute
            continue
        players.append(
            PlayerAvailabilityRecord(
                external_id=str(external_id),
                provider=PROVIDER,
                team_side=team_side,  # type: ignore[arg-type]
                status=_injury_status(player),  # type: ignore[arg-type]
                is_starter=False,
                name=player.get("name"),
                team_external_id=team_external_id,
                raw_payload=row,
            )
        )
    return players


def parse_api_football_availability(
    *,
    fixture_id: int,
    home_team_external_id: int | str | None,
    away_team_external_id: int | str | None,
    kickoff_at: datetime,
    lineups_payload: dict[str, Any] | None,
    injuries_payload: dict[str, Any] | None,
) -> MatchAvailabilitySnapshot | None:
    """Build a snapshot from API-Football ``/fixtures/lineups`` + ``/injuries``."""
    home_id = str(home_team_external_id) if home_team_external_id is not None else None
    away_id = str(away_team_external_id) if away_team_external_id is not None else None

    lineup_rows = []
    if isinstance(lineups_payload, dict):
        response = lineups_payload.get("response")
        if isinstance(response, list):
            lineup_rows = [row for row in response if isinstance(row, dict)]

    injury_rows: list[dict[str, Any]] = []
    if isinstance(injuries_payload, dict):
        response = injuries_payload.get("response")
        if isinstance(response, list):
            injury_rows = [row for row in response if isinstance(row, dict)]

    players: list[PlayerAvailabilityRecord] = []
    resolved_home_id = home_id
    resolved_away_id = away_id

    if len(lineup_rows) >= 1:
        team_id_0, players_0 = _parse_lineup_side(
            lineup_rows[0],
            team_side="home",
            home_team_id=home_id,
            away_team_id=away_id,
        )
        players.extend(players_0)
        if team_id_0 and not resolved_home_id:
            resolved_home_id = team_id_0
    if len(lineup_rows) >= 2:
        team_id_1, players_1 = _parse_lineup_side(
            lineup_rows[1],
            team_side="away",
            home_team_id=home_id,
            away_team_id=away_id,
        )
        players.extend(players_1)
        if team_id_1 and not resolved_away_id:
            resolved_away_id = team_id_1

    # Deduplicate injuries against lineup player ids (injury wins as unavailable).
    injury_players = _parse_injuries(
        injury_rows,
        home_team_id=resolved_home_id,
        away_team_id=resolved_away_id,
    )
    injury_ids = {player.external_id for player in injury_players}
    players = [player for player in players if player.external_id not in injury_ids]
    players.extend(injury_players)

    if not players:
        return None

    kickoff = _parse_kickoff(kickoff_at)
    # Prefer fixture date from injuries payload when present.
    if injury_rows:
        fixture_block = injury_rows[0].get("fixture")
        if isinstance(fixture_block, dict):
            kickoff = _parse_kickoff(
                fixture_block.get("date") or fixture_block.get("timestamp"),
                fallback=kickoff,
            )

    home_starters = sum(
        1 for player in players if player.team_side == "home" and player.is_starter
    )
    away_starters = sum(
        1 for player in players if player.team_side == "away" and player.is_starter
    )
    unavailable_statuses = {"injured", "suspended", "unavailable", "doubtful"}
    home_unavailable = sum(
        1
        for player in players
        if player.team_side == "home" and player.status in unavailable_statuses
    )
    away_unavailable = sum(
        1
        for player in players
        if player.team_side == "away" and player.status in unavailable_statuses
    )

    if home_starters >= 11 and away_starters >= 11:
        coverage = "full_lineup"
    elif players:
        coverage = "partial"
    else:
        coverage = "none"

    # API-Football has no market values; use unavailable counts as missing-value proxy.
    home_missing_value = float(home_unavailable) if home_unavailable else None
    away_missing_value = float(away_unavailable) if away_unavailable else None

    return MatchAvailabilitySnapshot(
        fixture_id=fixture_id,
        provider=PROVIDER,
        source=SOURCE,
        snapshot_at=kickoff,
        kickoff_at=kickoff,
        home_team_external_id=resolved_home_id,
        away_team_external_id=resolved_away_id,
        players=players,
        home_starter_count=home_starters,
        away_starter_count=away_starters,
        home_unavailable_count=home_unavailable,
        away_unavailable_count=away_unavailable,
        home_missing_value=home_missing_value,
        away_missing_value=away_missing_value,
        coverage_level=coverage,
        raw_payload={
            "lineups": lineups_payload,
            "injuries": injuries_payload,
        },
    )
