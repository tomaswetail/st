"""SofaScore historical football data adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from data_sources.football_data.http_client import ThrottledHttpClient
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    ProviderLeague,
    ProviderMatch,
    ProviderMatchDetails,
    ProviderSeason,
    ProviderShot,
)

logger = logging.getLogger(__name__)


def parse_sofascore_categories(payload: Any) -> list[dict[str, Any]]:
    """Return category dicts from SofaScore football categories payload."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    categories = payload.get("categories") or payload.get("data") or []
    if isinstance(categories, list):
        return [item for item in categories if isinstance(item, dict)]
    return []


def parse_sofascore_unique_tournaments(
    payload: Any,
    *,
    country: str | None = None,
    country_code: str | None = None,
) -> list[ProviderLeague]:
    """Flatten unique-tournaments payload for one country/category."""
    tournaments_raw: Any
    if isinstance(payload, list):
        tournaments_raw = payload
    elif isinstance(payload, dict):
        groups = payload.get("groups")
        if isinstance(groups, list):
            tournaments_raw = []
            for group in groups:
                if not isinstance(group, dict):
                    continue
                unique = group.get("uniqueTournaments") or group.get("tournaments") or []
                if isinstance(unique, list):
                    tournaments_raw.extend(unique)
        else:
            tournaments_raw = (
                payload.get("uniqueTournaments")
                or payload.get("tournaments")
                or payload.get("data")
                or []
            )
    else:
        tournaments_raw = []

    leagues: list[ProviderLeague] = []
    for item in tournaments_raw or []:
        if not isinstance(item, dict):
            continue
        # Nested uniqueTournament object is common.
        tournament = item.get("uniqueTournament") if isinstance(item.get("uniqueTournament"), dict) else item
        league_id = tournament.get("id")
        name = tournament.get("name")
        if league_id is None or not name:
            continue
        category = tournament.get("category") if isinstance(tournament.get("category"), dict) else {}
        leagues.append(
            ProviderLeague(
                provider_league_id=str(league_id),
                name=str(name),
                country=country or category.get("name"),
                country_code=country_code or category.get("alpha2") or category.get("flag"),
                raw_payload=item if item is not tournament else tournament,
            )
        )
    return leagues


def parse_sofascore_available_leagues(
    category_payload: Any,
    tournaments_by_category: dict[str, Any],
) -> list[ProviderLeague]:
    """Combine categories + per-category tournament payloads into a flat catalogue."""
    leagues: list[ProviderLeague] = []
    seen: set[str] = set()
    for category in parse_sofascore_categories(category_payload):
        category_id = str(category.get("id") or "")
        if not category_id:
            continue
        country = str(category.get("name") or "") or None
        country_code = category.get("alpha2") or category.get("flag")
        payload = tournaments_by_category.get(category_id) or {}
        for league in parse_sofascore_unique_tournaments(
            payload,
            country=country,
            country_code=str(country_code) if country_code else None,
        ):
            if league.provider_league_id in seen:
                continue
            seen.add(league.provider_league_id)
            leagues.append(league)
    return leagues


def _parse_kickoff(value: Any) -> datetime:
    """Parse SofaScore kickoff timestamps into UTC datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_sofascore_seasons(payload: Any, provider_league_id: str) -> list[ProviderSeason]:
    """Parse SofaScore seasons into ProviderSeason rows."""
    seasons_raw = payload
    if isinstance(payload, dict):
        seasons_raw = payload.get("seasons") or payload.get("data") or []
    seasons: list[ProviderSeason] = []
    for item in seasons_raw or []:
        if isinstance(item, str):
            seasons.append(
                ProviderSeason(
                    provider_season_id=item,
                    name=item,
                    raw_payload={"season": item, "league_id": provider_league_id},
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        year = item.get("year") if isinstance(item.get("year"), dict) else {}
        season_id = str(
            item.get("id")
            or item.get("seasonId")
            or year.get("year")
            or item.get("name")
            or ""
        )
        if not season_id:
            continue
        name = str(item.get("name") or year.get("year") or season_id)
        seasons.append(
            ProviderSeason(
                provider_season_id=season_id,
                name=name,
                start_year=_to_int(item.get("startDate") or year.get("start")),
                end_year=_to_int(item.get("endDate") or year.get("end")),
                raw_payload=item,
            )
        )
    return seasons


def parse_sofascore_matches(
    payload: Any,
    provider_league_id: str,
    provider_season_id: str,
) -> list[ProviderMatch]:
    """Parse SofaScore events into ProviderMatch rows."""
    events: list[Any] = []
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = payload.get("events") or payload.get("matches") or payload.get("data") or []

    result: list[ProviderMatch] = []
    for item in events or []:
        if not isinstance(item, dict):
            continue
        match = _parse_sofascore_event(item, provider_league_id, provider_season_id)
        if match is not None:
            result.append(match)
    return result


def _parse_sofascore_event(
    item: dict[str, Any],
    provider_league_id: str,
    provider_season_id: str | None,
) -> ProviderMatch | None:
    """Build a ProviderMatch from one SofaScore event dict."""
    match_id = item.get("id") or item.get("provider_match_id")
    if match_id is None:
        return None
    home = item.get("homeTeam") or item.get("home") or {}
    away = item.get("awayTeam") or item.get("away") or {}
    if not isinstance(home, dict):
        home = {"name": str(home)}
    if not isinstance(away, dict):
        away = {"name": str(away)}

    kickoff_raw = (
        item.get("startTimestamp")
        or item.get("kickoff_at")
        or item.get("startTime")
        or item.get("date")
    )
    if kickoff_raw is None:
        return None

    status_obj = item.get("status") if isinstance(item.get("status"), dict) else {}
    status = str(
        status_obj.get("type")
        or status_obj.get("description")
        or item.get("status")
        or "unknown"
    )
    score = item.get("homeScore") if isinstance(item.get("homeScore"), dict) else {}
    away_score_obj = item.get("awayScore") if isinstance(item.get("awayScore"), dict) else {}

    return ProviderMatch(
        provider_match_id=str(match_id),
        provider_league_id=str(
            (item.get("tournament") or {}).get("uniqueTournament", {}).get("id")
            if isinstance(item.get("tournament"), dict)
            else item.get("provider_league_id") or provider_league_id
        ),
        provider_season_id=str(
            (item.get("season") or {}).get("id")
            if isinstance(item.get("season"), dict)
            else item.get("provider_season_id") or provider_season_id or ""
        )
        or None,
        home_team_id=str(home.get("id") or item.get("home_team_id") or ""),
        away_team_id=str(away.get("id") or item.get("away_team_id") or ""),
        home_team_name=str(home.get("name") or item.get("home_team_name") or ""),
        away_team_name=str(away.get("name") or item.get("away_team_name") or ""),
        kickoff_at=_parse_kickoff(kickoff_raw),
        status=status,
        home_score=_to_int(
            item.get("home_score")
            or score.get("current")
            or score.get("display")
        ),
        away_score=_to_int(
            item.get("away_score")
            or away_score_obj.get("current")
            or away_score_obj.get("display")
        ),
        raw_payload=item,
    )


def parse_sofascore_shots(payload: Any) -> list[ProviderShot]:
    """Parse SofaScore shotmap payload into ProviderShot rows."""
    shots_raw: Any = payload
    if isinstance(payload, dict):
        shots_raw = (
            payload.get("shotmap")
            or payload.get("shots")
            or payload.get("data")
            or []
        )
        if isinstance(shots_raw, dict):
            shots_raw = shots_raw.get("shotmap") or shots_raw.get("shots") or []

    shots: list[ProviderShot] = []
    for item in shots_raw or []:
        if not isinstance(item, dict):
            continue
        player = item.get("player") if isinstance(item.get("player"), dict) else {}
        team_id = str(
            item.get("teamId")
            or item.get("team_id")
            or (item.get("team") or {}).get("id")
            or ""
        )
        if not team_id:
            if item.get("isHome") is True:
                team_id = str(
                    item.get("home_team_id") or item.get("homeTeamId") or "home"
                )
            elif item.get("isHome") is False:
                team_id = str(
                    item.get("away_team_id") or item.get("awayTeamId") or "away"
                )
        if not team_id:
            continue

        player_coords = item.get("playerCoordinates") or item.get("coordinates") or {}
        coords = None
        if isinstance(player_coords, dict) and (
            player_coords.get("x") is not None or player_coords.get("y") is not None
        ):
            coords = {"x": player_coords.get("x"), "y": player_coords.get("y")}

        situation = item.get("situation") or item.get("shotType")
        is_penalty = bool(
            item.get("is_penalty")
            or str(situation).lower() in {"penalty", "penalties"}
            or item.get("footballShotType") == "penalty"
        )
        outcome = item.get("shotType") or item.get("outcome") or item.get("goalType")
        shots.append(
            ProviderShot(
                provider_shot_id=(
                    str(item["id"])
                    if item.get("id") is not None
                    else (
                        str(item["provider_shot_id"])
                        if item.get("provider_shot_id") is not None
                        else None
                    )
                ),
                team_id=team_id,
                player_id=(
                    str(player.get("id") or item.get("player_id"))
                    if (player.get("id") or item.get("player_id")) is not None
                    else None
                ),
                minute=_to_int(
                    (item.get("time") if isinstance(item.get("time"), int) else None)
                    or item.get("minute")
                ),
                second=_to_int(
                    item.get("timeSeconds")
                    if item.get("timeSeconds") is not None
                    else item.get("second")
                ),
                xg=_to_float(item.get("xg") or item.get("expectedGoals")),
                xgot=_to_float(
                    item.get("xgot")
                    or item.get("xGOT")
                    or item.get("expectedGoalsOnTarget")
                ),
                outcome=str(outcome) if outcome is not None else None,
                situation=str(situation) if situation is not None else None,
                body_part=(
                    str(item.get("bodyPart") or item.get("body_part"))
                    if (item.get("bodyPart") or item.get("body_part")) is not None
                    else None
                ),
                shot_type=(
                    str(item.get("footballShotType") or item.get("shot_type"))
                    if (item.get("footballShotType") or item.get("shot_type")) is not None
                    else None
                ),
                is_penalty=is_penalty,
                is_own_goal=bool(item.get("isOwnGoal") or item.get("is_own_goal")),
                coordinates=coords,
                raw_payload=item,
            )
        )
    return shots


def parse_sofascore_match_details(payload: dict[str, Any]) -> ProviderMatchDetails:
    """Parse SofaScore event/shotmap/stats into ProviderMatchDetails."""
    event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    match = _parse_sofascore_event(
        event,
        provider_league_id=str(
            payload.get("provider_league_id")
            or (event.get("tournament") or {}).get("uniqueTournament", {}).get("id")
            if isinstance(event.get("tournament"), dict)
            else ""
        ),
        provider_season_id=(
            str(payload.get("provider_season_id") or (event.get("season") or {}).get("id"))
            if payload.get("provider_season_id")
            or (isinstance(event.get("season"), dict) and event.get("season", {}).get("id"))
            else None
        ),
    )
    if match is None:
        raise ValueError("Unable to parse SofaScore match details")

    # Inject resolved team ids onto shots that only have isHome.
    shots_payload = payload.get("shotmap") or payload.get("shots") or payload
    if isinstance(shots_payload, list):
        enriched = []
        for shot in shots_payload:
            if not isinstance(shot, dict):
                continue
            row = dict(shot)
            row.setdefault("homeTeamId", match.home_team_id)
            row.setdefault("awayTeamId", match.away_team_id)
            row.setdefault("home_team_id", match.home_team_id)
            row.setdefault("away_team_id", match.away_team_id)
            enriched.append(row)
        shots = parse_sofascore_shots(enriched)
    else:
        shots = parse_sofascore_shots(shots_payload)

    statistics = payload.get("statistics") or payload.get("stats") or {}
    home_xg = _to_float(payload.get("home_xg"))
    away_xg = _to_float(payload.get("away_xg"))
    if home_xg is None or away_xg is None:
        extracted_home, extracted_away = _extract_xg_from_statistics(statistics)
        home_xg = home_xg if home_xg is not None else extracted_home
        away_xg = away_xg if away_xg is not None else extracted_away

    return ProviderMatchDetails(
        match=match,
        shots=shots,
        statistics=statistics if isinstance(statistics, dict) else {"raw": statistics},
        lineups=payload.get("lineups") if isinstance(payload.get("lineups"), dict) else None,
        raw_payload=payload,
        home_xg=home_xg,
        away_xg=away_xg,
        home_xgot=_to_float(payload.get("home_xgot")),
        away_xgot=_to_float(payload.get("away_xgot")),
    )


def _extract_xg_from_statistics(statistics: Any) -> tuple[float | None, float | None]:
    """Extract home/away xG from SofaScore statistics groups."""
    if not isinstance(statistics, dict):
        return None, None
    groups = statistics.get("statistics") or statistics.get("groups") or []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            for row in group.get("statisticsItems") or group.get("items") or []:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or row.get("key") or "").lower()
                if "expected goals" in name or name in {"xg", "expected_goals"}:
                    return (
                        _to_float(row.get("home") or row.get("homeValue")),
                        _to_float(row.get("away") or row.get("awayValue")),
                    )
    return (
        _to_float(statistics.get("home_xg")),
        _to_float(statistics.get("away_xg")),
    )


def _to_int(value: Any) -> int | None:
    """Coerce a value to int, or None if invalid."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Coerce a value to float, or None if invalid."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SofaScoreProvider:
    """SofaScore adapter. Endpoint paths stay inside this module."""

    name = "sofascore"

    def __init__(
        self,
        client: ThrottledHttpClient | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Create or wrap a throttled HTTP client for SofaScore."""
        self.config = DataSourceConfig()
        self._owns_client = client is None
        self.client = client or ThrottledHttpClient(
            base_url=self.config.sofascore_base_url,
            timeout_sec=self.config.football_data_http_timeout_sec,
            max_retries=self.config.football_data_max_retries,
            request_delay_ms=self.config.football_data_request_delay_ms,
            cache_ttl_seconds=self.config.football_data_cache_ttl_seconds,
            cache_dir=self.config.football_data_cache_dir / "sofascore",
            user_agent=self.config.football_data_user_agent,
        )

    def close(self) -> None:
        """Close the owned HTTP client if any."""
        if self._owns_client:
            self.client.close()

    def fetch_available_leagues(self) -> list[ProviderLeague]:
        """Fetch categories and unique tournaments as a flat catalogue."""
        categories_payload = self.client.get_json("sport/football/categories/all")
        categories = parse_sofascore_categories(categories_payload)
        tournaments_by_category: dict[str, Any] = {}
        for category in categories:
            category_id = category.get("id")
            if category_id is None:
                continue
            try:
                tournaments_by_category[str(category_id)] = self.client.get_json(
                    f"unique-tournaments/{category_id}"
                )
            except Exception as exc:  # noqa: BLE001 — skip one bad category
                logger.warning(
                    "SofaScore unique tournaments unavailable for category %s: %s",
                    category_id,
                    exc,
                )
        return parse_sofascore_available_leagues(
            categories_payload, tournaments_by_category
        )

    def fetch_league_seasons(self, provider_league_id: str) -> list[ProviderSeason]:
        """Fetch seasons for a SofaScore unique tournament."""
        payload = self.client.get_json(
            f"unique-tournament/{provider_league_id}/seasons"
        )
        return parse_sofascore_seasons(payload, provider_league_id)

    def fetch_season_matches(
        self,
        provider_league_id: str,
        provider_season_id: str,
    ) -> list[ProviderMatch]:
        """Fetch recent season events for a unique tournament."""
        payload = self.client.get_json(
            f"unique-tournament/{provider_league_id}/season/{provider_season_id}/events/last/0"
        )
        return parse_sofascore_matches(payload, provider_league_id, provider_season_id)

    def fetch_match_details(self, provider_match_id: str) -> ProviderMatchDetails:
        """Fetch event, shotmap, and statistics for a match."""
        event = self.client.get_json(f"event/{provider_match_id}")
        shotmap: Any = {}
        try:
            shotmap = self.client.get_json(f"event/{provider_match_id}/shotmap")
        except Exception as exc:  # noqa: BLE001 — soft-fail shots
            logger.warning("SofaScore shotmap unavailable for %s: %s", provider_match_id, exc)
        statistics: Any = {}
        try:
            statistics = self.client.get_json(f"event/{provider_match_id}/statistics")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SofaScore statistics unavailable for %s: %s", provider_match_id, exc
            )

        payload: dict[str, Any]
        if isinstance(event, dict):
            payload = dict(event)
        else:
            payload = {"event": event}
        if "event" not in payload and isinstance(event, dict) and "id" in event:
            payload = {"event": event}
        payload["shotmap"] = shotmap
        payload["statistics"] = statistics
        return parse_sofascore_match_details(payload)

    def fetch_match_shots(self, provider_match_id: str) -> list[ProviderShot]:
        """Return shots from SofaScore match details."""
        details = self.fetch_match_details(provider_match_id)
        return details.shots
