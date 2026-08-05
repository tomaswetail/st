"""FotMob historical football data adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from data_sources.football_data.http_client import (
    FootballDataHttpError,
    ThrottledHttpClient,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    ProviderLeague,
    ProviderMatch,
    ProviderMatchDetails,
    ProviderSeason,
    ProviderShot,
)

logger = logging.getLogger(__name__)


def parse_fotmob_all_leagues(payload: Any) -> list[ProviderLeague]:
    """Flatten FotMob allLeagues catalogue into ProviderLeague rows."""
    if not isinstance(payload, dict):
        return []

    leagues: list[ProviderLeague] = []
    seen: set[str] = set()

    def _add_league(
        item: dict[str, Any],
        *,
        country: str | None,
        country_code: str | None,
    ) -> None:
        """Append one unique ProviderLeague from a catalogue item."""
        league_id = item.get("id")
        name = item.get("name") or item.get("localizedName")
        if league_id is None or not name:
            return
        external_id = str(league_id)
        if external_id in seen:
            return
        seen.add(external_id)
        leagues.append(
            ProviderLeague(
                provider_league_id=external_id,
                name=str(name),
                country=country,
                country_code=country_code or item.get("ccode"),
                raw_payload=item,
            )
        )

    # Prefer country groupings so domestic leagues keep real country metadata.
    for section_key in ("countries", "international"):
        groups = payload.get(section_key) or []
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            country_name = str(
                group.get("name") or group.get("localizedName") or ""
            ) or None
            country_code = group.get("ccode")
            for item in group.get("leagues") or []:
                if isinstance(item, dict):
                    _add_league(
                        item,
                        country=country_name,
                        country_code=country_code or item.get("ccode"),
                    )

    popular = payload.get("popular") or []
    if isinstance(popular, list):
        for item in popular:
            if isinstance(item, dict):
                _add_league(
                    item,
                    country=item.get("country"),
                    country_code=item.get("ccode"),
                )

    return leagues


def _parse_kickoff(value: Any) -> datetime:
    """Parse FotMob kickoff timestamps into UTC datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        # FotMob often uses milliseconds.
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


def parse_fotmob_seasons(payload: Any, provider_league_id: str) -> list[ProviderSeason]:
    """Parse FotMob league seasons into ProviderSeason rows."""
    seasons_raw = payload
    if isinstance(payload, dict):
        seasons_raw = (
            payload.get("seasons")
            or payload.get("allAvailableSeasons")
            or payload.get("data")
            or []
        )
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
        season_id = str(
            item.get("id")
            or item.get("seasonId")
            or item.get("tour")
            or item.get("name")
            or ""
        )
        if not season_id:
            continue
        name = str(item.get("name") or item.get("seasonName") or season_id)
        seasons.append(
            ProviderSeason(
                provider_season_id=season_id,
                name=name,
                start_year=_to_int(item.get("startDate") or item.get("startYear")),
                end_year=_to_int(item.get("endDate") or item.get("endYear")),
                raw_payload=item,
            )
        )
    return seasons


def parse_fotmob_matches(payload: Any, provider_league_id: str, provider_season_id: str) -> list[ProviderMatch]:
    """Parse FotMob league fixtures into ProviderMatch rows."""
    matches_raw: list[Any] = []
    if isinstance(payload, list):
        matches_raw = payload
    elif isinstance(payload, dict):
        matches_raw = (
            payload.get("matches")
            or payload.get("fixtures")
            or ((payload.get("allMatches") or {}).get("allMatches") if isinstance(payload.get("allMatches"), dict) else payload.get("allMatches"))
            or payload.get("data")
            or []
        )
        if isinstance(matches_raw, dict):
            matches_raw = matches_raw.get("matches") or matches_raw.get("allMatches") or []

    result: list[ProviderMatch] = []
    for item in matches_raw or []:
        if not isinstance(item, dict):
            continue
        match = _parse_fotmob_match_dict(item, provider_league_id, provider_season_id)
        if match is not None:
            result.append(match)
    return result


def _parse_fotmob_match_dict(
    item: dict[str, Any],
    provider_league_id: str,
    provider_season_id: str | None,
) -> ProviderMatch | None:
    """Build a ProviderMatch from one FotMob match dict."""
    match_id = item.get("id") or item.get("matchId") or item.get("provider_match_id")
    if match_id is None:
        return None
    home = item.get("home") or item.get("homeTeam") or {}
    away = item.get("away") or item.get("awayTeam") or {}
    if not isinstance(home, dict):
        home = {"name": str(home), "id": item.get("home_team_id")}
    if not isinstance(away, dict):
        away = {"name": str(away), "id": item.get("away_team_id")}

    home_id = str(home.get("id") or item.get("home_team_id") or "")
    away_id = str(away.get("id") or item.get("away_team_id") or "")
    home_name = str(home.get("name") or home.get("longName") or item.get("home_team_name") or "")
    away_name = str(away.get("name") or away.get("longName") or item.get("away_team_name") or "")

    kickoff_raw = (
        item.get("status", {}).get("utcTime")
        if isinstance(item.get("status"), dict)
        else None
    ) or item.get("kickoff_at") or item.get("time") or item.get("matchDate") or item.get("date")
    if kickoff_raw is None:
        return None

    status_obj = item.get("status") if isinstance(item.get("status"), dict) else {}
    status = str(
        status_obj.get("reason", {}).get("short")
        if isinstance(status_obj.get("reason"), dict)
        else status_obj.get("reason")
        or item.get("status")
        or status_obj.get("finished")
        or "unknown"
    )

    score = item.get("score") if isinstance(item.get("score"), dict) else {}
    home_score = _to_int(
        item.get("home_score") or home.get("score") or score.get("home")
    )
    away_score = _to_int(
        item.get("away_score") or away.get("score") or score.get("away")
    )

    return ProviderMatch(
        provider_match_id=str(match_id),
        provider_league_id=str(
            item.get("provider_league_id")
            or item.get("leagueId")
            or provider_league_id
        ),
        provider_season_id=(
            str(item.get("provider_season_id") or item.get("seasonId") or provider_season_id)
            if (item.get("provider_season_id") or item.get("seasonId") or provider_season_id)
            else None
        ),
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_name=home_name,
        away_team_name=away_name,
        kickoff_at=_parse_kickoff(kickoff_raw),
        status=status,
        home_score=home_score,
        away_score=away_score,
        raw_payload=item,
    )


def parse_fotmob_shots(payload: Any) -> list[ProviderShot]:
    """Parse FotMob shotmap payload into ProviderShot rows."""
    shots_raw: Any = payload
    if isinstance(payload, dict):
        content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
        shotmap = content.get("shotmap") if isinstance(content.get("shotmap"), dict) else {}
        shots_raw = (
            payload.get("shotmap")
            or payload.get("shots")
            or shotmap.get("shots")
            or payload.get("data")
            or []
        )
        if isinstance(shots_raw, dict):
            shots_raw = shots_raw.get("shots") or shots_raw.get("shotmap") or []

    shots: list[ProviderShot] = []
    for item in shots_raw or []:
        if not isinstance(item, dict):
            continue
        team_id = str(
            item.get("teamId")
            or item.get("team_id")
            or (item.get("team") or {}).get("id")
            or ""
        )
        if not team_id:
            continue
        coords = None
        if item.get("x") is not None or item.get("y") is not None:
            coords = {"x": item.get("x"), "y": item.get("y")}
        elif isinstance(item.get("coordinates"), dict):
            coords = item["coordinates"]

        situation = item.get("situation") or item.get("shotType")
        is_penalty = bool(
            item.get("is_penalty")
            or item.get("isPenalty")
            or str(situation).lower() in {"penalty", "penalties"}
        )
        outcome = item.get("outcome") or item.get("eventType") or item.get("shotResult")
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
                    str(item.get("playerId") or item.get("player_id"))
                    if (item.get("playerId") or item.get("player_id")) is not None
                    else None
                ),
                minute=_to_int(item.get("minute") or item.get("min")),
                second=_to_int(item.get("second") or item.get("sec")),
                xg=_to_float(item.get("xg") or item.get("expectedGoals")),
                xgot=_to_float(
                    item.get("xgot")
                    or item.get("expectedGoalsOnTarget")
                    or item.get("xGOT")
                ),
                outcome=str(outcome) if outcome is not None else None,
                situation=str(situation) if situation is not None else None,
                body_part=(
                    str(item.get("bodyPart") or item.get("body_part"))
                    if (item.get("bodyPart") or item.get("body_part")) is not None
                    else None
                ),
                shot_type=(
                    str(item.get("shotType") or item.get("shot_type"))
                    if (item.get("shotType") or item.get("shot_type")) is not None
                    else None
                ),
                is_penalty=is_penalty,
                is_own_goal=bool(item.get("isOwnGoal") or item.get("is_own_goal")),
                coordinates=coords,
                raw_payload=item,
            )
        )
    return shots


def parse_fotmob_match_details(payload: dict[str, Any]) -> ProviderMatchDetails:
    """Parse FotMob matchDetails into ProviderMatchDetails."""
    general = payload.get("general") or payload
    header = payload.get("header") or {}
    teams = header.get("teams") if isinstance(header, dict) else None

    home_team: dict[str, Any] = {}
    away_team: dict[str, Any] = {}
    if isinstance(teams, list) and len(teams) >= 2:
        home_team = teams[0] if isinstance(teams[0], dict) else {}
        away_team = teams[1] if isinstance(teams[1], dict) else {}
    else:
        home_team = general.get("homeTeam") or payload.get("home") or {}
        away_team = general.get("awayTeam") or payload.get("away") or {}

    match_id = (
        general.get("matchId")
        or payload.get("matchId")
        or payload.get("id")
        or payload.get("provider_match_id")
    )
    league_id = str(
        (general.get("parentLeagueId") if isinstance(general, dict) else None)
        or payload.get("provider_league_id")
        or ""
    )
    season_id = (
        str(general.get("seasonId") or payload.get("provider_season_id"))
        if (isinstance(general, dict) and general.get("seasonId"))
        or payload.get("provider_season_id")
        else None
    )
    kickoff = (
        (general.get("matchTimeUTC") if isinstance(general, dict) else None)
        or payload.get("kickoff_at")
        or payload.get("matchDate")
    )
    match = ProviderMatch(
        provider_match_id=str(match_id),
        provider_league_id=league_id,
        provider_season_id=season_id,
        home_team_id=str(home_team.get("id") or payload.get("home_team_id") or ""),
        away_team_id=str(away_team.get("id") or payload.get("away_team_id") or ""),
        home_team_name=str(
            home_team.get("name") or payload.get("home_team_name") or ""
        ),
        away_team_name=str(
            away_team.get("name") or payload.get("away_team_name") or ""
        ),
        kickoff_at=_parse_kickoff(kickoff or datetime.now(tz=timezone.utc)),
        status=str(
            (general.get("finished") if isinstance(general, dict) else None)
            or payload.get("status")
            or "unknown"
        ),
        home_score=_to_int(
            home_team.get("score")
            or payload.get("home_score")
            or (payload.get("score") or {}).get("home")
        ),
        away_score=_to_int(
            away_team.get("score")
            or payload.get("away_score")
            or (payload.get("score") or {}).get("away")
        ),
        raw_payload=payload,
    )
    shots = parse_fotmob_shots(payload)
    if not shots and "shots" in payload:
        shots = parse_fotmob_shots(payload["shots"])

    stats = payload.get("statistics") or payload.get("stats") or {}
    home_xg, away_xg = _extract_xg_from_stats(stats)
    if payload.get("home_xg") is not None:
        home_xg = _to_float(payload.get("home_xg"))
    if payload.get("away_xg") is not None:
        away_xg = _to_float(payload.get("away_xg"))

    return ProviderMatchDetails(
        match=match,
        shots=shots,
        statistics=stats if isinstance(stats, dict) else {"raw": stats},
        lineups=payload.get("lineups") if isinstance(payload.get("lineups"), dict) else None,
        raw_payload=payload,
        home_xg=home_xg,
        away_xg=away_xg,
        home_xgot=_to_float(payload.get("home_xgot")),
        away_xgot=_to_float(payload.get("away_xgot")),
    )


def _extract_xg_from_stats(stats: Any) -> tuple[float | None, float | None]:
    """Extract home/away xG from FotMob statistics blocks."""
    if not isinstance(stats, dict):
        return None, None
    periods = stats.get("Periods") or stats.get("periods") or stats
    if isinstance(periods, dict):
        all_stats = periods.get("All") or periods.get("all") or periods
        if isinstance(all_stats, dict):
            teams = all_stats.get("stats") or all_stats.get("Teams") or []
            # FotMob shape varies; also support flat home/away.
            if "expected_goals" in all_stats or "xg" in all_stats:
                return (
                    _to_float(all_stats.get("expected_goals") or all_stats.get("xg")),
                    None,
                )
            for block in teams if isinstance(teams, list) else []:
                title = str(block.get("title") or block.get("key") or "").lower()
                if "expected goals" in title or title in {"xg", "expected_goals"}:
                    stats_list = block.get("stats") or block.get("statsItems") or []
                    # Sometimes values are [home, away]
                    if isinstance(block.get("stats"), list) and block["stats"]:
                        pass
                    home = block.get("home") or block.get("homeValue")
                    away = block.get("away") or block.get("awayValue")
                    if home is not None or away is not None:
                        return _to_float(home), _to_float(away)
                    for row in stats_list if isinstance(stats_list, list) else []:
                        if not isinstance(row, dict):
                            continue
                        key = str(row.get("key") or row.get("title") or "").lower()
                        if "expected goals" in key or key == "xg":
                            return (
                                _to_float(row.get("home") or row.get("homeValue")),
                                _to_float(row.get("away") or row.get("awayValue")),
                            )
    return (
        _to_float(stats.get("home_xg") or stats.get("homeXg")),
        _to_float(stats.get("away_xg") or stats.get("awayXg")),
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


class FotMobProvider:
    """FotMob adapter. Endpoint paths stay inside this module."""

    name = "fotmob"

    def __init__(
        self,
        client: ThrottledHttpClient | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Create or wrap a throttled HTTP client for FotMob."""
        self.config = DataSourceConfig()
        self._owns_client = client is None
        self.client = client or ThrottledHttpClient(
            base_url=self.config.fotmob_base_url,
            timeout_sec=self.config.football_data_http_timeout_sec,
            max_retries=self.config.football_data_max_retries,
            request_delay_ms=self.config.football_data_request_delay_ms,
            cache_ttl_seconds=self.config.football_data_cache_ttl_seconds,
            cache_dir=self.config.football_data_cache_dir / "fotmob",
            user_agent=self.config.football_data_user_agent,
        )

    def close(self) -> None:
        """Close the owned HTTP client if any."""
        if self._owns_client:
            self.client.close()

    def fetch_available_leagues(self) -> list[ProviderLeague]:
        """Fetch and parse the FotMob allLeagues catalogue."""
        try:
            payload = self.client.get_json("allLeagues")
        except FootballDataHttpError:
            payload = self.client.post_json("allLeagues")
        return parse_fotmob_all_leagues(payload)

    def fetch_league_seasons(self, provider_league_id: str) -> list[ProviderSeason]:
        """Fetch seasons for a FotMob league id."""
        payload = self.client.get_json(
            "leagues",
            params={"id": provider_league_id},
        )
        return parse_fotmob_seasons(payload, provider_league_id)

    def fetch_season_matches(
        self,
        provider_league_id: str,
        provider_season_id: str,
        *,
        country_code: str | None = None,
    ) -> list[ProviderMatch]:
        """Fetch fixtures via leagues?id&season&ccode3."""
        if not country_code:
            raise ValueError(
                f"FotMob leagues require ccode3 (league_id={provider_league_id})"
            )
        payload = self.client.get_json(
            "leagues",
            params={
                "id": provider_league_id,
                "season": provider_season_id,
                "ccode3": country_code,
            },
        )
        return parse_fotmob_matches(payload, provider_league_id, provider_season_id)

    def fetch_match_details(self, provider_match_id: str) -> ProviderMatchDetails:
        """Fetch and parse FotMob matchDetails."""
        payload = self.client.get_json(
            "matchDetails",
            params={"matchId": provider_match_id},
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected FotMob match details payload for {provider_match_id}")
        return parse_fotmob_match_details(payload)

    def fetch_match_shots(self, provider_match_id: str) -> list[ProviderShot]:
        """Return shots from FotMob match details."""
        details = self.fetch_match_details(provider_match_id)
        return details.shots
