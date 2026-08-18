"""Collect unique team names per data source and upsert Team rows CSV-first."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.orm import Session

from data_sources.api_football_client import (
    APIFootballClient,
    get_team_names_by_league,
)
from data_sources.data_collector import refresh_league_codes
from data_sources.entity_resolver import EntityResolver
from data_sources.football_data.providers.fotmob import FotMobProvider
from data_sources.football_data.service import _ccode_from_all_leagues_csv
from data_sources.football_data_tournaments_xslx_provider import (
    FootballDataTournamentProvider,
)
from data_sources.football_data_uk_xlsx_provider import FootballDataUKProvider
from data_sources.soccerdata_espn_client import SoccerDataEspnClient
from data_sources.soccerdata_league_mapping import (
    espn_league_for_code,
    soccerdata_league_for_code,
)
from data_sources.soccerdata_match_history_client import SoccerDataMatchHistoryClient
from objects.repositories.external_entity_mapping_repository import ExternalEntityMappingRepository
from objects.repositories.league_repository import LeagueRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.historical_match import HistoricalMatchDraft
from utils.common import get_season_rev

logger = logging.getLogger(__name__)

SOURCE_FOTMOB = "fotmob"
SOURCE_FOOTBALL_DATA = "football-data.co.uk"
SOURCE_ESPN = "espn"
SOURCE_SOCCERDATA = "soccerdata"
SOURCE_API_FOOTBALL = "api-football"

DEFAULT_SEASONS = ["2223", "2324", "2425", "2526"]


@dataclass(frozen=True)
class SourceTeam:
    """One provider team spelling, optionally tied to a football-data league code."""

    external_id: str
    name: str
    league_code: str | None = None


def fetch_team_names(
    leagues: list[str] | None = None,
    seasons: list[str] | str | None = None,
    *,
    session: Session,
    config: DataSourceConfig | None = None,
) -> dict[str, list[str]]:
    """Fetch names per source, upsert CSV teams first, then map other sources.

    League codes come from ``refresh_league_codes``. Pass ``leagues`` to
    partition a subset; pass ``None`` for the full refresh set. ``seasons``
    are YYXX codes; default is 2223–2526. ``session`` is required so teams
    can be upserted.
    """
    config = config or DataSourceConfig()

    season_list = _normalize_seasons(seasons)

    source_codes = refresh_league_codes(session, leagues)

    csv_teams = _teams_from_csv(
        source_codes["football-data.co.uk"], season_list, config
    )
    tournament_teams = _teams_from_tournaments(source_codes["tournaments"], config)
    football_data_teams = csv_teams + tournament_teams
    _upsert_source_teams(
        session, SOURCE_FOOTBALL_DATA, football_data_teams, config
    )
    session.commit()

    soccerdata_teams = _teams_from_soccerdata(
        source_codes["soccerdata"], season_list
    )
    _upsert_source_teams(session, SOURCE_SOCCERDATA, soccerdata_teams, config)
    session.commit()

    espn_teams = _teams_from_espn(source_codes["espn"], season_list)
    _upsert_source_teams(session, SOURCE_ESPN, espn_teams, config)
    session.commit()

    api_football_teams = _teams_from_api_football(
        source_codes["api-football"], season_list
    )
    _upsert_source_teams(session, SOURCE_API_FOOTBALL, api_football_teams, config)
    session.commit()
    fotmob_teams = _teams_from_fotmob(
        source_codes["fotmob"], season_list, session, config
    )
    _upsert_source_teams(session, SOURCE_FOTMOB, fotmob_teams, config)
    session.commit()
    return {
        SOURCE_FOTMOB: _names_from_teams(fotmob_teams),
        #SOURCE_FOOTBALL_DATA: _names_from_teams(football_data_teams),
        #SOURCE_ESPN: _names_from_teams(espn_teams),
        #SOURCE_SOCCERDATA: _names_from_teams(soccerdata_teams),
        #SOURCE_API_FOOTBALL: _names_from_teams(api_football_teams),
    }

def _sorted_unique(names: Iterable[str]) -> list[str]:
    return sorted({name.strip() for name in names if name and str(name).strip()})


def _normalize_seasons(seasons: list[str] | str | None) -> list[str]:
    if seasons is None:
        return list(DEFAULT_SEASONS)
    if isinstance(seasons, str):
        return [seasons]
    return list(seasons)


def _names_from_teams(teams: list[SourceTeam]) -> list[str]:
    return _sorted_unique(team.name for team in teams)


def season_code_to_label(season: str) -> str:
    """Convert football-data season code 2425 to FotMob label 2024/2025."""
    year = int(get_season_rev(season))
    return f"{year}/{year + 1}"


def _upsert_source_teams(
    session: Session,
    provider: str,
    teams: list[SourceTeam],
    config: DataSourceConfig,
) -> None:
    if not teams:
        return
    league_repo = LeagueRepository(session)
    resolver = EntityResolver(session, config=config, provider=provider)
    seen: set[tuple[str, str, str | None]] = set()
    for team in teams:
        key = (team.external_id, team.name, team.league_code)
        if key in seen:
            continue
        seen.add(key)
        league_id = None
        if team.league_code:
            league = league_repo.ensure_from_code(team.league_code)
            if league is not None:
                league_id = league.id
        resolver.resolve_team(
            provider_team_id=team.external_id,
            provider_team_name=team.name,
            league_id=league_id,
            create_if_missing=True,
        )


def _teams_from_drafts(drafts: list[HistoricalMatchDraft]) -> list[SourceTeam]:
    teams: list[SourceTeam] = []
    seen: set[tuple[str, str]] = set()
    for draft in drafts:
        for name in (draft.home_team, draft.away_team):
            cleaned = name.strip()
            if not cleaned:
                continue
            key = (draft.league, cleaned)
            if key in seen:
                continue
            seen.add(key)
            teams.append(
                SourceTeam(
                    external_id=cleaned,
                    name=cleaned,
                    league_code=draft.league,
                )
            )
    return teams


def _teams_from_csv(
    leagues: list[str],
    seasons: list[str],
    config: DataSourceConfig,
) -> list[SourceTeam]:
    if not leagues:
        return []
    try:
        drafts = FootballDataUKProvider(config).fetch_historical_matches(
            leagues, seasons
        )
    except Exception:
        logger.exception("Failed fetching football-data CSV team names")
        return []
    return _teams_from_drafts(drafts)


def _teams_from_tournaments(
    tournaments: list[str],
    config: DataSourceConfig,
) -> list[SourceTeam]:
    if not tournaments:
        return []
    try:
        drafts = FootballDataTournamentProvider(config).fetch_historical_matches(
            tournaments
        )
    except Exception:
        logger.exception("Failed fetching tournament team names")
        return []
    return _teams_from_drafts(drafts)


def _teams_from_match_rows(
    rows: list[dict],
    league_code: str,
) -> list[SourceTeam]:
    teams: list[SourceTeam] = []
    seen: set[str] = set()
    for row in rows:
        home = row.get("home_team") or row.get("HomeTeam")
        away = row.get("away_team") or row.get("AwayTeam")
        for name in (home, away):
            if not name:
                continue
            cleaned = str(name).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            teams.append(
                SourceTeam(
                    external_id=cleaned,
                    name=cleaned,
                    league_code=league_code,
                )
            )
    return teams


def _teams_from_soccerdata(leagues: list[str], seasons: list[str]) -> list[SourceTeam]:
    teams: list[SourceTeam] = []
    client = SoccerDataMatchHistoryClient()
    for code in leagues:
        soccerdata_league = soccerdata_league_for_code(code)
        if soccerdata_league is None:
            logger.info("Skipping soccerdata for unmapped league %s", code)
            continue
        try:
            rows = client.fetch_matches_for_seasons(soccerdata_league, seasons)
        except Exception:
            logger.exception(
                "Failed fetching soccerdata team names for league %s", code
            )
            continue
        teams.extend(_teams_from_match_rows(rows, code))
    return teams


def _teams_from_espn(leagues: list[str], seasons: list[str]) -> list[SourceTeam]:
    teams: list[SourceTeam] = []
    client = SoccerDataEspnClient()
    for code in leagues:
        espn_league = espn_league_for_code(code)
        if espn_league is None:
            logger.info("Skipping ESPN for unmapped league %s", code)
            continue
        try:
            rows = client.fetch_matches_for_seasons(espn_league, seasons)
        except Exception:
            logger.exception("Failed fetching ESPN team names for league %s", code)
            continue
        teams.extend(_teams_from_match_rows(rows, code))
    return teams


def _teams_from_fotmob(
    leagues: list[str],
    seasons: list[str],
    session: Session,
    config: DataSourceConfig,
) -> list[SourceTeam]:
    teams: list[SourceTeam] = []
    resolver = EntityResolver(session, config=config, provider="fotmob")
    mapping_repo = ExternalEntityMappingRepository(session)
    league_repo = LeagueRepository(session)
    provider = FotMobProvider(config=config)
    try:
        for code in leagues:
            try:
                mapping = mapping_repo.get_by_external(provider='fotmob', entity_type='league', external_entity_id=code)
                league = league_repo.get(mapping.internal_entity_id)
                if league is None:
                    logger.info("Skipping FotMob for unknown league code %s", code)
                    continue
                provider_league_id = resolver.resolve_provider_league_id(league.id)
                if not provider_league_id:
                    logger.info("Skipping FotMob for unmapped league %s", code)
                    continue
                country_code = _fotmob_ccode(
                    resolver, league.id, provider_league_id
                )
                if not country_code:
                    logger.warning(
                        "Skipping FotMob for league %s: missing ccode", code
                    )
                    continue
                seen: set[str] = set()
                for season in seasons:
                    try:
                        matches = provider.fetch_season_matches(
                            provider_league_id,
                            season_code_to_label(season),
                            country_code=country_code,
                        )
                    except Exception:
                        logger.exception(
                            "Failed fetching FotMob team names for league %s season %s",
                            code,
                            season,
                        )
                        continue
                    for match in matches:
                        for external_id, name in (
                            (match.home_team_id, match.home_team_name),
                            (match.away_team_id, match.away_team_name),
                        ):
                            cleaned = (name or "").strip()
                            if not cleaned:
                                continue
                            team_id = (
                                str(external_id).strip() if external_id else cleaned
                            )
                            key = f"{code}:{team_id}"
                            if key in seen:
                                continue
                            seen.add(key)
                            teams.append(
                                SourceTeam(
                                    external_id=team_id,
                                    name=cleaned,
                                    league_code=code,
                                )
                            )
            except Exception:
                logger.exception("Failed fetching FotMob team names for league %s", code)
    finally:
        provider.close()
    return teams


def _fotmob_ccode(
    resolver: EntityResolver,
    league_id: int,
    provider_league_id: str,
) -> str | None:
    mapping = resolver.mapping_repo.get_by_internal(
        provider="fotmob",
        entity_type="league",
        internal_entity_id=league_id,
    )
    if mapping is not None and mapping.metadata_json:
        ccode = mapping.metadata_json.get("ccode")
        if ccode:
            return str(ccode)
    return _ccode_from_all_leagues_csv(provider_league_id)


def _teams_from_api_football(
    league_ids: list[int], seasons: list[str]
) -> list[SourceTeam]:
    if not league_ids:
        return []
    teams: list[SourceTeam] = []
    try:
        client = APIFootballClient()
    except Exception:
        logger.exception("Failed creating API-Football client")
        return []
    for league_id in league_ids:
        for season in seasons:
            try:
                for external_id, name in get_team_names_by_league(
                    client, league_id, season
                ):
                    teams.append(SourceTeam(external_id=external_id, name=name))
            except Exception:
                logger.exception(
                    "Failed fetching API-Football team names for league_id=%s season=%s",
                    league_id,
                    season,
                )
    return teams
