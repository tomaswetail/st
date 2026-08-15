from __future__ import annotations

import logging

from soccerdata import MatchHistory

from data_sources.soccerdata_espn_client import ensure_espn_league_dict
from utils.common import all_football_data_league_codes

logger = logging.getLogger(__name__)

DEFAULT_SOCCERDATA_LEAGUES = frozenset(
    {
        "ENG-Premier League",
        "ESP-La Liga",
        "FRA-Ligue 1",
        "GER-Bundesliga",
        "ITA-Serie A",
    }
)

# Football-data / app codes → soccerdata canonical league names that ESPN supports.
FOOTBALL_DATA_CODE_TO_ESPN_LEAGUE: dict[str, str] = {
    "E0": "ENG-Premier League",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "SP1": "ESP-La Liga",
    "F1": "FRA-Ligue 1",
    "ENG-FA Cup": "ENG-FA Cup",
}

FOOTBALL_DATA_CODE_TO_SOCCERDATA_LEAGUE: dict[str, str] = {
    "E0": "ENG-Premier League",
    "E1": "ENG-Championship",
    "E2": "ENG-League One",
    "E3": "ENG-League Two",
    "EC": "ENG-National League",
    "SC0": "SCO-Premiership",
    "SC1": "SCO-Championship",
    "SC2": "SCO-League One",
    "SC3": "SCO-League Two",
    "D1": "GER-Bundesliga",
    "D2": "GER-2. Bundesliga",
    "I1": "ITA-Serie A",
    "I2": "ITA-Serie B",
    "SP1": "ESP-La Liga",
    "SP2": "ESP-Segunda Division",
    "F1": "FRA-Ligue 1",
    "F2": "FRA-Ligue 2",
    "N1": "NED-Eredivisie",
    "B1": "BEL-First Division A",
    "P1": "POR-Primeira Liga",
    "T1": "TUR-Super Lig",
    "G1": "GRC-Super League",
    "ARG": "ARG-Primera Division",
    "AUT": "AUT-Bundesliga",
    "BRA": "BRA-Serie A",
    "CHN": "CHN-Super League",
    "DNK": "DNK-Superliga",
    "FIN": "FIN-Veikkausliiga",
    "IRL": "IRL-Premier Division",
    "JPN": "JPN-J1 League",
    "MEX": "MEX-Liga MX",
    "NOR": "NOR-Eliteserien",
    "POL": "POL-Ekstraklasa",
    "ROU": "ROU-Liga I",
    "RUS": "RUS-Premier League",
    "SWE": "SWE-Allsvenskan",
    "SWZ": "SWZ-Super League",
    "USA": "USA-MLS",
}

SOCCERDATA_LEAGUE_TO_FOOTBALL_DATA_CODE: dict[str, str] = {
    soccerdata_league: code
    for code, soccerdata_league in FOOTBALL_DATA_CODE_TO_SOCCERDATA_LEAGUE.items()
}


def soccerdata_league_for_code(league_code: str) -> str | None:
    return FOOTBALL_DATA_CODE_TO_SOCCERDATA_LEAGUE.get(league_code)


def espn_league_for_code(league_code: str) -> str | None:
    ensure_espn_league_dict()
    return FOOTBALL_DATA_CODE_TO_ESPN_LEAGUE.get(league_code)


def soccerdata_league_to_code(soccerdata_league: str) -> str | None:
    return SOCCERDATA_LEAGUE_TO_FOOTBALL_DATA_CODE.get(soccerdata_league)


def soccerdata_leagues_for_codes(codes: list[str]) -> dict[str, str]:
    """Map football-data codes to soccerdata league IDs; log and skip unmapped."""
    mapped: dict[str, str] = {}
    for code in codes:
        soccerdata_league = soccerdata_league_for_code(code)
        if soccerdata_league is None:
            logger.warning("No soccerdata mapping for football-data league code %s", code)
            continue
        mapped[code] = soccerdata_league
    return mapped


def all_mapped_league_codes() -> list[str]:
    return [
        code
        for code in all_football_data_league_codes()
        if code in FOOTBALL_DATA_CODE_TO_SOCCERDATA_LEAGUE
    ]


def available_soccerdata_leagues() -> set[str]:
    return set(MatchHistory.available_leagues())


def partition_leagues_by_source(
    codes: list[str],
) -> tuple[list[str], list[str], list[str]]:
    available = available_soccerdata_leagues()
    soccerdata_codes: list[str] = []
    espn_codes: list[str] = []
    csv_codes: list[str] = []
    for code in codes:
        soccerdata_league = soccerdata_league_for_code(code)
        if soccerdata_league and soccerdata_league in available:
            soccerdata_codes.append(code)
        elif espn_league_for_code(code) is not None:
            espn_codes.append(code)
        else:
            csv_codes.append(code)

    logger.info(
        "Refreshing %d leagues via soccerdata, %d via ESPN, %d via football-data CSV",
        len(soccerdata_codes),
        len(espn_codes),
        len(csv_codes),
    )
    return soccerdata_codes, espn_codes, csv_codes
