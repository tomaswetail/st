from enum import Enum
from typing import Literal

MAIN_LEAGUE_CODES = frozenset(
    {
        "E0",
        "E1",
        "E2",
        "E3",
        "EC",
        "SC0",
        "SC1",
        "SC2",
        "SC3",
        "D1",
        "D2",
        "I1",
        "I2",
        "SP1",
        "SP2",
        "F1",
        "F2",
        "N1",
        "B1",
        "P1",
        "T1",
        "G1",
    }
)

LEAGUES = [
    "Premier League",
    "Championship",
    "League One",
    "League Two",
    "Allsvenskan",
    "Superettan",
    "Ettan Norra",
    "Ettan Södra",
    "Division 2 Norra Svealand",
    "Division 2 Södra Götaland",
    "Division 2 Södra Svealand",
    "Eliteserien",
    "Superligaen",
    "LaLiga"

]

LEAGUE_MAPPINGS = {
    "Premier League": "E0",
    "Championship": "E1",
    "League One": "E2",
    "League Two": "E3",
    "Allsvenskan": "SWE",
    "Superettan": "SWE_SE",
    "Ettan Norra": "SWE_ETTAN_NORRA",
    "Ettan Södra": "SWE_ETTAN_SODRA",
    "Division 2 Norra Svealand": "SWE_DIV2_NORRA_SVEALAND",
    "Division 2 Södra Götaland": "SWE_DIV2_SODRA_GOTALAND",
    "Division 2 Södra Svealand": "SWE_DIV2_SODRA_SVEALAND",
    "Eliteserien": "NOR",
    "Superligaen": "DNK",
    "LaLiga": "SP1"
}


LEAGUE_NAMES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "EC": "England National League",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "SC2": "Scotland League One",
    "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda División",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium First Division A",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Süper Lig",
    "G1": "Greece Super League",
    "SWE": "Allsvenskan",
}

LEAGUE_NAMES_REV = {
    "England Premier League": "E0",
    "England Championship": "E1",
    "England League One": "E2",
    "England League Two": "E3",
    "England National League": "EC",
    "Scotland Premiership": "SC0",
    "Scotland Championship": "SC1",
    "Scotland League One": "SC2",
    "Scotland League Two": "SC3",
    "Germany Bundesliga": "D1",
    "Germany 2. Bundesliga": "D2",
    "Italy Serie A": "I1",
    "Italy Serie B": "I2",
    "Spain La Liga":"SP1" ,
    "Spain Segunda División":"SP2",
    "France Ligue 1": "F1",
    "France Ligue 2": "F2",
    "Netherlands Eredivisie": "N1",
    "Belgium First Division A": "B1",
    "Portugal Primeira Liga": "P1",
    "Turkey Süper Lig": "T1",
    "Greece Super League": "G1",
    "Allsvenskan": "SWE",
    "Ettan Norra": "SWE"
}

LEAGUE_COUNTRIES = {
    "E0": "England",
    "E1": "England",
    "E2": "England",
    "E3": "England",
    "EC": "England",
    "SC0": "Scotland",
    "SC1": "Scotland",
    "SC2": "Scotland",
    "SC3": "Scotland",
    "D1": "Germany",
    "D2": "Germany",
    "I1": "Italy",
    "I2": "Italy",
    "SP1": "Spain",
    "SP2": "Spain",
    "F1": "France",
    "F2": "France",
    "N1": "Netherlands",
    "B1": "Belgium",
    "P1": "Portugal",
    "T1": "Turkey",
    "G1": "Greece",
    "SWE": "Sweden",
}

EXTRA_LEAGUE_CODES = frozenset(
    {
        "ARG",
        "AUT",
        "BRA",
        "CHN",
        "DNK",
        "FIN",
        "IRL",
        "JPN",
        "MEX",
        "NOR",
        "POL",
        "ROU",
        "RUS",
        "SWE",
        "SWZ",
        "USA",
    }
)

def all_football_data_league_codes() -> list[str]:
    """Return all supported main and extra Football-Data league codes."""
    return sorted(MAIN_LEAGUE_CODES | EXTRA_LEAGUE_CODES)

Outcome = Literal["1", "X", "2"]
OUTCOMES: tuple[Outcome, ...] = ("1", "X", "2")


class SignType(str, Enum):
    SPIK = "spik"
    HALV = "halv"
    HEL = "hel"


def fix_swedish_name(team_name):
    _name = swedish_to_ascii(team_name)
    if _name == 'BK Hacken':
        return 'Hacken'
    return _name

def swedish_to_ascii(text: str) -> str:
    """Convert Swedish å/ä/ö characters to ASCII-style equivalents."""
    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "Å": "A",
        "Ä": "A",
        "Ö": "O",
    }

    return "".join(replacements.get(char, char) for char in text)


def odds_to_probabilities(
    win_home: float,
    draw: float,
    win_away: float,
) -> dict[Outcome, float]:
    """Convert decimal 1X2 odds to normalized probabilities."""
    for label, odds in (("win_home", win_home), ("draw", draw), ("win_away", win_away)):
        if odds <= 0:
            raise ValueError(f"{label} odds must be positive, got {odds}")

    home = 1.0 / win_home
    draw_prob = 1.0 / draw
    away = 1.0 / win_away
    total = home + draw_prob + away

    return {
        "1": home / total,
        "X": draw_prob / total,
        "2": away / total,
    }

def probabilities_to_result(
    win_home: float,
    draw: float,
    win_away: float,
) -> dict[Outcome, float]:
    _all = [win_home, draw, win_away]
    if max(_all) == win_home:
        return '1'
    elif max(_all) == draw:
        return 'X'
    elif max(_all) == win_away:
        return '2'

def get_season_rev(season: str):
    seasons = {
        '2223': '2022',
        '2324': '2023',
        '2425': '2024',
        '2526': '2025',
    }
    return seasons[season]


def get_season(season: str):
    seasons = {
        '2022': '2223',
        '2023': '2324',
        '2024': '2425',
        '2025': '2526',
    }
    return seasons[season]
