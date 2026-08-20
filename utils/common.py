import re
from enum import Enum
from typing import Literal

LEAGUES_EXTERNAL_IDS = ["39","40","41","42","43","179","180","183","184","78","79","135","136","140","141","61","62","88","144","94","203","197","113","114","563","564","593","597","595","592","594","549","736","45"]


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
    "England FA Cup": "ENG-FA Cup",
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
    "SWE_SE": "Sweden",
    "SWE_ETTAN_NORRA": "Sweden",
    "SWE_ETTAN_SODRA": "Sweden",
    "SWE_DIV2_NORRA_SVEALAND": "Sweden",
    "SWE_DIV2_SODRA_GOTALAND": "Sweden",
    "SWE_DIV2_SODRA_SVEALAND": "Sweden",
    "SWE_DIV2_NORRA_GOTALAND": "Sweden",
    "SWE_DIV2_NORRA_NORRLAND": "Sweden",
    "SWE_DIV2_OSTRA_GOTALAND": "Sweden",
    "SWE_DAM": "Sweden",
    "SWE_ELITETAN": "Sweden",
    "SWE_DIV1_NORRA": "Sweden",
    "SWE_DIV1_SODRA": "Sweden",
    "SWE_U19": "Sweden",
    "SWE_U17": "Sweden",
    "ENG-FA Cup": "England",
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
        "ENG-FA Cup",
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


def ensure_unit_probabilities(
    market_probabilities: dict[str, float | None],
) -> dict[str, float | None]:
    """Return 1X2 values on the 0–1 scale. Reject mixed 0–1 / percentage inputs."""
    present_values = [
        value for value in market_probabilities.values() if value is not None
    ]
    if not present_values:
        return dict(market_probabilities)
    if any(value < 0 for value in present_values):
        raise ValueError("Market probabilities must be non-negative")

    all_unit_scale = all(value <= 1 for value in present_values)
    all_percent_scale = any(value > 1 for value in present_values) and all(
        value == 0 or value > 1 for value in present_values
    )
    if all_unit_scale:
        return dict(market_probabilities)
    if all_percent_scale:
        if any(value > 100 for value in present_values):
            raise ValueError("Percentage-style market probabilities must be <= 100")
        return {
            key: None if value is None else value / 100.0
            for key, value in market_probabilities.items()
        }
    raise ValueError("Mixed 0–1 and percentage market probability scales")

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

def strtr(s, repl):
  pattern = '|'.join(map(re.escape, sorted(repl, key=len, reverse=True)))
  return re.sub(pattern, lambda m: repl[m.group()], s)

def parse_swedish_decimal(value: str | float | int | None) -> float | None:
    """Parse Svenska Spel decimal strings like '3,10' or '1,00'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None

def sanitize_string(name: str):

  chars = {
    ',': '-',
    '.': '-',
    '/':'-',
    ' ':'-',
    '_':'-',
    'ª':'a',
    'º':'o',
    'À':'A',
    'Á':'A',
    'Â':'A',
    'Ã':'A',
    'Ä':'A',
    'Å':'A',
    'Æ':'AE',
    'Ç':'C',
    'È':'E',
    'É':'E',
    'Ê':'E',
    'Ë':'E',
    'Ì':'I',
    'Í':'I',
    'Î':'I',
    'Ï':'I',
    'Ð':'D',
    'Ñ':'N',
    'Ò':'O',
    'Ó':'O',
    'Ô':'O',
    'Õ':'O',
    'Ö':'O',
    'Ù':'U',
    'Ú':'U',
    'Û':'U',
    'Ü':'U',
    'Ý':'Y',
    'Þ':'TH',
    'ß':'s',
    'à':'a',
    'á':'a',
    'â':'a',
    'ã':'a',
    'ä':'a',
    'å':'a',
    'æ':'ae',
    'ç':'c',
    'è':'e',
    'é':'e',
    'ê':'e',
    'ë':'e',
    'ì':'i',
    'í':'i',
    'î':'i',
    'ï':'i',
    'ð':'d',
    'ñ':'n',
    'ò':'o',
    'ó':'o',
    'ô':'o',
    'õ':'o',
    'ö':'o',
    'ø':'o',
    'ù':'u',
    'ú':'u',
    'û':'u',
    'ü':'u',
    'ý':'y',
    'þ':'th',
    'ÿ':'y',
    'Ø':'O',
    'Ā':'A',
    'ā':'a',
    'Ă':'A',
    'ă':'a',
    'Ą':'A',
    'ą':'a',
    'Ć':'C',
    'ć':'c',
    'Ĉ':'C',
    'ĉ':'c',
    'Ċ':'C',
    'ċ':'c',
    'Č':'C',
    'č':'c',
    'Ď':'D',
    'ď':'d',
    'Đ':'D',
    'đ':'d',
    'Ē':'E',
    'ē':'e',
    'Ĕ':'E',
    'ĕ':'e',
    'Ė':'E',
    'ė':'e',
    'Ę':'E',
    'ę':'e',
    'Ě':'E',
    'ě':'e',
    'Ĝ':'G',
    'ĝ':'g',
    'Ğ':'G',
    'ğ':'g',
    'Ġ':'G',
    'ġ':'g',
    'Ģ':'G',
    'ģ':'g',
    'Ĥ':'H',
    'ĥ':'h',
    'Ħ':'H',
    'ħ':'h',
    'Ĩ':'I',
    'ĩ':'i',
    'Ī':'I',
    'ī':'i',
    'Ĭ':'I',
    'ĭ':'i',
    'Į':'I',
    'į':'i',
    'İ':'I',
    'ı':'i',
    'Ĳ':'IJ',
    'ĳ':'ij',
    'Ĵ':'J',
    'ĵ':'j',
    'Ķ':'K',
    'ķ':'k',
    'ĸ':'k',
    'Ĺ':'L',
    'ĺ':'l',
    'Ļ':'L',
    'ļ':'l',
    'Ľ':'L',
    'ľ':'l',
    'Ŀ':'L',
    'ŀ':'l',
    'Ł':'L',
    'ł':'l',
    'Ń':'N',
    'ń':'n',
    'Ņ':'N',
    'ņ':'n',
    'Ň':'N',
    'ň':'n',
    'ŉ':'n',
    'Ŋ':'N',
    'ŋ':'n',
    'Ō':'O',
    'ō':'o',
    'Ŏ':'O',
    'ŏ':'o',
    'Ő':'O',
    'ő':'o',
    'Œ':'OE',
    'œ':'oe',
    'Ŕ':'R',
    'ŕ':'r',
    'Ŗ':'R',
    'ŗ':'r',
    'Ř':'R',
    'ř':'r',
    'Ś':'S',
    'ś':'s',
    'Ŝ':'S',
    'ŝ':'s',
    'Ş':'S',
    'ş':'s',
    'Š':'S',
    'š':'s',
    'Ţ':'T',
    'ţ':'t',
    'Ť':'T',
    'ť':'t',
    'Ŧ':'T',
    'ŧ':'t',
    'Ũ':'U',
    'ũ':'u',
    'Ū':'U',
    'ū':'u',
    'Ŭ':'U',
    'ŭ':'u',
    'Ů':'U',
    'ů':'u',
    'Ű':'U',
    'ű':'u',
    'Ų':'U',
    'ų':'u',
    'Ŵ':'W',
    'ŵ':'w',
    'Ŷ':'Y',
    'ŷ':'y',
    'Ÿ':'Y',
    'Ź':'Z',
    'ź':'z',
    'Ż':'Z',
    'ż':'z',
    'Ž':'Z',
    'ž':'z',
    'ſ':'s',
    'Ș':'S',
    'ș':'s',
    'Ț':'T',
    'ț':'t',
    '€':'E',
    '£':'',
    'Ơ':'O',
    'ơ':'o',
    'Ư':'U',
    'ư':'u',
    'Ầ':'A',
    'ầ':'a',
    'Ằ':'A',
    'ằ':'a',
    'Ề':'E',
    'ề':'e',
    'Ồ':'O',
    'ồ':'o',
    'Ờ':'O',
    'ờ':'o',
    'Ừ':'U',
    'ừ':'u',
    'Ỳ':'Y',
    'ỳ':'y',
    'Ả':'A',
    'ả':'a',
    'Ẩ':'A',
    'ẩ':'a',
    'Ẳ':'A',
    'ẳ':'a',
    'Ẻ':'E',
    'ẻ':'e',
    'Ể':'E',
    'ể':'e',
    'Ỉ':'I',
    'ỉ':'i',
    'Ỏ':'O',
    'ỏ':'o',
    'Ổ':'O',
    'ổ':'o',
    'Ở':'O',
    'ở':'o',
    'Ủ':'U',
    'ủ':'u',
    'Ử':'U',
    'ử':'u',
    'Ỷ':'Y',
    'ỷ':'y',
    'Ẫ':'A',
    'ẫ':'a',
    'Ẵ':'A',
    'ẵ':'a',
    'Ẽ':'E',
    'ẽ':'e',
    'Ễ':'E',
    'ễ':'e',
    'Ỗ':'O',
    'ỗ':'o',
    'Ỡ':'O',
    'ỡ':'o',
    'Ữ':'U',
    'ữ':'u',
    'Ỹ':'Y',
    'ỹ':'y',
    'Ấ':'A',
    'ấ':'a',
    'Ắ':'A',
    'ắ':'a',
    'Ế':'E',
    'ế':'e',
    'Ố':'O',
    'ố':'o',
    'Ớ':'O',
    'ớ':'o',
    'Ứ':'U',
    'ứ':'u',
    'Ạ':'A',
    'ạ':'a',
    'Ậ':'A',
    'ậ':'a',
    'Ặ':'A',
    'ặ':'a',
    'Ẹ':'E',
    'ẹ':'e',
    'Ệ':'E',
    'ệ':'e',
    'Ị':'I',
    'ị':'i',
    'Ọ':'O',
    'ọ':'o',
    'Ộ':'O',
    'ộ':'o',
    'Ợ':'O',
    'ợ':'o',
    'Ụ':'U',
    'ụ':'u',
    'Ự':'U',
    'ự':'u',
    'Ỵ':'Y',
    'ỵ':'y',
    'ɑ':'a',
    'Ǖ':'U',
    'ǖ':'u',
    'Ǘ':'U',
    'ǘ':'u',
    'Ǎ':'A',
    'ǎ':'a',
    'Ǐ':'I',
    'ǐ':'i',
    'Ǒ':'O',
    'ǒ':'o',
    'Ǔ':'U',
    'ǔ':'u',
    'Ǚ':'U',
    'ǚ':'u',
    'Ǜ':'U',
    'ǜ':'u',
    '&':'-',
    ' - ':'-',
    ')':'-',
    '(':'-',
  }
  name = strtr(name, chars).lower()
  name = name.rstrip('-')
  return name
