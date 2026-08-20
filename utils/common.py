import re
from enum import Enum
from typing import Literal

LEAGUES_EXTERNAL_IDS = ["39","40","41","42","43","179","180","183","184","78","79","135","136","140","141","61","62","88","144","94","203","197","113","114","563","564","593","597","595","592","594","549","736","45"]

FOTMOB_TO_API_FOOTBALL_LEAGUE_MAPPING = {
    39: 47,       # Premier League
    40: 48,       # Championship
    41: 108,      # League One
    42: 109,      # League Two
    43: 117,      # National League
    45: 132,      # FA Cup
    61: 53,       # Ligue 1
    62: 110,      # Ligue 2
    78: 54,       # Bundesliga -> 1. Bundesliga
    79: 146,      # 2. Bundesliga
    88: 57,       # Eredivisie
    94: 61,       # Primeira Liga -> Liga Portugal
    113: 67,      # Allsvenskan
    114: 168,     # Superettan
    135: 55,      # Serie A
    136: 86,      # Serie B
    140: 87,      # La Liga -> LaLiga
    141: 140,     # Segunda División -> LaLiga2
    144: 40,      # Jupiler Pro League -> First Division A
    179: 64,      # Premiership
    180: 123,     # Championship
    183: 124,     # League One
    184: 125,     # League Two
    197: 135,     # Super League 1
    203: 71,      # Süper Lig -> Super Lig
    549: 9089,    # Damallsvenskan -> Damallsvenskan (W)
    563: 169,     # Ettan - Norra -> Ettan
    564: 169,     # Ettan - Södra -> Ettan

    592: None,    # Division 2 - Norra Götaland
    593: None,    # Division 2 - Norra Svealand
    594: None,    # Division 2 - Norrland
    595: None,    # Division 2 - Södra Svealand
    597: None,    # Division 2 - Södra Götaland
    736: None,    # Elitettan
}

API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING = {
    47: 39,       # Premier League
    48: 40,       # Championship
    108: 41,      # League One
    109: 42,      # League Two
    117: 43,      # National League
    132: 45,      # FA Cup
    53: 61,       # Ligue 1
    110: 62,      # Ligue 2
    54: 78,       # 1. Bundesliga -> Bundesliga
    146: 79,      # 2. Bundesliga
    57: 88,       # Eredivisie
    61: 94,       # Liga Portugal -> Primeira Liga
    67: 113,      # Allsvenskan
    168: 114,     # Superettan
    55: 135,      # Serie A
    86: 136,      # Serie B
    87: 140,      # LaLiga -> La Liga
    140: 141,     # LaLiga2 -> Segunda División
    40: 144,      # First Division A -> Jupiler Pro League
    64: 179,      # Premiership
    123: 180,     # Championship
    124: 183,     # League One
    125: 184,     # League Two
    135: 197,     # Super League 1
    71: 203,      # Super Lig -> Süper Lig
    9089: 549,    # Damallsvenskan (W) -> Damallsvenskan

    169: [563, 564],  # Ettan - Norra / Ettan - Södra

    None: [592, 593, 594, 595, 597, 736],
}


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
