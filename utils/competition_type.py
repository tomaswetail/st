"""Classify historical match league codes into competition types."""

from __future__ import annotations

from data_sources.football_data_tournaments_xslx_provider import is_tournament_code
from utils.common import EXTRA_LEAGUE_CODES, MAIN_LEAGUE_CODES

DOMESTIC_CUP_CODES = frozenset({"ENG-FA Cup"})


def competition_type_flags(league_code: str) -> tuple[bool, bool, bool]:
    """Return (is_domestic_cup, is_international_cup, is_friendly) for a league code."""
    code = league_code.strip()
    upper = code.upper()

    if is_tournament_code(code):
        return False, True, False
    if "FRIEND" in upper:
        return False, False, True
    if code in DOMESTIC_CUP_CODES:
        return True, False, False
    if code in MAIN_LEAGUE_CODES or code in EXTRA_LEAGUE_CODES:
        return False, False, False
    return False, False, False


def is_league_match(league_code: str) -> bool:
    """True when the match is a normal league fixture (reference category)."""
    is_domestic, is_international, is_friendly = competition_type_flags(league_code)
    return not (is_domestic or is_international or is_friendly)
