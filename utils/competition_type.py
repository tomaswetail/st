"""Classify historical match league codes into competition types."""

from __future__ import annotations


DOMESTIC_CUP_CODES = frozenset({"ENG-FA Cup"})
TOURNAMENT_CODE_PREFIXES = ("WC",)


def is_tournament_code(league_code: str) -> bool:
    """True for international tournament codes (e.g. World Cup WC2022)."""
    code = league_code.strip().upper()
    return any(code.startswith(prefix) for prefix in TOURNAMENT_CODE_PREFIXES)


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
    return False, False, False


def is_league_match(league_code: str) -> bool:
    """True when the match is a normal league fixture (reference category)."""
    is_domestic, is_international, is_friendly = competition_type_flags(league_code)
    return not (is_domestic or is_international or is_friendly)
