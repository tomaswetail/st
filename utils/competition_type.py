"""Classify competitions using API-Football league metadata."""

from __future__ import annotations

INTERNATIONAL_COUNTRY_NAMES = frozenset({"world"})


def competition_type_flags(
    *,
    league_type: str,
    country_name: str,
) -> tuple[bool, bool, bool]:
    """Return (is_domestic_cup, is_international_cup, is_friendly)."""
    normalized_type = league_type.strip().lower()
    normalized_country = country_name.strip().lower()

    if normalized_type == "friendly":
        return False, False, True
    if normalized_type == "cup":
        if normalized_country in INTERNATIONAL_COUNTRY_NAMES:
            return False, True, False
        return True, False, False
    return False, False, False


def is_league_match(*, league_type: str, country_name: str) -> bool:
    """True when the match is a normal league fixture (reference category)."""
    is_domestic, is_international, is_friendly = competition_type_flags(
        league_type=league_type,
        country_name=country_name,
    )
    return not (is_domestic or is_international or is_friendly)
