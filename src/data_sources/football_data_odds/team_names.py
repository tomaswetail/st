"""Map football-data.co.uk archive team names onto canonical names.

``config/team_aliases.json`` mostly maps canonical/Svenska names → archive
short forms. EntityResolver looks up ``aliases.get(provider_team_name)``, so
archive names like ``Goteborg`` miss those keys. This module inverts expanding
aliases and overlays known football-data.co.uk short names without changing
EntityResolver's Svenska Spel / xG behavior.
"""

from __future__ import annotations

from src.utils.team_name_matcher import _load_aliases

# Known football-data.co.uk spellings → names stored on our teams/fixtures.
# Applied before EntityResolver so broken aliases (e.g. Man United → Bristol
# Rovers) cannot hijack this ingest.
FOOTBALL_DATA_CO_UK_TO_CANONICAL = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Weds": "Sheffield Wednesday",
    "Sheff Wed": "Sheffield Wednesday",
    "Sheff Utd": "Sheffield United",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Brighton": "Brighton and Hove Albion",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    #"Norwich": "Norwich City",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Hull": "Hull City",
    "Stoke": "Stoke City",
    "Ipswich": "Ipswich Town",
    "Derby": "Derby County",
    "Coventry": "Coventry City",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Huddersfield": "Huddersfield Town",
    "Peterboro": "Peterborough United",
    "Plymouth": "Plymouth Argyle",
    "Charlton": "Charlton Athletic",
    "Preston": "Preston North End",
    "MK Dons": "Milton Keynes Dons",
    "Bristol Rvs": "Bristol Rovers",
    "Forest Green": "Forest Green Rovers",
    "Accrington": "Accrington Stanley",
    "Dag and Red": "Dagenham and Redbridge",
    "Nott County": "Notts County",
    "Notts County": "Notts County",
    "AFC Wimbledon": "AFC Wimbledon",
    "Wimbledon": "AFC Wimbledon",
}


def canonical_team_name(archive_name: str) -> str:
    """Return a canonical/Svenska-style name for EntityResolver."""
    trimmed = archive_name.strip()
    if not trimmed:
        return trimmed
    overlay = FOOTBALL_DATA_CO_UK_TO_CANONICAL.get(trimmed)
    if overlay:
        return overlay
    aliases = _load_aliases()
    reverse = expanding_reverse_aliases(aliases)
    mapped = reverse.get(trimmed)
    if mapped:
        return mapped
    forward = aliases.get(trimmed)
    if forward:
        return forward
    return trimmed


def expanding_reverse_aliases(aliases: dict[str, str]) -> dict[str, str]:
    """Invert canonical → archive aliases only when the reverse expands the name."""
    reverse: dict[str, str] = {}
    for canonical, archive in aliases.items():
        archive_name = archive.strip()
        canonical_name = canonical.strip()
        if not archive_name or archive_name == canonical_name:
            continue
        if len(canonical_name) < len(archive_name):
            continue
        existing = reverse.get(archive_name)
        if existing is None or len(canonical_name) > len(existing):
            reverse[archive_name] = canonical_name
    return reverse
