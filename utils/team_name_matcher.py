"""Map a Svenska Spel team name to a Football-Data name."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from objects.schema.data_classes.data_sources import DataSourceConfig
from utils.common import fix_swedish_name, swedish_to_ascii

_SUFFIXES = (
    " ff",
    " if",
    " fk",
    " bk",
    " cf",
    " afc",
    " fc",
    " utd",
    " united",
    " city",
)

_ALIASES: dict[str, str] = {
    "Accrington Stanley": "Accrington",
    "Athletic Bilbao": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Bayer Leverkusen": "Leverkusen",
    "Bayern München": "Bayern Munich",
    "BK Häcken": "Hacken",
    "Bodö/Glimt": "Bodo/Glimt",
    "Boreham Wood FC": "Boreham Wood",
    "Borussia Dortmund": "Dortmund",
    "Bröndby": "Brondby",
    "Bristol Rovers": "Bristol Rvs",
    "Cambridge United": "Cambridge",
    "Celta de Vigo": "Celta",
    "Club Brügge": "Club Brugge",
    "Dagenham & Redbridge": "Dag and Red",
    "Djurgården": "Djurgarden",
    "Halmstads BK": "Halmstad",
    "Djurgårdens IF FF": "Djurgarden",
    "Dover Athletic FC": "Dover Athletic",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Espanyol": "Espanol",
    "FC Köpenhamn": "FC Copenhagen",
    "FC Nordsjälland": "Nordsjaelland",
    "Häcken": "Hacken",
    "IFK Göteborg": "Goteborg",
    "Inter Åbo": "Inter Turku",
    "Malmö FF": "Malmo FF",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Manchester C": "Man City",
    "Manchester U": "Man United",
    "Mönchengladbach": "M'gladbach",
    "New York City FC": "New York City",
    "Nordsjälland": "Nordsjaelland",
    "Nottingham": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Östersund": "Ostersunds",
    "Östersunds FK": "Ostersunds",
    "Paris Saint-Germain": "Paris SG",
    "Peterborough ": "Peterboro",
    "Peterborough": "Peterboro",
    "Queens Park Rangers": "QPR",
    "Real Sociedad": "Sociedad",
    "Royal Antwerp": "Antwerp",
    "Sheffield W": "Sheffield Weds",
    "Sporting Lissabon": "Sp Lisbon",
    "Sporting Gijón": "Sp Gijon",
    "Standard Liege": "Standard",
    "West Bromwich": "West Brom",
    "Wolverhampton": "Wolves",
    "Zenith": "Zenit",
    "Örebro SK": "Orebro",
    "IFK Norrköping": "Norrkoping",
    "Mjällby AIF": "Mjallby",
    "Helsingborgs IF": "Helsingborg",
    "Kalmar FF": "Kalmar",
    "IFK Värnamo": "Varnamo",
    "Värnamo": "Varnamo",
    "Västerås SK": "Vasteras SK",
    "GIF Sundsvall": "Sundsvall",
    "IK Sirius": "Sirius",
    "Ham-Kam": "HamKam",
    "Fredrikstad FK": "Fredrikstad",
    "Sarpsborg": "Sarpsborg 08",
    "Odds BK": "Odd",
    "Odds": "Odd",
    "St. Truidense": "St Truiden",
    "St. Mirren": "St Mirren",
    "St.Johnstone": "St Johnstone",
    "St.Mirren": "St Mirren",
    "St.Etienne": "St Etienne",
    "Saint Etienne": "St Etienne",
    "Sverige": "Sweden",
    "Tyskland": "Germany",
    "Frankrike": "France",
    "Spanien": "Spain",
    "Italien": "Italy",
    "Nederländerna": "Netherlands",
    "Belgien": "Belgium",
    "Schweiz": "Switzerland",
    "Österrike": "Austria",
    "Danmark": "Denmark",
    "Norge": "Norway",
    "Finland": "Finland",
    "Polen": "Poland",
    "Turkiet": "Turkey",
    "Ungern": "Hungary",
    "Grekland": "Greece",
    "Skottland": "Scotland",
    "Irland": "Ireland",
    "Island": "Iceland",
    "Kroatien": "Croatia",
    "Serbien": "Serbia",
    "Ukraina": "Ukraine",
    "Rumänien": "Romania",
    "Ryssland": "Russia",
    "Brasilien": "Brazil",
    "Mexiko": "Mexico",
    "Australien": "Australia",
    "Sydkorea": "South Korea",
    "Marocko": "Morocco",
    "Egypten": "Egypt",
    "Saudiarabien": "Saudi Arabia",
    "Algeriet": "Algeria",
    "Tunisien": "Tunisia",
    "Elfenbenskusten": "Ivory Coast",
    "Kamerun": "Cameroon",
}


def _normalize(name: str) -> str:
    text = fix_swedish_name(name.strip())
    text = swedish_to_ascii(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`.´]", "", text)
    text = re.sub(r"[^a-z0-9/ ]", " ", text)
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()

    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break

    return text


def normalize_team_name(name: str) -> str:
    """Public alias for team-name normalization used by entity resolution."""
    return _normalize(name)


def _load_aliases() -> dict[str, str]:
    aliases = dict(_ALIASES)
    path = DataSourceConfig().team_aliases_path
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            aliases.update({str(k): str(v) for k, v in data.items()})
    return aliases


def to_football_data_name(
    svenska_spel_name: str,
    football_data_names: list[str],
) -> str | None:
    """Return the Football-Data name for a Svenska Spel team name."""
    football_set = set(football_data_names)
    if svenska_spel_name in football_set:
        return svenska_spel_name

    aliases = _load_aliases()
    alias = aliases.get(svenska_spel_name)
    if alias and alias in football_set:
        return alias

    normalized_lookup = {_normalize(name): name for name in football_data_names}
    normalized_match = normalized_lookup.get(_normalize(svenska_spel_name))
    if normalized_match:
        return normalized_match

    threshold = DataSourceConfig().fuzzy_match_threshold / 100
    best_name = ""
    best_score = 0.0
    target = _normalize(svenska_spel_name)
    for football_name in football_data_names:
        score = SequenceMatcher(None, target, _normalize(football_name)).ratio()
        if score > best_score:
            best_score = score
            best_name = football_name

    if best_score >= threshold:
        return best_name

    return None
