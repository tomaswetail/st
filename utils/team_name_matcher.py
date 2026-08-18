"""Map a Svenska Spel team name to a Football-Data name."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

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
    path = DataSourceConfig().team_aliases_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


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
