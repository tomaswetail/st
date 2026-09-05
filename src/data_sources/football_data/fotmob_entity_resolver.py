from __future__ import annotations

import csv
from pathlib import Path
from difflib import SequenceMatcher

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from src.data_sources.football_data.providers.fotmob import FotMobProvider
from src.objects.repositories.league_repository import LeagueRepository
from src.objects.repositories.team_repository import TeamRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.provider_dtos import ProviderTeam
from src.utils.common import (
    API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING,
    FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE,
)
from src.utils.team_mappings import FOTMOB_TO_API_FOOTBALL_TEAMS
from src.utils.team_name_matcher import _load_aliases, normalize_team_name

EXTRA = {
    8814: "BRA",
    132: "ENG",
    117: "ENG",
    8944: "ENG",
    8947: "ENG",
    10176: "ENG",
    9084: "ENG",
}

# API-Football team id → FotMob team id (last FotMob id wins on collisions).
_API_FOOTBALL_TO_FOTMOB_TEAMS = {
    api_id: fotmob_id for fotmob_id, api_id in FOTMOB_TO_API_FOOTBALL_TEAMS.items()
}


def _get_team_names(provider: FotMobProvider | None = None) -> list[str]:
    country_codes = {}

    for k, v in API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.items():
        if v:
            country_codes[v] = FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE[k]

    for k, v in EXTRA.items():
        country_codes[k] = v
    fotmob_team_ids = list(API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.values()) + list(
        EXTRA.keys()
    )
    fotmob = provider or FotMobProvider()
    teams = fotmob.fetch_teams_for_leagues(
        fotmob_team_ids,
        country_codes=country_codes,
    )
    return sorted([t.name for t in teams])


class FotMobEntityResolver:
    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        provider: FotMobProvider | None = None,
    ) -> None:
        """Wire repositories and team-name aliases for a provider."""
        self.session = session
        self.config = config or DataSourceConfig()
        self.team_repo = TeamRepository(session)
        self.league_repo = LeagueRepository(session)
        self.provider = provider or FotMobProvider(config=self.config)
        _aliases = _load_aliases()
        self._aliases = {value: key for key, value in _aliases.items()}
        self._team_name_cache: list[str] | None = None
        self.fotmob_team_names = _get_team_names(self.provider)

    def _find_by_normalized_name(self, name: str) -> str | None:
        """Find a candidate matching the normalized team name."""
        lookup = {
            normalize_team_name(candidate): candidate
            for candidate in self.fotmob_team_names
        }
        return lookup.get(normalize_team_name(name))

    def _append_team(
        self,
        team_name: str,
    ) -> None:
        """Append an unresolved team to CSV unless it is already listed."""
        csv_path = Path(self.config.missing_teams_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                target = normalize_team_name(team_name)
                for row in reader:
                    existing_name = row.get("team_name", "")
                    if normalize_team_name(existing_name) == target:
                        return

        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        row = {
            "key": "",
            "team_name": team_name,
        }
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("key", "team_name"))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _best_search_match(self, team_name: str) -> ProviderTeam | None:
        """Pick the best FotMob search hit for *team_name*, if any."""
        try:
            hits = self.provider.search_teams(team_name)
        except Exception:
            return None
        if not hits:
            return None

        target = normalize_team_name(team_name)
        for hit in hits:
            if normalize_team_name(hit.name) == target:
                return hit

        threshold = self.config.fuzzy_match_threshold / 100
        best_hit: ProviderTeam | None = None
        best_score = 0.0
        for hit in hits:
            candidate = normalize_team_name(hit.name)
            scores = [
                SequenceMatcher(None, target, candidate).ratio(),
                fuzz.ratio(target, candidate) / 100,
                fuzz.token_sort_ratio(target, candidate) / 100,
                fuzz.token_set_ratio(target, candidate) / 100,
            ]
            score = max(scores)
            if score > best_score:
                best_score = score
                best_hit = hit
        if best_hit is not None and best_score >= threshold:
            return best_hit
        return None

    def resolve_team(self, team_name, team_external_id: int) -> str | int:
        fotmob_id = _API_FOOTBALL_TO_FOTMOB_TEAMS.get(int(team_external_id))
        if fotmob_id is not None:
            return fotmob_id

        alias = self._aliases.get(team_name)
        if alias:
            return alias

        exact = self._find_by_normalized_name(team_name)
        if exact is not None:
            spellings_differ = team_name.strip().lower() != exact.strip().lower()
            if not spellings_differ:
                return exact



        threshold = self.config.fuzzy_match_threshold / 100
        best_name_s = ""
        best_score_s = 0.0
        best_name = ""
        best_score = 0.0
        target = normalize_team_name(team_name)
        for candidate in self.fotmob_team_names:
            score = SequenceMatcher(
                None, target, normalize_team_name(candidate)
            ).ratio()
            if score > best_score_s:
                best_score_s = score
                best_name_s = candidate
        if best_name_s and best_score_s >= threshold:
            return best_name_s

        for candidate in self.fotmob_team_names:
            scores = [
                fuzz.ratio(target, candidate),
                fuzz.token_sort_ratio(target, candidate),
                fuzz.token_set_ratio(target, candidate),
            ]

            score = max(scores) / 100
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_name and best_score >= threshold:
            return best_name

        search_hit = self._best_search_match(team_name)
        if search_hit is not None:
            try:
                return int(search_hit.provider_team_id)
            except (TypeError, ValueError):
                return search_hit.name

        self._append_team(team_name)
        return team_name
