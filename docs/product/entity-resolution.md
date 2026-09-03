# Purpose

Map **provider-specific** team, league, and match identifiers to **internal** database entities so data from Svenska Spel, API-Football, SofaScore, and FotMob can be joined for modeling.

# Core Concepts

- **Resolution** — process of finding or creating internal entity for a provider record
- **Method** — how resolution succeeded (`static_mapping`, `alias`, `exact`, `fuzzy`, `unresolved`, etc.)
- **Confidence** — score attached to team resolution
- **external_id** — API-Football id stored on `TeamModel` / `LeagueModel` and used for lookups

# Entities

- `TeamModel`, `LeagueModel`, `FixtureModel`
- Static maps: `config/team_aliases.json`, `utils/team_mappings.py` (`SVENSKA_SPEL_TO_API_FOOTBALL_TEAMS`, `FOTMOB_TO_API_FOOTBALL_TEAMS`)

# Primary Workflows

## Team resolution (`EntityResolver.resolve_team`)

Order varies by provider; for **svenska-spel**:
1. Static Svenska Spel id → API-Football id map
2. Lookup team by API-Football `external_id`
3. Further alias/exact/fuzzy paths (see implementation)

For **api-football** / xG providers:
1. Alias file
2. Exact normalized name
3. Fuzzy match (threshold from config)
4. Postgres duplicate helpers (`find_exact_normalized`, club affix, etc.)

Evidence: `data_sources/entity_resolver.py`, `tests/football_data/test_ingestion.py`

## Match resolution

Provider match → fixture row by date, team names, league, tolerance windows.

Evidence: `EntityResolver.resolve_match`, `tests/repositories/test_find_for_st_match.py`

# State/Lifecycle

- Teams/leagues carry `external_id` from API-Football upserts
- Unresolved teams may be logged to `missing_team_mappings.csv` / `missing_teams.csv`

# Business Rules

- **BR-007** — resolve via `EntityResolver` + team/league `external_id`
- National team Swedish names → English via `NATIONAL_TEAMS_SE_TO_EN`
- Fuzzy threshold default 85 (`fuzzy_match_threshold`)

# Permissions

None.

# Side Effects

- May create `TeamModel` when historical upsert paths allow create-on-miss
- CSV logs for unresolved matches

# Integrations

All external providers depend on this layer.

# Failure Scenarios

- Unresolved team → ST import may skip match or collect `missed_teams`
- Ambiguous same-date fixtures → warnings in `MatchResolution.warnings`

# Edge Cases

- Duplicate team merge logic tested in `tests/repositories/test_merge_duplicate_team.py`
- Team name matcher: `tests/test_team_name_matcher.py`
- Provider match id vs internal fixture id confusion (see DOMAIN.md)

# Important Tests

- `tests/football_data/test_ingestion.py` — resolution order
- `tests/repositories/test_find_for_st_match.py`
- `tests/repositories/test_merge_duplicate_team.py`
- `tests/test_team_name_matcher.py`

# Relevant Code

- `data_sources/entity_resolver.py`
- `data_sources/football_data/fotmob_entity_resolver.py`
- `utils/team_name_matcher.py`
- `utils/team_mappings.py`
- `config/team_aliases.json`

# Known Limitations

- Large static mapping tables (aliases / Svenska Spel maps) require manual maintenance
- Fuzzy matching can false-positive — tests define expected behavior

# Unknowns

- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Policy for auto-creating missing teams vs strict rejection
