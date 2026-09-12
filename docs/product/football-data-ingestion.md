# Purpose

Build and maintain a **historical football database** (fixtures, teams, leagues, xG, shots) used as the feature foundation for Stryktipset modeling.

# Core Concepts

- **Results source:** API-Football → `fixtures` table
- **Advanced stats source:** SofaScore (documented) or FotMob (implemented) → `match_advanced_stats`, `match_shots`
- **Attachment model:** xG is fetched for **existing** fixtures, not created by default

# Entities

| Entity | Table | Notes |
|--------|-------|-------|
| `FixtureModel` | `fixtures` | Unique on `fixture_id` |
| `TeamModel` | `teams` | Internal registry; `external_id` = API-Football |
| `LeagueModel` | `leagues` | Internal registry; `external_id` = API-Football |
| `MatchAdvancedStatsModel` | `match_advanced_stats` | Per fixture + provider |
| `MatchShotModel` | `match_shots` | Shot-level events |

# Primary Workflows

## WF-A — Bulk fixture refresh

`DataCollector(session).refresh_all_data(seasons)` → API-Football → upsert fixtures/teams/leagues.

## WF-B — League history xG import

`ExtendedMatchDataService.fetch_and_store_league_history(league_id, ...)`

## WF-C — Missing stats backfill

`scripts/missing_stats.py` → `fetch_and_store_all_fixtures` over `LEAGUES_EXTERNAL_IDS`.

## WF-D — Wipe and reimport

Documented SQL TRUNCATE + sequential reimport in `docs/football_data_ingestion.md`.

# State/Lifecycle

- Fixture progresses via API-Football `status_short` (e.g. `FT`)
- Advanced stats rows keyed by `(match_id, provider)` — verify in model
- Unresolved provider matches → CSV at `data/unresolved_matches.csv`

# Business Rules

- **BR-006** — fixture upsert idempotent on `fixture_id`
- **INV-006** — xG import attaches to existing `fixtures` rows; unresolved matches logged to CSV
- Docs: fixture odds from API-Football **not** persisted. Historical 1X2 odds from football-data.co.uk **are** persisted on `fixture_odds` (DEC-016); live ST odds path unchanged.
- Season codes: YYXX or calendar year → stored as calendar start year (`utils/seasons.py`)

# Permissions

None.

# Side Effects

- HTTP calls to external APIs (rate-limited)
- Disk cache under `data/cache/`
- DB upserts; optional CSV logs for unresolved entities
- `scripts/missing_stats.py` prints `alias_candidates` on exit (debug)

# Integrations

| Provider | Code |
|----------|------|
| API-Football | `data_sources/api_football_client.py` |
| SofaScore | `data_sources/football_data/providers/sofascore.py` |
| FotMob | `data_sources/football_data/providers/fotmob.py` |

# Failure Scenarios

- 404 from provider → `NotFoundError`
- 429/5xx → retries with backoff
- Unresolved fixture → skipped or logged, counted in `BatchImportResult.unresolved`

# Edge Cases

- Kickoff tolerance: `kickoff_match_tolerance_minutes` (default 24h) for matching
- xG aggregate tolerance: `xg_aggregate_tolerance` (default 0.15)
- Swedish lower leagues may lack FotMob mapping (`None` in mapping dict)
- **Stale doc:** `services/football_data_cli.py` referenced but missing

# Important Tests

- `tests/football_data/test_ingestion.py`
- `tests/football_data/test_fetch_all_fixtures.py`
- `tests/data_sources/test_data_collector.py`
- `tests/data_sources/test_api_football_client_cache.py`
- `tests/repositories/test_historical_match_same_date.py`

# Relevant Code

- `data_sources/football_data/service.py`
- `data_sources/data_collector.py`
- `objects/repositories/fixture_repository.py`
- `docs/football_data_ingestion.md` (operational guide; verify against code)

# Known Limitations

- No `football_data_cli` module despite documentation
- `main.py` still uses FotMob loop while docs prefer SofaScore
- Hardcoded API key default in config (see DEC-U03)
- Full league list in `LEAGUES_EXTERNAL_IDS` may include duplicates (e.g. `595` twice)

# Unknowns

- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Authoritative xG provider for production
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Whether Understat will be implemented (`xg_provider` default)
