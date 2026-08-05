# Historical Football Data Ingestion

Provider-agnostic historical xG and shot ingestion for **FotMob** and **SofaScore**. Data is attached to existing `historical_matches` rows (not `stryktipset_matches`).

## Architecture

```text
HistoricalFootballDataService
    ├── FootballDataProvider (Protocol)
    ├── FotMobProvider
    └── SofaScoreProvider
```

| Layer | Location |
|-------|----------|
| Public import service | `data_sources/football_data/service.py` |
| League catalogue API | `data_sources/football_data/league_catalogue.py` |
| Providers | `data_sources/football_data/providers/` |
| HTTP client | `data_sources/football_data/http_client.py` |
| Entity resolution | `data_sources/football_data/entity_resolver.py` |
| Derived metrics | `data_sources/football_data/metrics.py` |
| Team features | `services/team_strength_feature_service.py` |
| CLI | `services/football_data_cli.py` |

Schema delivery uses SQLAlchemy `create_all` via `init_db()` (no Alembic).

## Configuration

Environment variables (also mirrored on `DataSourceConfig`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOOTBALL_DATA_PROVIDER` | `fotmob` | Default primary provider |
| `FOTMOB_BASE_URL` | `https://www.fotmob.com/api/data` | FotMob API base |
| `SOFASCORE_BASE_URL` | `https://api.sofascore.com/api/v1` | SofaScore API base |
| `FOOTBALL_DATA_REQUEST_DELAY_MS` | `500` | Throttle between requests |
| `FOOTBALL_DATA_MAX_RETRIES` | `3` | Retries for 429/5xx/transport |
| `FOOTBALL_DATA_CACHE_TTL_SECONDS` | `3600` | Disk cache TTL |

Other config fields: `kickoff_match_tolerance_minutes`, `xg_aggregate_tolerance`, `create_missing_historical_matches` (default `False`), `fuzzy_match_threshold`.

FotMob league fixtures use a fixed season list (`2024/2025`, `2025/2026`) and:

`GET {FOTMOB_BASE_URL}/leagues?id={fotmob_id}&season={season}&ccode3={ccode}`

## External-ID mapping

Internal IDs are **not** FotMob/SofaScore IDs. Map them in `external_entity_mapping`:

- Unique on `(provider, entity_type, external_entity_id)`
- Unique on `(provider, entity_type, internal_entity_id)`

Entity types: `league`, `team`, `match`, `season`.

### Discover and map leagues (Python API)

Use `LeagueCatalogueService` to list provider competitions and write league mappings:

```python
from database import SessionLocal
from data_sources.football_data import LeagueCatalogueService

session = SessionLocal()
catalogue = LeagueCatalogueService(provider="fotmob", session=session)

# Full catalogue
leagues = catalogue.list_available_leagues()

# Search
premier = catalogue.find_leagues(query="Premier League", country="England")

# Suggest matches for internal leagues.rows (never writes)
suggestions = catalogue.suggest_mappings()

# Persist mapping for leagues.id
result = catalogue.map_league(league_id=1, external_entity_id="47")
# or high-confidence name match:
result = catalogue.map_league(league_id=1, query="Premier League")

catalogue.close()
```

Team and match mappings are created automatically when resolution succeeds during import.

Optional CLI wrappers (same functions):

```bash
python -m services.football_data_cli list-leagues --provider fotmob --query Premier
python -m services.football_data_cli suggest-league-mappings --provider fotmob
python -m services.football_data_cli map-league --provider fotmob --league-id 1 --external-id 47
```

## Example service usage

```python
from database import SessionLocal
from data_sources.football_data import HistoricalFootballDataService

session = SessionLocal()
service = HistoricalFootballDataService(
    provider="fotmob",
    fallback_provider="sofascore",
    session=session,
)
result = service.fetch_and_store_league_history(
    league_id=1,
    season="2025",
)
print(result.imported, result.unresolved, result.failed)
service.close()
```

Single match (`match_id` = `historical_matches.id`):

```python
result = service.fetch_and_store_match(match_id=456, force_refresh=True)
```

## CLI examples

```bash
python -m services.football_data_cli import-league \
  --provider fotmob \
  --league-id 123 \
  --season 2025

python -m services.football_data_cli import-match \
  --provider sofascore \
  --match-id 456

python -m services.football_data_cli import-league \
  --provider fotmob \
  --fallback-provider sofascore \
  --league-id 123 \
  --force-refresh \
  --dry-run \
  --request-delay 750 \
  --limit 50
```

## Database schema changes

New tables created by `init_db()`:

### `external_entity_mapping`

Maps provider entities to internal `leagues` / `teams` / `historical_matches` ids.

### `match_advanced_stats`

Unique `(match_id, provider)`. Stores provider-reported and shot-derived xG aggregates (`home_xg`, `away_xg`, npxG, xGOT, set-piece/open-play, averages, payload hash).

### `match_shots`

Unique `(match_id, provider, shot_fingerprint)`. Fingerprint is deterministic when `provider_shot_id` is absent.

`match_id` always references `historical_matches.id`.

## Supported fields per provider

| Field | FotMob | SofaScore |
|-------|--------|-----------|
| Season list | Yes | Yes |
| Season fixtures | Yes | Yes (paged last events) |
| Kickoff UTC | Yes | Yes |
| Scores / status | Yes | Yes |
| Shot map + xG | Yes (when present) | Yes (shotmap endpoint) |
| Aggregate xG stats | Partial (varies by payload) | Partial (statistics endpoint) |
| xGOT | When present on shots | When present on shots |
| Lineups | Stored raw when present | Stored raw when present |

Missing values are stored as `NULL`, never coerced to zero.

## Rate-limit behaviour

The shared `ThrottledHttpClient`:

1. Waits `FOOTBALL_DATA_REQUEST_DELAY_MS` between requests
2. Retries on HTTP 429 and 5xx with exponential backoff
3. Raises `NotFoundError` on 404 (no retry)
4. Optionally caches JSON responses on disk for `FOOTBALL_DATA_CACHE_TTL_SECONDS`

Do not bypass authentication, paywalls, or anti-bot protections. Use only endpoints the application may access under the provider’s terms.

## Unresolved teams and matches

Resolution order:

**Teams:** provider mapping → exact normalized name → alias → fuzzy (above threshold) → unresolved (logged, not linked).

**Matches:** provider mapping → league + teams + kickoff window → season + teams (postponed/reschedule) → unresolved.

Creating missing historical matches is **disabled by default** (`create_missing_historical_matches=False`). Unresolved fixtures appear in `BatchImportResult.unresolved`.

## Fallback provider

Optional `fallback_provider` is used only when the primary has no match, no xG/shots, or fails after retries. Provider data is **not** merged field-by-field; each stats/shot row records its `provider`. Prefer the configured primary when both exist; disagreements produce warnings.

## Team strength features

```python
from services.team_strength_feature_service import TeamStrengthFeatureService

features = TeamStrengthFeatureService(session=session).calculate_features(
    team_id=10,
    before=kickoff_utc,
    venue="home",
    lookback_matches=20,
    decay=0.9,
)
```

Only matches with `match_date < before` are used (no leakage). Ratings can be shrunk toward league averages. Opponent adjustment method is configurable (`none` / `simple`).

## Adding another provider

1. Implement `FootballDataProvider` in `data_sources/football_data/providers/`
2. Keep endpoint paths and IDs inside the adapter
3. Parse into `ProviderMatch` / `ProviderShot` / `ProviderMatchDetails`
4. Register the name in `build_provider()` and config literals
5. Add JSON fixtures + parser tests under `tests/football_data/`

## Tests

```bash
python -m pytest tests/football_data -v
```

Tests use saved JSON fixtures and mocked HTTP — no live network calls.
