# Historical Football Data Ingestion

**Results / teams / leagues** come from **API-Football** only (via `config/api_football_leagues.json`).  
Fixture odds are not persisted (ST odds on Svenska Spel draws are unchanged).  
**xG / shots** come from **SofaScore** only, attached to existing `fixtures` rows.

## Architecture

```text
main.py / DataCollector          → APIFootballClient → fixtures, teams, leagues
ExtendedMatchDataService         → SofaScoreProvider → match_advanced_stats, match_shots
```

| Layer | Location |
|-------|----------|
| League code → AF id map | `config/api_football_leagues.json` |
| API-Football client | `data_sources/api_football_client.py` |
| Match / team import | `data_sources/data_collector.py`, `team_name_fetcher.py` |
| SofaScore xG service | `data_sources/football_data/service.py` |
| League catalogue (SofaScore) | `data_sources/football_data/league_catalogue.py` |
| Entity resolution | `data_sources/entity_resolver.py` |
| CLI | `services/football_data_cli.py` |

## Wipe and reimport

Old `source` values (`football-data.co.uk`, `espn`, …) and `provider=fotmob` mappings are obsolete. After backup:

```sql
TRUNCATE fixtures, match_advanced_stats, match_shots,
  external_entity_mapping RESTART IDENTITY CASCADE;
-- optionally also truncate teams / leagues if starting clean
```

Then:

1. `fetch_team_names(session=session)`
2. `DataCollector(...).refresh_all_data(seasons=[...])`
3. SofaScore xG via `main_extra_data()` or `python -m services.football_data_cli import-league ...`

Requires `API_FOOTBALL_KEY` in the environment (no hardcoded fallback).

## API-Football configuration

| Variable / field | Purpose |
|------------------|---------|
| `API_FOOTBALL_KEY` | API-Football auth key |
| `api_football_leagues_path` | JSON map of internal codes → AF league ids |
| `FOOTBALL_DATA_CACHE_TTL_SECONDS` | Shared disk cache TTL (default 1 year) |

Seasons accept YYXX (`2425`) or calendar year (`2024`); stored `league_season` is the calendar start year.

To rediscover missing MAIN/EXTRA codes (requires `API_FOOTBALL_KEY`):

```bash
python -m data_sources.api_football_leagues
```

## SofaScore xG configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOOTBALL_DATA_PROVIDER` | `sofascore` | Only supported value |
| `SOFASCORE_BASE_URL` | `https://api.sofascore.com/api/v1` | SofaScore API base |
| `FOOTBALL_DATA_REQUEST_DELAY_MS` | `500` | Throttle between requests |
| `FOOTBALL_DATA_MAX_RETRIES` | `3` | Retries for 429/5xx/transport |
| `FOOTBALL_DATA_CACHE_TTL_SECONDS` | `31536000` (1 year) | Disk cache TTL |

Other config: `kickoff_match_tolerance_minutes`, `xg_aggregate_tolerance`, `create_missing_historical_matches` (default `False`), `fuzzy_match_threshold`.

## External-ID mapping

Internal IDs are not SofaScore/API-Football IDs. Map them in `external_entity_mapping`:

- Unique on `(provider, entity_type, external_entity_id)`
- Unique on `(provider, entity_type, internal_entity_id)`

Providers in use: `api-football` (teams/leagues/fixtures), `sofascore` (xG leagues/matches).

### Discover and map SofaScore leagues

```python
from database import SessionLocal
from data_sources.football_data.league_catalogue import LeagueCatalogueService

session = SessionLocal()
catalogue = LeagueCatalogueService(provider="sofascore", session=session)
premier = catalogue.find_leagues(query="Premier League", country="England")
result = catalogue.map_league(league_id=1, external_entity_id="17")
catalogue.close()
```

```bash
python -m services.football_data_cli list-leagues --provider sofascore --query Premier
python -m services.football_data_cli suggest-league-mappings --provider sofascore
python -m services.football_data_cli map-league --provider sofascore --league-id 1 --external-id 17
```

## Example xG import

```python
from database import SessionLocal
from data_sources.football_data import ExtendedMatchDataService

session = SessionLocal()
service = ExtendedMatchDataService(provider="sofascore", session=session)
result = service.fetch_and_store_league_history(league_id=1)
service.close()
```

```bash
python -m services.football_data_cli import-league \
  --provider sofascore \
  --league-id 123 \
  --season 2024/2025
```

## Rate-limit behaviour

`ThrottledHttpClient` waits `FOOTBALL_DATA_REQUEST_DELAY_MS`, retries 429/5xx with backoff, raises `NotFoundError` on 404, and optionally caches JSON on disk.

## Tests

```bash
python -m pytest tests/football_data tests/data_sources -v
```

Tests use fixtures and mocked HTTP — no live network calls.
