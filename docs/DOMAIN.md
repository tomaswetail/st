# Domain Overview

This system models **Swedish pool betting (Stryktipset)** on top of **European football match data**. The domain spans three overlapping worlds:

1. **Coupon domain** — Svenska Spel draws, events, odds, public bet shares
2. **Football domain** — leagues, teams, fixtures, goals, advanced stats
3. **Modeling domain** — features, baselines, probabilities, backtest metrics

Agents must not conflate provider-specific IDs with internal database IDs.

---

# Glossary

## Stryktipset

**Definition:** A Svenska Spel betting product where players predict outcomes (1/X/2) for typically **13 football matches** per coupon/draw.

**Not the same as:** A single football fixture, a league season, or a generic "bet slip" from another bookmaker.

**Evidence:** `productName: "Stryktipset"` in tests (`tests/data_sources/test_svenskaspel_api_client.py`), `STRoundModel.product_name`.

---

## Draw / Round / Coupon

**Definition:** One Stryktipset **draw** (Swedish: omgång) identified by `draw_number`. Stored as `STRoundModel` / table `stryktipset_rounds`.

**Important properties:**
- Unique on `(product_id, draw_number)`
- Contains many `STMatchModel` rows (coupon lines)

**Lifecycle:** Fetched from Svenska Spel API; states include `Open`, `Finalized` (provider-side).

**Evidence:** `objects/models/st_round.py`, `data_sources/svenskaspel_api_client.py`

---

## ST Match (Coupon Match)

**Definition:** One row on a Stryktipset coupon — a single football match with Svenska Spel metadata, odds, bet distribution, and final result.

**Not the same as:** `FixtureModel` (historical API-Football result row). A coupon match **links to** internal teams; linking to a historical fixture is a separate resolution step.

**Important properties:**
- `external_id` = Svenska Spel `matchId` (unique)
- `stryktipset_result` ∈ {`1`, `X`, `2`}
- Has optional `match_odds`, `match_bet`

**Evidence:** `objects/models/st_match.py`

---

## Fixture / Historical Match

**Definition:** A finished (or scheduled) football match stored in `fixtures`, shaped like API-Football responses. Alias: `HistoricalMatchModel` (legacy name).

**Not the same as:** ST Match. Fixtures are the **canonical historical record** for modeling; coupon matches reference teams that must align with fixture data.

**Important properties:**
- Unique on `fixture_id` (API-Football fixture id)
- `fixture_date`, `goals_home`, `goals_away`, league/team ids

**Evidence:** `objects/models/fixture.py`, `docs/football_data_ingestion.md`

---

## Outcome (1 / X / 2)

**Definition:** Match result classification for pool betting:
- `1` — home win
- `X` — draw
- `2` — away win

**Not the same as:** Raw scores; goals are converted via `fixture_result` / `stryktipset_result`.

**Evidence:** `utils/common.py` (`Outcome = Literal["1", "X", "2"]`)

---

## Market Baseline

**Definition:** Normalized implied probabilities from Svenska Spel decimal odds (1/X/2).

**Evidence:** `calc/market_probabilities.py`, `calc/residual_ml/baseline.py` (`market_baseline`)

---

## Engine Baseline

**Definition:** Model-derived 1X2 probabilities, typically from **Dixon–Coles** goal model (`p_home_dc`, `p_draw_dc`, `p_away_dc` in features).

**Not the same as:** Final output when ML is enabled (ML may override blend).

**Evidence:** `calc/residual_ml/baseline.py` (`engine_baseline`)

---

## Blend Baseline

**Definition:** Weighted combination of market and engine baselines (default 0.7/0.3). Input to residual ML when enabled.

**Evidence:** `calc/residual_ml/baseline.py` (`blend_baselines`)

---

## Residual ML

**Definition:** Gradient-boosting model predicting **residual adjustments** to the blend baseline in logit space, outputting final 1X2 probabilities.

**Evidence:** `calc/residual_ml/model.py`, `calc/residual_ml/trainer.py`

---

## Team Strength Features

**Definition:** Recency-weighted xG/shot/set-piece/GK metrics for a team before a cutoff date, optionally venue-split and opponent-adjusted.

**Evidence:** `objects/schema/data_classes/team_strength_features.py`, `calc/strength_calculator.py`

---

## Classic Dixon–Coles (DC)

**Definition:** Poisson goals model with low-score dependence (ρ), fit per league with configurable lookback and time decay (ξ).

**Evidence:** `calc/dixon_coles/model.py`, `config/classic_dc_league_params.json`

---

## External Entity Mapping

**Definition:** Maps `(provider, entity_type, external_entity_id)` ↔ `internal_entity_id` for teams, leagues, fixtures across providers.

**Providers in use:** `api-football`, `sofascore`, `fotmob`, `svenska-spel` (inferred from resolver/service code).

**Evidence:** `objects/models/external_entity_mapping.py`, `data_sources/entity_resolver.py`

---

## Provider

**Definition:** External data source identifier string used in mappings and stats rows (e.g. `"fotmob"`, `"sofascore"`, `"svenska-spel"`).

**Not the same as:** Internal league code (E0, SP1) or API-Football numeric league id without mapping context.

---

## League ID (ambiguous — use qualified names)

| Term | Meaning |
|------|---------|
| **API-Football league id** | External id in `fixtures.league_id`, `LEAGUES_EXTERNAL_IDS` |
| **Internal league id** | `leagues.id` primary key |
| **SofaScore/FotMob league id** | Provider-specific; stored in `external_entity_mapping` |

Always specify which ID space when writing code or docs.

---

# Entity Relationships

```
STRoundModel (stryktipset_rounds)
└── STMatchModel (stryktipset_matches) [many]
    ├── home_team → TeamModel
    ├── away_team → TeamModel
    ├── match_odds → STMatchOddsModel [0..1]
    └── match_bet → STMatchBetModel [0..1]

TeamModel (teams)
├── referenced by ST matches
├── referenced by fixtures (via team names/ids)
└── external_entity_mapping [many providers]

LeagueModel (leagues)
└── external_entity_mapping

FixtureModel (fixtures)
├── match_advanced_stats [by match_id + provider]
└── match_shots [by match_id + provider]

ExternalEntityMappingModel
└── links provider entities → internal Team/League/Fixture ids
```

**Cross-domain link:** ST matches → modeling features via `ResidualMLFeatureAssembler`, which resolves teams and loads historical fixtures/stats before kickoff.

Evidence: `calc/residual_ml/feature_assembler.py`, entity resolver tests.

---

# State Machines

## Svenska Spel draw state (provider)

Observed values in tests:
- `Open` — registration open; short cache TTL
- `Finalized` — completed; long cache TTL

**UNKNOWN / NEEDS PRODUCT CONFIRMATION** — full list of states and transitions.

Evidence: `tests/data_sources/test_svenskaspel_api_client.py`

## Fixture status

Fixtures store `status_short` / `status_long` from API-Football (e.g. `FT` for finished). Used in queries for missing stats.

Evidence: `objects/models/fixture.py`, `FixtureRepository.find_missing_stats`

## ST match result

```
scheduled (no result)
  → finished → stryktipset_result ∈ {1, X, 2}
```

Scores may populate before result char is set during import.

Evidence: `STMatchRepository.upsert_from_draw`

---

# Domain Invariants

1. **Internal IDs ≠ provider IDs** — always resolve through `EntityResolver` or mapping tables.
2. **Feature cutoff < match kickoff** — modeling features for a coupon match use data strictly before that match's start.
3. **Three-outcome probability space** — all probability outputs must cover exactly `{1, X, 2}` and sum to ~1 after normalization.
4. **Coupon match external_id is Svenska Spel matchId** — not API-Football fixture id.

---

# Domain Edge Cases

1. **Swedish team names on coupons** — normalized via `NATIONAL_TEAMS_SE_TO_EN`, `fix_swedish_name`, `config/team_aliases.json`.
2. **Svenska Spel → API-Football team mapping** — static map `SVENSKA_SPEL_TO_API_FOOTBALL_TEAMS` tried before fuzzy match.
3. **Same-date fixture ambiguity** — repository tests cover multiple fixtures on same date; resolution uses team names + tolerance.
4. **Missing xG** — matches without `match_advanced_stats` skipped in strength calculation; may reduce feature coverage.
5. **Unmapped FotMob leagues** — `API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING` contains `None` for some Swedish lower divisions.
6. **`draw_number` in ML features** — code passes `match.stryktipset_round_id` (round PK), which may not equal Svenska Spel `draw_number`. Verify before using this field analytically.

Evidence: `utils/common.py`, `data_sources/entity_resolver.py`, `tests/repositories/test_find_for_st_match.py`, `calc/probability_manager.py`

---

# Terminology That Must Not Be Used Interchangeably

| Do not conflate | Why |
|-----------------|-----|
| **ST Match** vs **Fixture** | Different tables, different external IDs, different ingestion paths |
| **draw_number** vs **stryktipset_round_id** | Round PK vs Svenska Spel draw number |
| **fixture_id** vs **fixtures.id** | API-Football id vs internal surrogate PK |
| **Market probability** vs **Bet distribution %** | Odds-implied probs vs public stake shares (`STMatchBetModel`) |
| **Engine baseline** vs **Final probability** | DC/strength vs ML-adjusted output |
| **HistoricalMatchModel** vs **FixtureModel** | Legacy alias only; prefer FixtureModel |
