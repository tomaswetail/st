# Production cutoff alignment (Phase 4)

Point-in-time assumptions for live `ProbabilityManager` vs residual ML backtest.

## Feature cutoff

Assembler cutoff is **match kickoff date** (`match.start_time` → `feature_cutoff_date`).
Historical strength / rest / league features only use fixtures **before** that cutoff.

## Injury / availability

`PlayerAvailabilityCalculator` loads the latest `match_availability` row with
`provider=api-football` and **`snapshot_at <= match.start_time`**. Missing snapshots
set `has_availability=0` and null missing-value features (no future leakage).

Live path uses the **same** `ResidualMLFeatureAssembler` as dataset build — no separate
injury bypass.

Env / ops: run injury backfill (`scripts/backfill_injury_snapshots.py`) before coupon
scoring so pre-kickoff snapshots exist.

## Market odds timing (coverage caveat)

`p_*_market` come from **`STMatchOdds` stored on the ST match** via
`ResidualMLFeatureAssembler._market_probabilities` — typically the odds captured when
the coupon/draw was imported (often **start / early odds**), **not** a refreshed
pre-kickoff book.

There is **no** separate “odds as-of injury snapshot” timestamp in the current schema.
If stored ST odds are stale relative to injury news, market and injury features are
**not** guaranteed to be synchronized. Backtest and live share this limitation.

Do **not** treat market probs as kickoff-locked closing lines unless product ingestion
is changed to refresh odds.

## Pipeline order (production)

1. Conditional blend (`config/blend_weights.json`)
2. Draw adjustment (`config/draw_adjustment.json`)
3. HGB residual (when `RESIDUAL_ML_ENABLED`)
4. Optional shrink: `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` (applied in `ProbabilityManager`)

## Other production caveats

- Dataset CSV gains `league_external_id` on **rebuild**; existing CSV may lack it until then.
- HGB may still be trained on pre-draw-adjust blend until a full rebuild/retrain.
- Home advantage: production/research default often `RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast`;
  `full` mode deferred (expensive rebuild) — see roadmap Phase 4.4.
