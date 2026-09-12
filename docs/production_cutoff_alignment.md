# Production cutoff alignment

Point-in-time assumptions for live `ProbabilityManager` vs residual ML backtest, post [DEC-015](DECISIONS.md).

## Feature cutoff

Assembler cutoff is **match kickoff date** (`match.start_time` → `feature_cutoff_date`).
Historical strength / rest / league features only use fixtures **before** that cutoff.

## Injury / availability

`PlayerAvailabilityCalculator` loads the latest `match_availability` row with
`provider=api-football` and **`snapshot_at <= match.start_time`**. Missing snapshots
set `has_availability=0` and null missing-value features (no future leakage).

Live path uses the **same** `ResidualMLFeatureAssembler` as dataset build.

## Market odds timing (coverage caveat)

`p_*_market` come from **`STMatchOdds` stored on the ST match** via
`ResidualMLFeatureAssembler._market_probabilities` — typically the odds captured when
the coupon/draw was imported (often **start / early odds**), **not** a refreshed
pre-kickoff book.

There is **no** separate "odds as-of injury snapshot" timestamp in the current schema.
If stored ST odds are stale relative to injury news, market and injury features are
**not** guaranteed to be synchronized. Backtest and live share this limitation.

Do **not** treat market probs as kickoff-locked closing lines unless product ingestion
is changed to refresh odds.

## Pipeline order (production)

1. Market baseline from stored ST odds
2. Optional residual ML delta (when `RESIDUAL_ML_ENABLED`)
3. Optional shrink toward market (`RESIDUAL_ML_FINAL_SHRINK_TO_MARKET`)

## Other production caveats

- Dataset CSV writes `league_external_id`; older CSVs may lack it until rebuild.
- The shipped HGB was trained against the pre-pivot blend baseline (see DEC-015 OI1) and requires a retrain against the market baseline before production use.
