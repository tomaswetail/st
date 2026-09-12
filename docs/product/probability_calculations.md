# Probability calculations

Worked examples for the 1X2 scoring pipeline (see DEC-015). Formulas are taken from code, not from memory. The same numbers are locked in [`tests/test_calc/test_probability_calculations.py`](../../tests/test_calc/test_probability_calculations.py).

Pipeline (live and dataset build):

```
ST odds → market_baseline  →  (optional) HGB residual  →  (optional) shrink to market
```

All 1X2 vectors are required to **sum to 1** over `{1, X, 2}`.

---

## 1. Decimal odds → market probabilities

**Code:** `utils.common.odds_to_probabilities` via `calc/market_probabilities.py`.

Odds must be finite and **greater than 1**. Null, non-numeric, NaN, inf, and values in `(0, 1]` are rejected (`validate_decimal_odds`). Overround < 1 is valid (flagged, not dropped).

Remove overround by normalizing inverse odds:

\[
p_i = \frac{1/o_i}{\sum_{j \in \{1,X,2\}} 1/o_j}
\]

**Example.** Odds \(o_1=2.00\), \(o_X=3.50\), \(o_2=4.00\):

| Outcome | \(1/o\) | Normalized \(p\) |
|---------|---------|------------------|
| 1 | 0.500000 | **0.48275862069** |
| X | 0.285714 | **0.275862068966** |
| 2 | 0.250000 | **0.241379310345** |

Sum of \(1/o\) = 1.035714; each row is \( (1/o) / 1.035714 \).

`market_baseline` renormalizes again (and accepts 0–100 percentages via `ensure_unit_probabilities`). If the three values already sum to 1, they are unchanged.

**Not used as market:** public bet % (`STMatchBet`). Those are stake shares, not implied odds.

---

## 2. Residual ML (HGB) vs the market baseline

**Code:** `calc/residual_ml/baseline.py`, trainer in `calc/residual_ml/trainer.py`.

HGB predicts **three independent logit deltas** versus the market baseline (not a softmax). Training target with label smoothing \(s=0.05\) and observed `1`:

\[
\tilde{y}_1=1-2s=\mathbf{0.90},\quad \tilde{y}_X=\tilde{y}_2=s=\mathbf{0.05}
\]
\[
\delta_i=\mathrm{logit}(\tilde{y}_i)-\mathrm{logit}(p^{\text{mkt}}_i)
\]

**Example.** Market \((0.50, 0.28, 0.22)\), label `1`, \(s=0.05\):

\[
\delta_1=\mathrm{logit}(0.90)-\mathrm{logit}(0.50)\approx\mathbf{2.197225}
\]

Applying those \(\delta\) recovers the soft label \((0.90, 0.05, 0.05)\) (already a simplex).

At predict time, arbitrary deltas are applied independently then **renormalized**:

\[
q_i=\sigma\bigl(\mathrm{logit}(p_i)+\delta_i\bigr),\qquad
p^{\text{ml}}_i=q_i/\sum_j q_j
\]

**Example.** Same market, \(\delta=(0.2,-0.1,-0.1)\) → \(p^{\text{ml}}\approx(\mathbf{0.542540},\ \mathbf{0.256837},\ \mathbf{0.200624})\).

This is **not** a 3-way multinomial logit. The extra renormalize step is required because independent sigmoids do not sum to 1.

### Open issue OI1 — shipped HGB predates the pivot

The shipped HGB in `models/residual_ml/sweep_best/` was trained against the pre-pivot **blend** baseline and must be retrained against the **market** baseline before residuals are re-enabled in production. A backup of the pre-pivot model lives at `models/residual_ml/sweep_best_pre_market_pivot_bck/`. Residual ML is off by default (`RESIDUAL_ML_ENABLED=false`) until the retrain lands. See [DEC-015](../DECISIONS.md) OI1.

---

## 3. Shrink toward market

**Code:** `shrink_toward_market`. \(\alpha=0\) keeps ML; \(\alpha=1\) is market.

\[
p^{\text{final}}_i=(1-\alpha)\,p^{\text{ml}}_i+\alpha\,p^{\text{mkt}}_i
\]

**Example.** ML \((0.60, 0.20, 0.20)\), market \((0.50, 0.28, 0.22)\):

| \(\alpha\) | \(p_1\) | \(p_X\) | \(p_2\) |
|------------|---------|---------|---------|
| 0.0 | 0.60 | 0.20 | 0.20 |
| 0.5 | **0.55** | **0.24** | **0.21** |
| 0.9 | **0.51** | **0.272** | **0.218** |
| 1.0 | 0.50 | 0.28 | 0.22 |

`residual_ml_final_shrink_to_market` defaults to **0.7**.

---

## 4. Log loss (primary metric)

**Code:** `calc/probability_metrics.py`. Clip each \(p\) to \([\varepsilon,1-\varepsilon]\) with \(\varepsilon=10^{-15}\), renormalize, then:

\[
\mathrm{LL}=-\ln p_{\text{observed}}
\]

Pooled score is the **mean** over matches. Labels `{1,X,2}` map to indices `{0,1,2}` for `multiclass_log_loss`.

**Example.** Forecast \((0.50, 0.30, 0.20)\), result `1`:

\[
\mathrm{LL}=-\ln 0.5\approx\mathbf{0.693147}
\]

---

## 5. Live fallback order

**Code:** `calc/probability_manager.py` `process_match`.

1. Assemble features (cutoff-safe; injuries if snapshots exist).
2. Market baseline from ST odds.
3. If ML loaded: HGB residual delta → apply. Then optional shrink toward market.
4. Else return the market baseline.

If market probabilities are missing, `ProbabilityManager` raises `ValueError` for that match.

Live coupon scoring **does not** read `fixture_odds`. That table is for the residual-ML training dataset only (DEC-002 / DEC-016).

---

## 6. Fixture dataset path (training only)

**Code:** `ResidualMLFeatureAssembler.assemble`, `ResidualMLDatasetBuilder.iter_fixture_rows`, `load_fixture_market_probabilities`.

Historical training rows can be built from finished `fixtures` + `fixture_odds` instead of ST coupon matches:

```
cutoff = fixture.fixture_date  (history strictly before kickoff)
market = fixture_odds (default bookmaker Avg, price_type closing)
features = assemble(fixture, before_date=cutoff)
label = 1/X/2 from full-time goals
```

`assemble()` still accepts ST matches. If `before_date` is omitted, cutoff defaults to kickoff (live `ProbabilityManager` call style is unchanged). Fixture team ids are API-Football `teams.external_id` and are resolved to internal `teams.id` before strength / home-advantage queries.

Market-shape columns on every row (ST `describe()` and fixture loader): `market_overround`, `market_entropy`, `market_top_probability`, `market_second_probability`, `market_probability_gap`. `market_price_type` is metadata only.

### Closing odds incorporate late team news our features cannot see

Default `FIXTURE_ODDS_PRICE_TYPE=closing` is the canonical archive price, but closing 1X2 already prices lineup/injury news that arrives after our feature cutoff (`snapshot_at <= kickoff`, history `fixture_date < cutoff`). That is a property of the benchmark, not a bug to hide.

Opening prices are the leakage-clean control:

```bash
FIXTURE_ODDS_PRICE_TYPE=opening python -m src.scripts.build_residual_ml_dataset \
  --source fixtures --league E0 --season 2023 \
  --output data/residual_ml/fixtures_e0_2023_opening.csv
```

or `--price-type opening` on the same CLI. Never mix opening and closing on one row.

### CLI

ST (unchanged default):

```bash
python -m src.scripts.build_residual_ml_dataset
```

Fixtures smoke / full-build example:

```bash
python -m src.scripts.build_residual_ml_dataset --source fixtures --league E0 --season 2023 --output data/residual_ml/fixtures_e0_2023.csv
```

`--source fixtures` without `--output` writes `data/residual_ml/fixtures_dataset.csv` so it does not overwrite the ST CSV.

---

## Verification

```bash
python -m pytest tests/test_calc/test_probability_calculations.py -q
```
