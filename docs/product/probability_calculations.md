# Probability calculations

Worked examples for the 1X2 scoring pipeline. Formulas are taken from code, not from memory.
The same numbers are locked in [`tests/test_calc/test_probability_calculations.py`](../../tests/test_calc/test_probability_calculations.py).

Pipeline (live and dataset build):

```
ST odds → market  →  70/30 blend  →  draw adjust  →  HGB residual  →  shrink to market
              DC ↗
```

Current production turns **draw adjust off** and **conditional blend off**; HGB still sits on a 70/30 blend. See [`docs/production_profile.md`](../production_profile.md).

All 1X2 vectors are required to **sum to 1** over `{1, X, 2}`.

---

## 1. Decimal odds → market probabilities

**Code:** `utils.common.odds_to_probabilities` via `calc/market_probabilities.py`.

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

`market_baseline` then renormalizes again (and accepts 0–100 percentages via `ensure_unit_probabilities`). If the three values already sum to 1, they are unchanged.

**Not used as market:** public bet % (`STMatchBet`). Those are stake shares, not implied odds.

---

## 2. Dixon–Coles engine

**Code:** `calc/dixon_coles/model.py` (fit + λ) and `calc/strength_calculator.py` (`dixon_coles_matrix`).

### Expected goals

After MLE, for a fixture:

\[
\lambda_H = \alpha_{\text{home}} \cdot \beta_{\text{away}} \cdot \gamma, \qquad
\lambda_A = \alpha_{\text{away}} \cdot \beta_{\text{home}}
\]

\(\alpha\) = attack, \(\beta\) = defence, \(\gamma\) = home advantage. Attack is pinned so the geometric mean is 1 (`sum(log α) = 0`).

### Time decay (fit only)

\[
w = \exp(-\xi \cdot d) \quad (d = \text{days before cutoff};\; w=0 \text{ if } d<0)
\]

**Example.** \(\xi=0.005\), \(d=0\) → \(w=1\). \(d=365\) → \(w=\exp(-1.825)\approx\mathbf{0.161218}\).

Lookback keeps matches with `cutoff - lookback_days ≤ match_date < cutoff` (match on cutoff day is **excluded**).

### Scoreline probability

Independent Poisson, times Dixon–Coles \(\tau\) on 0–0, 0–1, 1–0, 1–1:

\[
P(h,a)=\mathrm{Poisson}(h;\lambda_H)\,\mathrm{Poisson}(a;\lambda_A)\,\tau(h,a)
\]

\[
\tau=\begin{cases}
1-\lambda_H\lambda_A\rho & (0,0)\\
1+\lambda_H\rho & (0,1)\\
1+\lambda_A\rho & (1,0)\\
1-\rho & (1,1)\\
1 & \text{otherwise}
\end{cases}
\]

Negative \(\tau P\) is clipped to 0. The \(11\times11\) matrix (\(0\ldots10\) goals) is then **renormalized** so 1/X/2 sum to 1 (mass above 10 goals is dropped, not assigned to a tail).

**Example.** \(\lambda_H=1.4\), \(\lambda_A=1.1\), \(\rho=-0.13\):

| Score | \(\tau\) | \(P(h,a)\) |
|-------|----------|------------|
| 1–0 | \(1+1.1(-0.13)=\mathbf{0.857}\) | **0.098486** |
| 0–0 | \(1-1.4\cdot1.1\cdot(-0.13)=\mathbf{1.2002}\) | **0.098518** |
| 0–1 | \(1+1.4(-0.13)=\mathbf{0.818}\) | **0.073860** |
| 1–1 | \(1-(-0.13)=\mathbf{1.13}\) | **0.142844** |

After summing the 0–10 grid and renormalizing:

\[
P(1)\approx\mathbf{0.421609},\quad P(X)\approx\mathbf{0.299212},\quad P(2)\approx\mathbf{0.279179}
\]

`engine_baseline` renormalizes `p_home_dc`, `p_draw_dc`, `p_away_dc` the same way as market. Missing any of the three → engine is `None` → blend falls back to market.

---

## 3. Blend (probability space, not logits)

**Code:** `calc/residual_ml/baseline.py` `blend_baselines`.

\[
p^{\text{blend}}_i = \frac{w_m p^{\text{mkt}}_i + w_{dc} p^{\text{dc}}_i}{w_m + w_{dc}}
\]

then renormalize (already sums to 1 if both inputs do).

**Example.** Market \((0.60, 0.25, 0.15)\), DC \((0.40, 0.30, 0.30)\), \(w_m=0.7\), \(w_{dc}=0.3\):

\[
p_1=0.7\cdot0.60+0.3\cdot0.40=\mathbf{0.54},\quad
p_X=\mathbf{0.265},\quad
p_2=\mathbf{0.195}
\]

If DC is missing, blend **is** market. If market is missing, blend **is** DC.

### Conditional weights (currently disabled)

**Code:** `calc/residual_ml/blend_weights.py`. When `config/blend_weights.json` has `"enabled": true`, start from defaults (0.7/0.3) and keep the candidate with the **highest** \(w_m\) among firing rules:

- `|market_vs_dc_*|` max ≥ threshold (default 0.15) → often 0.85/0.15
- DC quality missing/poor league → more market
- `has_availability==0` → more market

**Example.** `market_vs_dc_home=0.20` (others small) → magnitude 0.20 ≥ 0.15 → **(0.85, 0.15)**. If max gap is 0.05 → stay **(0.7, 0.3)**.

With `"enabled": false`, `select_blend_weights` ignores JSON defaults and returns DataSourceConfig fallback **0.7 / 0.3**.

---

## 4. Draw adjustment (currently disabled)

**Code:** `calc/draw_adjustment.py`.

Only the **draw** logit is shifted; 1 and 2 keep their pre-adjust values until renormalization:

\[
\mathrm{logit}(p'_X)=\mathrm{logit}(p_X)+\beta_0+\sum_i \beta_i x_i
\]
\[
p'_X=\sigma(\mathrm{logit}(p'_X)),\quad
(p_1,p'_X,p_2)\ \text{renormalized to sum }1
\]

\(\mathrm{logit}(p)=\ln(p/(1-p))\), \(\sigma(z)=1/(1+e^{-z})\). Missing features → 0.

**Example.** Blend \((0.45, 0.25, 0.30)\), \(\beta_0=0.5\), \(\beta_{\texttt{away\_npxg\_for}}=-0.2\), \(x=1\):

\[
\Delta=0.5-0.2=\mathbf{0.3},\quad
\mathrm{logit}(0.25)\approx\mathbf{-1.098612},\quad
p'_X=\sigma(-0.798612)\approx\mathbf{0.310025}
\]

Renormalize \((0.45, 0.310025, 0.30)\):

\[
(\mathbf{0.424399},\ \mathbf{0.292668},\ \mathbf{0.282933})
\]

Draw mass rises; home/away keep the same *ratio* as before the adjust, then all three are scaled.

---

## 5. Residual ML (HGB)

**Code:** `calc/residual_ml/baseline.py`, trainer in `calc/residual_ml/trainer.py`.

HGB predicts **three independent logit deltas** versus the blend (not a softmax). Training target with label smoothing \(s=0.05\) and observed `1`:

\[
\tilde{y}_1=1-2s=\mathbf{0.90},\quad \tilde{y}_X=\tilde{y}_2=s=\mathbf{0.05}
\]
\[
\delta_i=\mathrm{logit}(\tilde{y}_i)-\mathrm{logit}(p^{\text{blend}}_i)
\]

**Example.** Blend \((0.50, 0.28, 0.22)\), label `1`, \(s=0.05\):

\[
\delta_1=\mathrm{logit}(0.90)-\mathrm{logit}(0.50)\approx\mathbf{2.197225}
\]

Applying those \(\delta\) recovers the soft label \((0.90, 0.05, 0.05)\) (already a simplex).

At predict time, arbitrary deltas are applied independently then **renormalized**:

\[
q_i=\sigma\bigl(\mathrm{logit}(p_i)+\delta_i\bigr),\qquad
p^{\text{ml}}_i=q_i/\sum_j q_j
\]

**Example.** Same blend, \(\delta=(0.2,-0.1,-0.1)\) → \(p^{\text{ml}}\approx(\mathbf{0.542540},\ \mathbf{0.256837},\ \mathbf{0.200624})\).

This is **not** a 3-way multinomial logit. The extra renormalize step is required because independent sigmoids do not sum to 1.

---

## 6. Shrink toward market

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

Production currently recommends **α = 0.7** (MLE-ρ freeze; see [`production_profile.md`](../production_profile.md)). Raw ML still loses to market on the 518-row include-holdout slice; α=0.7 is the shipped shrink.

---

## 7. Log loss (primary metric)

**Code:** `calc/probability_metrics.py`. Clip each \(p\) to \([\varepsilon,1-\varepsilon]\) with \(\varepsilon=10^{-15}\), renormalize, then:

\[
\mathrm{LL}=-\ln p_{\text{observed}}
\]

Pooled score is the **mean** over matches. Labels `{1,X,2}` map to indices `{0,1,2}` for `multiclass_log_loss`.

**Example.** Forecast \((0.50, 0.30, 0.20)\), result `1`:

\[
\mathrm{LL}=-\ln 0.5\approx\mathbf{0.693147}
\]

Two matches both with \(p_{\text{obs}}=0.5\) → mean **0.693147**.

A “certain” `1` with \(p=(1,0,0)\) is clipped, so LL is \(\approx\varepsilon\), not 0.

**Official gate slice:** last 20% by `match_date`, **include-holdout**, 518 rows. Do not compare exclude-holdout numbers to that gate.

---

## 8. Ranked probability score (secondary)

**Code:** `calc/dixon_coles/metrics.py`. Ordered outcomes Home → Draw → Away, \(n=3\):

\[
\mathrm{RPS}=\frac12\sum_{k=1}^{2}(F_k-O_k)^2
\]

**Example.** Forecast \((0.5, 0.3, 0.2)\) → CDF \((0.5, 0.8)\).

| Result | Observed CDF \((F_1,F_2)\) | RPS |
|--------|----------------------------|-----|
| `1` | \((1, 1)\) | \(\tfrac12((0.5-1)^2+(0.8-1)^2)=\mathbf{0.145}\) |
| `X` | \((0, 1)\) | \(\tfrac12((0.5-0)^2+(0.8-1)^2)=\mathbf{0.145}\) |
| `2` | \((0, 0)\) | \(\tfrac12((0.5-0)^2+(0.8-0)^2)=\mathbf{0.445}\) |

---

## 9. Live fallback order

**Code:** `calc/probability_manager.py` `process_match`.

1. Assemble features (cutoff-safe; injuries if snapshots exist).
2. Market from ST odds; engine from DC columns.
3. Blend (70/30 unless conditional config is enabled).
4. Draw adjust (no-op if disabled).
5. If ML loaded **and** blend exists: HGB then shrink.
6. Else **keep the blend** (not DC-only). If blend is missing, use engine, then market.

A bug that discarded blend whenever ML was off (returning DC-only) was fixed when this document was written.

---

## What this document does not cover

Feature *inputs* (xG rolling windows, rest days, league rates, availability counts) are cutoff-safe statistics, not 1X2 identities. Injury `missing_value_*` from API-Football count proxies is treated as missing (see [`docs/reports/injury/injury_phase2_results.md`](../reports/injury/injury_phase2_results.md)). Those calculators are not re-derived here.

---

## Verification

```bash
python -m pytest tests/test_calc/test_probability_calculations.py -q
```
