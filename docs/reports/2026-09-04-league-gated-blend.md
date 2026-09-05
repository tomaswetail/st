# 2026-09-04 — P0 league-gated blend

**Decision:** **APPROVED** (protocol). Gate outcome: **KILL**. Production was **not** promoted.  
**Date:** 2026-09-05 00:20  
**Owner:** Project Leader  
**Implementation:** [developer](9cd5d5cb-69f2-4cfa-b0a7-03aa00716d36)  
**Verification:** [verifier](0b6c0d94-81f5-4f81-9ff3-dc81d60299c7) — VERIFIED; `tests/test_calc/` **230 passed, 2 xfailed**

---

## Outcome

League-gated blend was run as specified: lock a DC allowlist on the **492-row** tuning slice only, reblend the existing MLE CSV, retrain **one** pinned HGB, score official **518** at **α=0.7 frozen**.

| Item | Value |
|------|--------|
| Step 0 | **CONTINUE** — 492 pooled blend beats market, but league **41 is a tax** |
| Locked allowlist (492 only) | **39, 45, 180** |
| Official 518 shrink@0.7 | **1.00450** (HOLD band vs 1.0046) |
| League 39 shrink@0.7 | **0.9563** vs current **0.9526** (**+0.0037**, kill line **>0.9556**) |
| Gate | **KILL** |
| Production | Unchanged: `sweep_best/`, `config/blend_weights.json` still `enabled: false` → global 70/30 |
| Stretch ≤0.99 | still **MISSED** |

This is **not** the old conditional-blend bundle. 85/15 magnitude, DC-quality, and injury rules stayed **off**.

DEC-003 is **not** amended. Live scoring remains global 70/30 until a later confirmed PROMOTE.

---

## Task contract

### Goal

On the 492-row tuning slice only, lock a DC allowlist. Reblend the existing MLE CSV (no 20h rebuild). Retrain one pinned HGB. Score official 518 at α=0.7 frozen. Promote only if the gate says so.

Canonical ship to beat: official 518-row include-holdout **1.0046** (α=0.7), MLE CSV + `models/residual_ml/sweep_best/`. Beat **1.0046**, not historical 1.0022.

### Current behavior (before this task)

- `config/blend_weights.json` is `enabled: false` → DataSourceConfig fallback **0.7 / 0.3** for all rows.
- `select_blend_weights` ignored `league_overrides` when disabled. When enabled, it picked the candidate with the **highest** market weight, so a default of 1.0 would have silently beaten a 0.7 allowlist override.
- `reblend_residual_ml_dataset.py` left DC-null rows unchanged when the config was enabled (stale 70/30).
- `ResidualMLTrainer.predict_match_proba` uses stored row `p_*_blend` (does not re-blend from config).

### Required behavior

1. **Step 0.** Eval 492 (default, holdout excluded). Allowlist = leagues with n≥15 **and** blend LL **<** market LL. Everyone else market-only. League 45 only from 492, never from the 518 cell. STOP if pooled blend still beats market **and** 180 and 41 are not a tax on 492.
2. **Step 1.** Candidate JSON `enabled: true`, default **1.0 / 0.0**, overrides **0.7 / 0.3** for allowlisted leagues only; other rules off. Do not overwrite production blend JSON. Reblend to a new CSV. League override must apply even when it is *less* market than the default. DC-null → market-only.
3. **Step 2.** One pinned HGB (`max_depth=4`, `lr=0.03`, `max_iter=100`, `label_smoothing=0.05`, `--exclude-injury-features`). No sweep. No `--market-only-baseline`.
4. **Step 3.** Official 518 include-holdout. Ship metric is **shrink@0.7** vs **1.0046**. Do not retune α.

### Promote / hold / kill

| Result | Action |
|--------|--------|
| STOP at Step 0 | No production change. Report only. |
| 518 shrink@0.7 ≥ 1.0046 | **KILL** |
| 2026 shrink@0.7 worse than current 2026 by >0.002, or league 39 worse by >0.003 | **KILL** even if pooled looks better |
| 1.0036 < shrink@0.7 < 1.0046 | **HOLD** (keep candidates; no production copy) |
| shrink@0.7 ≤ 1.0036 and 2026/39 kill lines not hit | **PROMOTE** |

Current ship reference: 2026 shrink@0.7 ≈ **1.0265**; league 39 ≈ **0.9526**.

### Out of scope

Dataset rebuild, HGB sweep, injuries, draw adj, fail-closed rewrite, picking allowlist from 518, git commit/push, enabling 85/15 / injury / dc-quality, changing shrink default.

### Business invariants

- INV-003: 1X2 still normalize
- Dual baseline: market + classic DC **only where allowlisted**; else market
- Official gate remains 518 include-holdout
- DEC-003 stays global 70/30 unless PROMOTE (this run did not promote)

---

## Step 0 — 492 tuning slice (allowlist locked here)

```
env -u PYTHONPATH python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --json artifacts/dc_rho_mle_promotion/gate_eval_492_tuning.json
```

Default eval, holdout excluded: **492** rows.

**Pooled:** n=492 · market **1.0291** · blend **1.0256** (blend still beats market).

**Per-league (n≥15), market vs blend — used to lock the allowlist:**

| League | n | Market | Blend | Verdict |
|--------|--:|-------:|------:|---------|
| 39 | 144 | 0.9513 | 0.9491 | wins → **allowlist** |
| 41 | 80 | 1.0479 | 1.0483 | **tax** → market-only |
| 42 | 23 | 1.0467 | 1.0476 | loses → market-only |
| 45 | 19 | 0.9820 | 0.9300 | wins → **allowlist** |
| 180 | 201 | 1.0782 | 1.0733 | wins → **allowlist** |

Leagues with n<15 (61, 71, 78, 88, 113, 140, 179, 591, …) → market-only. Missing league / DC-null → market-only.

**STOP check:** pooled blend beats market, **but** league 41 is a tax (n=80≥15, blend > market). League 180 is not a tax. → **CONTINUE**.

### Locked allowlist

`["39", "45", "180"]`

Written to [`artifacts/dc_rho_mle_promotion/league_gated_allowlist.json`](../../artifacts/dc_rho_mle_promotion/league_gated_allowlist.json) **before** any `--include-holdout` eval.

League 45 was kept because **492** n=19≥15 and blend wins. The 518 cell (n=19, blend 0.9300) was **not** used to pick it. Those 19 rows happen to sit entirely in the 492 slice, so the numbers match; the policy source is still 492.

On later 518 scoring, allowlisted 180 blend slightly **loses** to market (1.0542 vs 1.0529). 180 was **not** dropped — allowlist stayed locked from 492.

---

## Steps 1–2 — candidate only

Candidate config: [`config/blend_weights_league_gated.json`](../../config/blend_weights_league_gated.json)

- `enabled: true`
- default **1.0 / 0.0**
- overrides 39 / 45 / 180 → **0.7 / 0.3**
- `market_vs_dc` / `dc_quality` / `injury_uncertainty` all **false**

`select_blend_weights` now **replaces** the default with a league override when present, so 0.7/0.3 can apply over a 1.0/0.0 default. Other conditional rules still compete by highest market weight; they were off for this candidate. Production `config/blend_weights.json` remains `enabled: false`, so live scoring still uses the DataSourceConfig 70/30 fallback.

Reblend wrote [`artifacts/dc_rho_mle_promotion/dataset_league_gated.csv`](../../artifacts/dc_rho_mle_promotion/dataset_league_gated.csv) (2589 rows). Canonical `dataset.csv` unchanged. Audit: 1627 allowlisted+DC rows at 0.7/0.3; 962 others at 1.0/0.0 with `p_*_blend` = market; 0 normalize failures. DC-null rows are market-only when the gated config is enabled.

Pinned HGB: `models/residual_ml/league_gated_candidate/` (`max_depth=4`, `lr=0.03`, `max_iter=100`, `label_smoothing=0.05`, no injury features, no sweep). `predict_match_proba` used stored row `p_*_blend`. `sweep_best/` unchanged.

---

## Step 3 — official 518, α=0.7 frozen

```
env -u PYTHONPATH python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset_league_gated.csv \
  --model models/residual_ml/league_gated_candidate/model.pkl \
  --include-holdout \
  --json artifacts/dc_rho_mle_promotion/gate_eval_league_gated.json
```

| Slice | n | Market | Gated blend | ML raw | **shrink@0.7** | Best shrink (diag.) |
|-------|--:|-------:|------------:|-------:|---------------:|---------------------|
| pooled | 518 | 1.0068 | 1.0042 | 1.0161 | **1.0045** | 1.0045 @0.7 |
| 2025 | 173 | 0.9647 | 0.9658 | 0.9636 | 0.9610 | 0.9601 @0.5 |
| 2026 | 345 | 1.0279 | 1.0234 | 1.0424 | 1.0263 | 1.0263 @0.8 |
| 39 | 135 | 0.9580 | 0.9538 | 0.9715 | **0.9563** | 0.9563 @0.7 |
| 41 | 67 | 1.0401 | 1.0401 | 1.0657 | 1.0422 | 1.0401 @1.0 |
| 180 | 158 | 1.0529 | 1.0542 | 1.0558 | 1.0498 | 1.0496 @0.6 |

Exact: pooled shrink@0.7 = **1.004500**; league 39 = **0.956303**; 2026 = **1.026329**.

Current ship (MLE CSV + `sweep_best/`, α=0.7): pooled **1.0046**, 2026 **1.0265**, league 39 **0.9526**.

---

## Gate decision: KILL

- Pooled **1.00450** is in the HOLD band (1.0036 < x < 1.0046), a hair under 1.0046.
- 2026 **1.0263** is inside the kill line (kill if > **1.0285**); slightly better than current 1.0265.
- League 39 **0.9563** is worse than current **0.9526** by **0.0037**, which exceeds **0.003** (kill if > **0.9556**).

The league-39 kill line wins even though pooled looks slightly better. **KILL.**

`sweep_best/` and `config/blend_weights.json` were left unchanged. `production_profile.md` was **not** updated. No DEC-003 note. Candidate artifacts are kept for inspection only.

Retraining HGB on a mostly market-only residual moved league 39 shrink@0.7 the wrong way even though 39 stayed allowlisted (gated blend on 39 is still 0.9538; ML raw 0.9715 is the damage).

---

## What changed (not production)

| Area | Change |
|------|--------|
| `select_blend_weights` | Enabled league override **replaces** the JSON default (0.7/0.3 can win over default 1.0/0.0). Other rules still compete by highest market weight. Live path still disabled. |
| Reblend | Enabled + DC-null → market-only 1.0/0.0 (no stale 70/30). |
| Candidate JSON / CSV / HGB | New artifacts only. Production blend JSON and `sweep_best/` untouched. |
| Tests | Override-below-default; allowlisted 70/30; non-allowlist / DC-null / missing league → market-only + INV-003. |

---

## Tests

```
env -u PYTHONPATH python -m pytest tests/test_calc/test_blend_weights.py \
  tests/test_calc/test_reblend_residual_ml_dataset.py \
  tests/test_calc/test_probability_calculations.py -q
```

Verifier: **23 passed**. Full `tests/test_calc/`: **230 passed, 2 xfailed**.

---

## Artifact paths

- [`artifacts/dc_rho_mle_promotion/gate_eval_492_tuning.json`](../../artifacts/dc_rho_mle_promotion/gate_eval_492_tuning.json)
- [`artifacts/dc_rho_mle_promotion/league_gated_allowlist.json`](../../artifacts/dc_rho_mle_promotion/league_gated_allowlist.json)
- [`config/blend_weights_league_gated.json`](../../config/blend_weights_league_gated.json) (candidate only)
- [`artifacts/dc_rho_mle_promotion/dataset_league_gated.csv`](../../artifacts/dc_rho_mle_promotion/dataset_league_gated.csv)
- `models/residual_ml/league_gated_candidate/`
- [`artifacts/dc_rho_mle_promotion/gate_eval_league_gated.json`](../../artifacts/dc_rho_mle_promotion/gate_eval_league_gated.json)

Unchanged: `config/blend_weights.json`, `models/residual_ml/sweep_best/`, `artifacts/dc_rho_mle_promotion/dataset.csv`.
