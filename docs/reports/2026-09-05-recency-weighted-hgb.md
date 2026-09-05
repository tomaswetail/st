# 2026-09-05 — Recency-weighted HGB retrain

**Decision:** **APPROVED** (protocol). Gate outcome: **KILL**. Production was **not** promoted.  
**Date:** 2026-09-05 10:25  
**Owner:** Project Leader  
**Implementation:** [developer](95ebde9b-69ad-4fe9-9773-ad58d49847a0)  
**Verification:** [verifier](49e26afb-c5ed-417c-a472-3f76137d9c33) — VERIFIED; `tests/test_calc/` **236 passed, 2 xfailed**

---

## Outcome

One pinned HGB retrain on the existing MLE CSV with pre-registered train sample weights (exponential decay, half-life **180** days vs max **train** `match_date`). Official 518 scored at **α=0.7 frozen**.

| Item | Value |
|------|--------|
| Official 518 shrink@0.7 | **1.0088** vs ship **1.0046** |
| 2026 shrink@0.7 | **1.0310** vs ship **1.0265** (kill line **>1.0285**) |
| League 39 shrink@0.7 | **0.9503** vs ship **0.9526** (improved; kill line **>0.9556**) |
| Gate | **KILL** |
| Production | Unchanged: `models/residual_ml/sweep_best/` still the pinned MLE retrain (2026-09-04); blend still global 70/30 |
| Stretch ≤0.99 | still **MISSED** |

A correct **KILL** is success. Do not force promotion. Candidate kept for audit at `models/residual_ml/recency_180_candidate/`.

P0 league-gated blend remains **KILL** — allowlists were not re-opened.

---

## Task contract

### Goal

One pinned HGB retrain on the **existing** MLE CSV with **one** pre-registered train sample-weight: exponential decay, **half-life 180 days**, versus the latest **train** `match_date`. Score official 518 at **α=0.7 frozen**. Promote only if the gate says so.

Business outcome: test whether recency-weighted training improves official-gate shrink@0.7 versus the current ship **1.0046**, without retuning shrink, without reblending, and without changing live predict math.

### Current behavior (verified before change)

- `ResidualMLTrainer.fit` called `self.model.fit(x_train, y_train)` with **no** `sample_weight`.
- Model is `MultiOutputRegressor(HistGradientBoostingRegressor(...))` — both support `sample_weight`.
- `train_residual_ml.py` had no recency flag.
- Time split: first 80% train / last 20% val by date (`time_split_dataset_rows`).
- Eval already prints `ml_production_shrink` (α from `DataSourceConfig`, default 0.7).
- Do **not** use `dataset_league_gated.csv` or `blend_weights_league_gated.json`.

Ship slices (`docs/reports/2026-09-04-dc-failclosed-and-slice-ml.md`): pooled **1.0046**, 2026 **1.0265**, league 39 **0.9526**.

### Required behavior

1. Helper + optional `sample_weight` in `ResidualMLTrainer.fit` (default **off**).
2. CLI `--train-recency-half-life-days` through `train_and_save` only.
3. Persist half-life on the candidate artifact. Do not change live predict math.
4. Tests for \(d=0/180/360\), as-of = max train date, default unweighted fit.
5. Retrain + official 518 eval at α=0.7 frozen.
6. Apply KILL / HOLD / PROMOTE exactly. A correct KILL or HOLD is success.

### Weight formula (pre-registered — not searched)

\[
w = 2^{-d / 180} = \exp(-\ln 2 \cdot d / 180)
\]

- \(d\) = days from row `match_date` to **`max(train match_date)`** (not val, not today).
- Train rows only. Validation is unweighted (eval only).
- If \(d < 0\): \(w = 1\) (do not drop the row).
- Half-life **180** only. No grid.

### Promote / hold / kill

| Result | Action |
|--------|--------|
| shrink@0.7 ≥ 1.0046 | **KILL** |
| league 39 worse than 0.9526 by **>0.003** (i.e. > **0.9556**), or 2026 worse than 1.0265 by **>0.002** (i.e. > **1.0285**) | **KILL** even if pooled looks better |
| 1.0036 < shrink@0.7 < 1.0046 | **HOLD** — keep candidate, do not copy `sweep_best/` |
| shrink@0.7 ≤ **1.0036** and kill lines clean | **PROMOTE**: backup `sweep_best/` then copy candidate; shrink stays 0.7; blend stays global 70/30 |

Do **not** retune α. Best-α on 518 is diagnostic only.

### Out of scope

Allowlist / blend_weights / league-gated CSV; half-life search; year-aware shrink; 6.2 market-only; injuries; draw adj; 20h rebuild; git commit / push; changing production unless PROMOTE.

### Acceptance criteria

All met (see verifier). Gate **KILL** applied; production unchanged.

---

## What changed

| Area | Change |
|------|--------|
| Trainer | `recency_sample_weight` / `train_recency_sample_weights`; optional `sample_weight` in `fit`; persist `train_recency_half_life_days` on pickle + `baseline_weights.json` |
| CLI | `--train-recency-half-life-days` → `train_and_save` only; sweep stays unweighted |
| Tests | `tests/test_calc/test_residual_ml_trainer.py` |
| Candidate | `models/residual_ml/recency_180_candidate/` |
| Eval | `artifacts/dc_rho_mle_promotion/gate_eval_recency_180.json` |
| Production HGB / blend / shrink / `production_profile.md` | **Unchanged** |

Live predict math is unchanged. Weights are train-only.

---

## Official 518 results

Command:

```
env -u PYTHONPATH python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/recency_180_candidate/model.pkl \
  --include-holdout \
  --json artifacts/dc_rho_mle_promotion/gate_eval_recency_180.json
```

Retrain: 2071 train / 518 val; train LL **0.8672**; val ML raw **1.0464** (ship pinned retrain was train ≈0.8372 / val raw ≈1.0149).

| Slice | n | Market | Blend | ML raw | Best α | Best shrink | Shrink@0.7 | Ship @0.7 | vs ship |
|-------|--:|-------:|------:|-------:|-------:|------------:|-----------:|----------:|--------:|
| pooled | 518 | 1.0068 | 1.0083 | 1.0464 | 0.9 | 1.0066 | **1.0088** | **1.0046** | **+0.0042 KILL** |
| year:2025 | 173 | 0.9647 | 0.9678 | 0.9879 | 0.8 | 0.9640 | 0.9645 | 0.9611 | +0.0034 |
| year:2026 | 345 | 1.0279 | 1.0287 | 1.0757 | 1.0 | 1.0279 | **1.0310** | **1.0265** | **+0.0045 KILL** (>1.0285) |
| league:39 | 135 | 0.9580 | 0.9538 | 0.9680 | 0.6 | 0.9496 | **0.9503** | **0.9526** | −0.0023 (improved) |
| league:41 | 67 | 1.0401 | 1.0427 | 1.1028 | 1.0 | 1.0401 | 1.0472 | 1.0407 | +0.0065 |
| league:180 | 158 | 1.0529 | 1.0542 | 1.0903 | 0.9 | 1.0520 | 1.0531 | 1.0503 | +0.0028 |

Best-α on 518 is **0.9 → 1.0066**, still worse than ship shrink@0.7 **1.0046**. Diagnostic only; α stays **0.7**.

---

## Gate decision: **KILL**

1. Pooled shrink@0.7 **1.0088 ≥ 1.0046** → **KILL**.
2. 2026 shrink@0.7 **1.0310 > 1.0285** → **KILL** even if pooled had looked better.
3. League 39 is **clean** (0.9503 < 0.9556) and slightly better than ship. That does not override the pooled/2026 kill lines.

`sweep_best/` was **not** copied. `docs/production_profile.md` was **not** updated. Blend stays global 70/30.

---

## Artifacts

- Candidate: `models/residual_ml/recency_180_candidate/` (`model.pkl`, `baseline_weights.json` has `train_recency_half_life_days: 180.0`)
- Eval JSON: `artifacts/dc_rho_mle_promotion/gate_eval_recency_180.json`
- Dataset (unchanged): `artifacts/dc_rho_mle_promotion/dataset.csv`
- Production: `models/residual_ml/sweep_best/` (still `dc_rho_mle_retrain`, 2026-09-04)

---

## Tests

```
env -u PYTHONPATH python -m pytest tests/test_calc/test_residual_ml_trainer.py tests/test_calc/ -q
```

**236 passed, 2 xfailed.**
