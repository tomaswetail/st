# 2026-09-04 — Classic DC fail-closed + slice ML/shrink

**Decision:** **APPROVED**  
**Date:** 2026-09-04 23:10  
**Owner:** Project Leader  
**Implementation:** [developer](a1a0f5e8-6a39-4afa-8a0b-9cba25029c9b)  
**Verification:** [verifier](42b2979a-e3a0-43ac-897c-71b102ed5de7) — VERIFIED, 26 tests passed

---

## Outcome

Two P0 items after the MLE-ρ freeze:

1. **Fail-closed classic DC.** When `RESIDUAL_ML_DC_ENGINE=classic` and fit/predict misses, `p_*_dc` and engine λs are **None**. Blend falls back to **market** only. Strength DC is no longer silently mixed into the same columns.
2. **Where the current ship edge lives.** Official 518-row include-holdout, current ship (MLE CSV + `sweep_best/` HGB, shrink α=0.7): pooled **1.0046** vs market **1.0068** (−0.0022). The edge is **mixed**, with more of it in **2025** and **league 39**. 2026 is most of the rows but a smaller per-row gain; Championship (41) is flat-to-slightly worse at production α. **No α retune, no model promotion.**

| Item | Value |
|------|--------|
| Official 518-row include-holdout ship metric | **1.0046** (α=**0.7**) — unchanged |
| Stretch ≤0.99 | still **MISSED** |
| Fail-closed | live + future assemble only; existing CSV not rebuilt |

---

## Task contract

### Goal

1. Classic miss must not write strength-DC into `p_*_dc`. Engine `None` → blend uses market (`blend_baselines` already does this).
2. Eval must report ML raw + shrink on year and league slices (same fields as pooled), then run the official 518-row include-holdout eval and state where the current ship edge lives.

Canonical ship after the MLE-ρ freeze: official 518-row gate **1.0046** (α=0.7). Phase 6.1 1.0022 is historical. Measure the **current ship** (MLE CSV + `sweep_best/` pinned HGB, shrink 0.7). Do not chase another 0.012.

### Current behavior (verified before change)

- `_engine_probabilities`: engine ≠ `"classic"` → strength tuple (correct). Classic `None` → **returned strength tuple** (bug).
- `_classic_dc_prediction` logged “falling back to strength DC” then returned None; caller filled strength.
- `blend_baselines`: `if engine is None: return dict(market)` already.
- `build_multi_slice_report` passed `scoring` only into **pooled**. Year/league/etc. had `ml_log_loss` / `best_shrink_*` null.

### Required behavior

- Classic miss → `(None, None, None, None, None)` for engine λs and `p_*_dc`. Strength features (npxG, etc.) unchanged. Explicit `engine=strength` still uses strength DC.
- Log text: engine omitted / blend will use market.
- Every existing slice (year, league, quarter, availability, injury) scores ML+shrink when a trainer is available. Expose production shrink α=0.7 per slice.
- Run official 518-row eval; do not retune α or promote a model.

### Out of scope

Dataset rebuild, HGB retrain, league-gated blend, INV-001 `<=` change, git commit/push, promoting a different model, rewriting the existing CSV.

### Business invariants

- INV-003: 1X2 still normalize when present
- BR-003 leakage unchanged
- Dual baseline: market + **classic** DC; missing classic → market only, not a third engine
- Official gate remains 518-row include-holdout

### Acceptance criteria

All met (see verifier). Official slice artifact: [`artifacts/dc_rho_mle_promotion/gate_eval_slices.json`](../../artifacts/dc_rho_mle_promotion/gate_eval_slices.json).

---

## What changed

| Area | Change |
|------|--------|
| Assembler | Classic miss returns Nones, not strength DC. Log: “classic engine omitted, blend will use market”. |
| Eval | `build_multi_slice_report` takes `trainer` and scores ML+shrink on every slice. `SliceMetrics` adds `production_shrink_alpha` / `production_shrink_log_loss` (α=0.7). |
| CLI | `eval_residual_ml` passes trainer + production shrink from `DataSourceConfig`. |
| Tests | Fit-fail and predict-fail assembler cases; year/league ML fields; pooled matches direct `run_backtest_scoring`; no-trainer stays null. |
| Dataset / HGB | **Unchanged.** Fail-closed is live + future assemble only. |

Live `ProbabilityManager` uses the same assembler, so fail-closed applies to live scoring automatically.

---

## Where the current ship edge lives

Official command (518/2589 rows, `--include-holdout`, `sweep_best/`):

```
python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --include-holdout \
  --json artifacts/dc_rho_mle_promotion/gate_eval_slices.json
```

Pooled matches the frozen gate: market **1.0068**, blend **1.0083**, ML raw **1.0149**, best shrink **1.0046** (α=0.7), shrink@0.7 **1.0046**.

Δ = shrink@0.7 − market (negative = beats market).

| Slice | n | Market | Blend | ML raw | Best α | Best shrink | Shrink@0.7 | Δ vs market |
|-------|--:|-------:|------:|-------:|-------:|------------:|-----------:|------------:|
| pooled | 518 | 1.0068 | 1.0083 | 1.0149 | 0.7 | **1.0046** | **1.0046** | **−0.0022** |
| year:2025 | 173 | 0.9647 | 0.9678 | 0.9633 | 0.4 | 0.9602 | 0.9611 | **−0.0036** |
| year:2026 | 345 | 1.0279 | 1.0287 | 1.0408 | 0.8 | 1.0264 | 1.0265 | −0.0014 |
| league:39 | 135 | 0.9580 | 0.9538 | 0.9567 | 0.5 | 0.9513 | 0.9526 | **−0.0054** |
| league:180 | 158 | 1.0529 | 1.0542 | 1.0569 | 0.6 | 1.0501 | 1.0503 | −0.0026 |
| league:41 | 67 | 1.0401 | 1.0427 | 1.0596 | 0.9 | 1.0398 | 1.0407 | +0.0006 |
| league:42 | 22 | 1.0660 | 1.0678 | 1.0751 | 0.7 | 1.0626 | 1.0626 | −0.0034 |
| league:45 | 19 | 0.9820 | 0.9300 | 0.9951 | 0.8 | 0.9800 | 0.9800 | −0.0020 |

Other leagues on this gate are n&lt;15 (noisy; not used for ship claims). Draw-quarter Q1–Q3 are empty (all 518 rows fall in Q4).

**Plain-language answer**

- The pooled −0.0022 vs market is **not** a 2026-only or Allsvenskan-only story.
- **2025** (173 rows): larger per-row edge (−0.0036). ML raw already beats market; shrink helps a bit more.
- **2026** (345 rows, ~2/3 of the gate): smaller per-row edge (−0.0014). ML raw is **worse** than market (1.0408); shrink does the work.
- Rough contribution to the pooled Δ: 2025 ≈ −0.0012, 2026 ≈ −0.0009 — **mixed**, with 2025 contributing slightly more despite fewer rows.
- **League 39** (Premier League, 135): largest n≥15 league edge (−0.0054). Blend is already strong (0.9538).
- **League 180** (Allsvenskan, 158): modest (−0.0026), similar to the pooled rate.
- **League 41** (Championship, 67): slightly **worse** than market at production α=0.7 (+0.0006).
- League 42 / 45 (n=22 / 19): small-n; 45’s blend (0.9300) is much stronger than ML — do not over-read.

**Do not retune α or promote a model from this eval.** Slice-best α (0.4 in 2025, 0.8 in 2026, 0.5 in 39, 0.9 in 41) is in-sample on that slice. Production stays **0.7**.

---

## Tests

```
python -m pytest tests/test_calc/test_residual_ml_feature_assembler.py tests/test_calc/test_eval_residual_ml.py -q
```

Developer: **17 passed**. Verifier also ran `tests/test_calc/test_residual_ml_baseline.py`: **26 passed**.
