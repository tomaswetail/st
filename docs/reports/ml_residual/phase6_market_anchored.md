# Phase 6.2 — Market-anchored residual HGB

Diagnostic train of **one** HGB residual **against market** (not the 70/30 blend) on the same archived Phase 1 CSV and pinned hyperparameters as Phase 6.1. Default: **do not** promote to production.

References: [`docs/reports/ml_residual/phase6_restore_phase1.md`](phase6_restore_phase1.md), [`docs/reports/ml_residual/baseline_after_dc_tune.md`](baseline_after_dc_tune.md).

---

## Decision: **FAIL**

Gate A **PASS** (tuning-slice ML raw beats market by 0.0407). Gate B **FAIL** (official 518 best shrink 1.0056 > Phase 1 1.0022). Gate C **PASS** (holdout 4951–4960 best shrink equals market at α=1.0).

Overall 6.2 requires A and B and C. **FAIL.**

This does **not** claim pooled LL ≤ 0.99. Official include-holdout best shrink is **1.0056475802248581**. Gate A is a clear pass, not CLOSE, so no hyperparameter sweep was run.

**Production left untouched:** `models/residual_ml/sweep_best/` and `models/residual_ml/phase1_restore/` were not overwritten. Leftover `models/residual_ml/market_only/` was not overwritten. `config/blend_weights.json` and `config/draw_adjustment.json` remain `"enabled": false`.

---

## Commands used

From repo root (`PYTHONPATH=src`). No `--sweep`. Explicit `--output-dir` so `--market-only-baseline` does not redirect to `models/residual_ml/market_only/`. No custom `--validation-fraction` (default 0.20). Time-split is first 80% of the archived CSV by `match_date`/`match_id`; no extra holdout-draw exclusion at train time.

### Train

```bash
export PYTHONPATH=src
python -u scripts/train_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --output-dir models/residual_ml/phase6_market_anchored \
  --version phase6_market_anchored \
  --max-depth 4 \
  --learning-rate 0.03 \
  --max-iter 100 \
  --label-smoothing 0.05 \
  --exclude-injury-features \
  --market-only-baseline
```

`--exclude-injury-features` is a no-op here (archived CSV has no injury columns). Trainer `random_state` default is 42.

### Eval (official include-holdout)

```bash
python src/scripts/eval_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model models/residual_ml/phase6_market_anchored/model.pkl \
  --include-holdout \
  --json artifacts/phase6_market_anchored_eval.json
```

Eval auto-detected `is_market_only_weights` (1.0 / 0.0) and rewrote `p_*_blend := p_*_market_norm` before scoring. JSON `baselines.blend` is therefore **post-rewrite** and equals market.

### Eval (tuning slice)

```bash
python src/scripts/eval_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model models/residual_ml/phase6_market_anchored/model.pkl \
  --json artifacts/phase6_market_anchored_eval_tuning.json
```

Default eval: time-split validation, holdout excluded (`>4950`). **492 rows.** Same market-only rewrite.

### Holdout-only (draws 4951–4960)

```bash
python src/scripts/backtest_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model models/residual_ml/phase6_market_anchored/model.pkl \
  --all-rows \
  --min-draw 4951 \
  --max-draw 4960 \
  > artifacts/phase6_market_anchored_holdout_only.txt
```

`--all-rows` is required so the time-split does not drop rows. Do not pass `--include-holdout` together with `--max-draw`. **129 rows.**

The model was **not** fit on 4951–4960 as a reserved set. Time-split first 80% contained **0** of those draws (train draws 4760–4920; all 129 holdout rows fell in the validation suffix). Shrink α was not retuned on this slice.

---

## Training stdout summary

| Field | Value |
|-------|-------|
| Dataset rows | 2589 |
| Market-only baseline | yes (`blend := market_norm`) |
| Train rows | 2071 |
| Validation rows | 518 |
| Train log loss | 0.8357 |
| Validation log loss (ML raw) | 1.0185 |
| Market validation log loss | 1.0068 |
| Blend validation log loss | 1.0068 (equals market after rewrite) |
| Model path | `models/residual_ml/phase6_market_anchored/model.pkl` |
| `baseline_weights.json` | `market_weight` 1.0 / `dc_weight` 0.0 |

---

## 6.1 restore vs 6.2 market-anchored

6.1 sources: [`artifacts/phase6_restore_phase1_eval.json`](../../../artifacts/phase6_restore_phase1_eval.json) (include-holdout); Phase 1 / 6.1 same-recipe tuning from [`artifacts/eval_post_dc_20250902.json`](../../../artifacts/eval_post_dc_20250902.json) (6.1 matched Phase 1 include-holdout exactly). 6.2 sources: [`artifacts/phase6_market_anchored_eval.json`](../../../artifacts/phase6_market_anchored_eval.json), [`artifacts/phase6_market_anchored_eval_tuning.json`](../../../artifacts/phase6_market_anchored_eval_tuning.json). Blend-as-stored vs after-rewrite from as-stored CSV baselines vs eval rewrite (confirmed with the same `score_baseline_log_losses` / `run_backtest_scoring` path the backtest uses).

### Include-holdout (518 rows, official)

| Metric | 6.1 restore (70/30) | 6.2 market-anchored |
|--------|---------------------|---------------------|
| Rows | 518 | 518 |
| Market LL | 1.0067856568672326 | 1.0067856568672326 |
| Blend-as-stored (70/30 on CSV) | 1.0056569644642361 | 1.0056569644642361 |
| Blend-after-market-only | n/a (70/30 model) | 1.0067856568672326 (= market) |
| ML raw | 1.0090960766726624 | 1.0184939058536717 |
| Best shrink | **1.0021925884473784 (α=0.5)** | 1.0056475802248581 (α=0.8) |

Eval JSON `baselines.blend` for 6.2 is post-rewrite (1.0067856568672326), not blend-as-stored.

### Tuning slice (492 rows, holdout excluded)

| Metric | 6.1 / Phase 1 (70/30) | 6.2 market-anchored |
|--------|-----------------------|---------------------|
| Rows | 492 | 492 |
| Market LL | 1.02905231971842 | 1.02905231971842 |
| Blend-as-stored (70/30 on CSV) | 1.0230805847414515 | 1.0230805847414515 |
| Blend-after-market-only | n/a (70/30 model) | 1.02905231971842 (= market) |
| ML raw | 0.9771044324416095 | 0.98835045808935 |
| Best shrink | 0.9771044324416095 (α=0.0) | 0.98835045808935 (α=0.0) |

6.2 ML raw beats market on this slice (Gate A) but is worse than 6.1’s blend-anchored ML raw.

---

## Holdout-only vs market (draws 4951–4960)

Source: [`artifacts/phase6_market_anchored_holdout_only.txt`](../../../artifacts/phase6_market_anchored_holdout_only.txt). Full precision from the same backtest scoring functions.

| Metric | Value |
|--------|-------|
| Rows | 129 |
| Market LL | 0.982585140938047 |
| Blend-as-stored | 0.990809134892062 |
| Blend-after-market-only | 0.982585140938047 (= market) |
| ML raw | 1.0228422537645376 |
| Best shrink | 0.982585140938047 (α=1.0) |

Best shrink equals market (α=1.0 = “use market”). Raw ML is worse than market. No retune on this slice.

Backtest printed as-stored baselines first (blend 0.9908), then auto-detected market-only weights and printed blend-after-market-only (0.9826 = market), then ML/shrink.

---

## Gates

| Gate | Slice | Rule | Result |
|------|-------|------|--------|
| A. ML raw vs market | Tuning `slices.pooled` (492) | `market_LL - ml_raw_LL >= 0.003` | **PASS** — 1.02905231971842 − 0.98835045808935 = **0.04070186162907** |
| B. Best shrink vs Phase 1 | Official 518 include-holdout | best shrink **≤ 1.0022** | **FAIL** — 1.0056475802248581 |
| C. Holdout 4951–4960 | 129 rows | ML+best-shrink **≤** market | **PASS** — equality at 0.982585140938047 (α=1.0) |

Gate A close/wide-fail bands (not used): CLOSE if −0.003 < (market − ML raw) < 0.003; WIDE FAIL if `ml_raw_LL > market_LL + 0.003`. Observed delta is +0.0407 (beats market). **Not CLOSE. Not a wide fail.** Optional 81-trial sweep skipped.

**Overall 6.2: FAIL** (B failed).

Eval auto-applied the market-only rewrite: model weights are 1.0 / 0.0; include-holdout and tuning JSON `baselines.blend` equal `baselines.market`.

---

## What was not done

- No overwrite of `models/residual_ml/sweep_best/`, `models/residual_ml/phase1_restore/`, or leftover `models/residual_ml/market_only/`
- No change to `config/blend_weights.json` or `config/draw_adjustment.json`
- No change to `data/residual_ml/dataset.csv`
- No `build_residual_ml_dataset.py`
- No hyperparameter sweep (Gate A passed; CLOSE rule did not fire)
- Injuries not re-enabled in HGB, conditional blend, or draw adjustment
- Phase 6.3 (draw adjustment) not started
- No promotion to production / no copy into `sweep_best`
- No git commit
- No assembler / trainer / eval code changes
