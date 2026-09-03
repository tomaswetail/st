# Phase 2 injury correction — results (2025-09-03)

Ablation and production fix after pooled log-loss regression following the injury rebuild.
Baseline reference: [`docs/reports/injury/injury_ablation_baseline.md`](injury_ablation_baseline.md), Phase 1 [`docs/reports/ml_residual/baseline_after_dc_tune.md`](../ml_residual/baseline_after_dc_tune.md).

---

## Problem recap

| Metric | Phase 1 | Post-injury rebuild |
|--------|---------|---------------------|
| Market LL (518, include-holdout) | 1.0068 | 1.0068 |
| Blend LL | 1.0057 | 1.00566 |
| ML raw LL | 1.0091 | **1.0152** |
| Best shrink LL | **1.0022** (α=0.5) | **1.0068** (α=0.9) |

Sources: [`artifacts/phase5_holdout_eval.json`](../../../artifacts/phase5_holdout_eval.json), [`artifacts/phase5_holdout_eval_after_rebuild.json`](../../../artifacts/phase5_holdout_eval_after_rebuild.json).

---

## Injury coverage

| Metric | Pre-correction CSV | Corrected CSV |
|--------|-------------------|-----------------|
| Rows | 2589 | 2589 |
| `has_availability=1` | 68.33% | 68.33% |
| Tiny scaled `missing_value_*` rows | 1769 | **0** (patched) |

Audits: [`artifacts/injury_coverage_audit.json`](../../../artifacts/injury_coverage_audit.json), [`artifacts/injury_coverage_audit_corrected.json`](../../../artifacts/injury_coverage_audit_corrected.json).

Root cause: API-Football stores unavailable **counts** as `home_missing_value`; ÷1e6 → ~1e-6 noise for HGB. Fixed in `PlayerAvailabilityCalculator` and CSV reblend patch.

---

## B0–B3 ablation (518-row include-holdout)

Fast reblend from injury rebuild CSV (blend rules only; features unchanged). Full sweep (81 trials) per config.

| ID | Blend config | Injury in HGB | ML raw | Blend | Best shrink | α |
|----|--------------|---------------|--------|-------|-------------|---|
| **Phase 1** | fixed 70/30 | no | 1.0091 | 1.0057 | **1.0022** | 0.5 |
| Post-rebuild | conditional full | yes | 1.0152 | 1.00566 | 1.0068 | 0.9 |
| **B0** | off (70/30 fallback) | yes | 1.0136 | 1.0068 | 1.0066 | 0.9 |
| **B1** | off | **no** | 1.0141 | 1.0068 | 1.0067 | 0.9 |
| **B2** | on, injury rule off | yes | 1.0169 | 1.0056 | 1.0068 | 1.0 |
| **B3** | full conditional | yes | 1.0145 | 1.00566 | **1.0054** | 0.7 |

Artifacts: [`artifacts/injury_ablation_summary.json`](../../../artifacts/injury_ablation_summary.json), per-config JSON under `artifacts/injury_ablation_B*.json`.

### Interpretation

1. **Conditional blend contributed to regression** — B0/B1 beat B2; disabling blend rules removes the worst shrink (α→1.0 market fallback in B2).
2. **Injury HGB columns are not the primary driver** — B1 ≈ B0 (Δ shrink ≈ 0.0001).
3. **Phase 1 LL not restored** — best ablation shrink (B3 **1.0054**) remains **+0.0032** above Phase 1 **1.0022**. Likely compound drift: injury-enriched feature space, hyperparam shift (sweep picks depth=3/smooth=0.1 vs Phase 1 depth=4/smooth=0.05), and partial reblend in early B0 runs (293 rows kept old 85/15 before reblend fix).
4. **B3 best among ablations** but still misses exit gate; production ships simpler B1-like config (blend off, no injury in HGB) for stability.

---

## Production fix (shipped)

Applied **C1 + C2** from correction plan:

| Change | Detail |
|--------|--------|
| **C2 blend** | `config/blend_weights.json` → `enabled: false`, `injury_uncertainty.enabled: false` |
| **C1 features** | Count-proxy `missing_value_*` → `None` in calculator; CSV patch in reblend |
| **Dataset** | `data/residual_ml/dataset_corrected.csv` promoted → `data/residual_ml/dataset.csv` (all rows 70/30 blend) |
| **HGB** | Retrain with `--exclude-injury-features` → `models/residual_ml/sweep_best/` |
| **Eval** | [`artifacts/injury_fix_eval.json`](../../../artifacts/injury_fix_eval.json) |

### Production fix metrics (518 include-holdout)

| Metric | Post-rebuild | Production fix | Δ vs rebuild |
|--------|--------------|----------------|----------------|
| ML raw | 1.0152 | 1.0141 | −0.0011 |
| Best shrink | 1.0068 (α=0.9) | 1.0067 (α=0.9) | −0.0001 |
| vs Phase 1 shrink | +0.0046 | +0.0045 | — |

**Verdict:** Stops further degradation vs post-rebuild; **does not** recover Phase 1 **1.0022**.

---

## Availability slices (B0, validation)

| Slice | Rows | Blend LL |
|-------|------|----------|
| `availability:0` | 185 | 0.9996 |
| `availability:1` | 333 | 1.0109 |
| `injury_heavy:>0.0` | 294* | 1.0089 |

\*Pre-patch threshold; after CSV patch most `missing_value_difference` are empty — injury-heavy slice not actionable until real market values or count-only features are re-evaluated.

No ≥ **0.003** injury-heavy LL gain observed; injury signal not additive vs market at coupon time.

---

## Phase 2 exit criteria

| Criterion | Target | Result |
|-----------|--------|--------|
| Pooled best shrink ≤ Phase 1 | **1.0022** | **FAIL** — best fix **1.0067**, best ablation **1.0054** |
| Pooled gain vs pre-injury | ≥ **0.005** | **FAIL** — regression ~0.0045 |
| Injury-heavy slice gain | ≥ **0.003** | **FAIL** — no measurable gain |
| ML raw vs market (tuning) | not worse by >0.005 | **FAIL** — ML raw ~1.014 vs market 1.007 |
| Holdout 4951–4960 | no regression vs market | **PASS** (unchanged; market ~0.983 on holdout-only — see Phase 5 artifacts) |

**Phase 2 exit: FAIL.** Injury infrastructure remains (snapshots, calculator, CSV columns) but **not used in HGB training** until signal is proven. Conditional blend **disabled** in production.

---

## Tooling added

- `scripts/audit_injury_coverage.py`
- `scripts/reblend_residual_ml_dataset.py` — fast blend ablation without full rebuild
- `scripts/run_injury_ablation.py` — B0–B3 orchestration
- `scripts/train_residual_ml.py --exclude-injury-features`
- Eval slices: `availability:0/1`, `injury_heavy:>T`

---

## Recommended next steps (out of scope for this pass)

1. Full dataset rebuild with calculator fix (not reblend-only) when DB time allows.
2. Retrain with Phase 1 hyperparam grid pinned (depth=4, smooth=0.05) on Phase-1-like CSV to test dataset drift hypothesis.
3. Use **counts only** (`home_unavailable_count`, `away_unavailable_count`) in a separate experiment — not scaled missing values.
4. Re-enable conditional blend only after row-level A/B shows pooled gain on include-holdout.
