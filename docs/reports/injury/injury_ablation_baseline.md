# Injury regression ablation baseline

Reference metrics before isolating injury vs blend vs retrain confounders.
See [`docs/reports/ml_residual/baseline_after_dc_tune.md`](../ml_residual/baseline_after_dc_tune.md) for Phase 1 canonical baseline.

---

## Problem summary

Post-injury rebuild **regressed** pooled best-shrink log loss on the 518-row include-holdout slice:

| Metric | Phase 1 (pre-injury) | Post-rebuild |
|--------|----------------------|--------------|
| Market LL | 1.0068 | 1.0068 |
| Blend LL | 1.0057 | 1.00566 |
| ML raw LL | 1.0091 | **1.0152** |
| Best shrink LL | **1.0022** (α=0.5) | **1.0068** (α=0.9) |

Sources:

- Phase 1: [`docs/reports/ml_residual/baseline_after_dc_tune.md`](../ml_residual/baseline_after_dc_tune.md), [`artifacts/eval_post_dc_20250902.json`](../../../artifacts/eval_post_dc_20250902.json)
- Post-rebuild: [`artifacts/phase5_holdout_eval_after_rebuild.json`](../../../artifacts/phase5_holdout_eval_after_rebuild.json)

---

## Bundled changes in post-rebuild

The rebuild was **not** injury-only:

1. Injury columns in `data/residual_ml/dataset.csv` (`has_availability`, counts, `missing_value_*`)
2. Conditional blend enabled (`config/blend_weights.json`, many rows **85/15** via `market_vs_dc`)
3. Draw adjustment disabled at eval time (`enabled: false`) but rebuild path still writes pre/post draw columns
4. HGB retrain with new sweep best (`max_depth` 3, `label_smoothing` 0.1)

---

## Injury coverage (current CSV)

Audit: [`artifacts/injury_coverage_audit.json`](../../../artifacts/injury_coverage_audit.json)

Run: `PYTHONPATH=src python src/scripts/audit_injury_coverage.py`

| Metric | Value |
|--------|-------|
| Rows | 2589 |
| `has_availability=1` | **68.33%** |
| Rows with tiny scaled `missing_value_*` (<1e-4) | **1769** |

Tiny scaled values come from API-Football storing **unavailable counts** as `home_missing_value` (÷1e6 → ~1e-6). Phase 2 counts-only constraint; calculator now treats these as `None`.

---

## Ablation matrix (B0–B3)

Configs under [`config/injury_ablation/`](../../../config/injury_ablation/). Orchestration:

```bash
PYTHONPATH=src python src/scripts/run_injury_ablation.py --step all
```

| ID | Blend config | Injury cols in HGB | Dataset source |
|----|--------------|-------------------|----------------|
| **B0** | `enabled: false` (70/30 fallback) | yes | Reblend from current CSV |
| **B1** | same as B0 | **no** (`--exclude-injury-features`) | Reuse B0 CSV |
| **B2** | conditional on, `injury_uncertainty` off | yes | Reblend |
| **B3** | full conditional (production-like) | yes | Reblend |

Results: [`docs/reports/injury/injury_phase2_results.md`](injury_phase2_results.md), [`artifacts/injury_ablation_summary.json`](../../../artifacts/injury_ablation_summary.json)

**Outcome (2025-09-03):** Phase 2 exit **FAIL**. Production: blend off, injury excluded from HGB. Best shrink **1.0067** vs Phase 1 **1.0022**.

---

## Decision rules (from correction plan)

- **B0 ≈ 1.0022** but **B3 worse** → conditional blend caused regression → fix blend (C2)
- **B1 ≈ B0** and both worse than Phase 1 → injury columns add noise → fix features/coverage (C1)
- **B1 better than B0** → injury features actively hurt → ship without them or regularize

---

## Success criteria

| Criterion | Target |
|-----------|--------|
| Pooled LL (518 include-holdout, best shrink) | ≤ **1.0022** |
| Injury-heavy slice LL gain vs no-injury features | ≥ **0.003** |
| ML raw vs market on tuning slice | not worse by > **0.005** |
| Holdout draws 4951–4960 | no regression vs market-only (report honestly) |
