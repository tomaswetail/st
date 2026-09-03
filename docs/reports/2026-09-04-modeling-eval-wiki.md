# 2026-09-04 — Modeling & evaluation wiki

**Status:** APPROVED  
**Task:** Create `docs/wiki.md` (modeling/eval jargon glossary) and light integration pointers; docs only; no commit.

## Outcome

Approved after one developer pass and independent verifier **PASS**. No rework required (one non-blocking nit on home/away Brier keys vs draw Brier in `score_outcome_metrics`).

## Deliverables

### Primary

- [`docs/wiki.md`](../wiki.md) — glossary for Slice, HGB residual, Logit, Shrink / Shrink to market, Blend, Brier; short see-also for Market/Engine baselines, Log loss, Draw adjustment.

### Integration

- `AGENTS.md` — wiki in Before-implementing list (step 3) and “Also useful”
- `docs/DOMAIN.md` — top pointer + Blend/Residual ML evidence links
- `docs/ARCHITECTURE.md` — docs layout table
- `docs/project_status.md` — one-liner to wiki
- `docs/product/probability_calculations.md` — production shrink α corrected **0.9 → 0.5** (optional, allowed)

## Accuracy anchors

- Production shrink **α=0.5** (`production_profile.md`, `DataSourceConfig`)
- Official gate slice: **518-row include-holdout**
- Code: `baseline.py` (`apply_residual_deltas`, `shrink_toward_market`, `blend_baselines`), `evaluation.py` (`_binary_brier`), `trainer.py` (`residual_logit_v1`)

## Non-goals (honored)

- No report moves, no historical α=0.9 cleanup across old reports, no git commit, no application code changes

## Agents

- Developer: [create wiki](08c70718-c002-4a96-82cc-fff1ffef4493)
- Verifier: [verify wiki](2c2ab1d7-8c56-4d4e-bcd9-6f9b14cb4990)
