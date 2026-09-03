# 2026-09-03 — Docs reports restructure

**Status:** APPROVED  
**Task:** Create `docs/reports/{injury,ml_draw,ml_residual}/`, move experiment write-ups, add `docs/project_status.md`, fix links/script paths.

## Outcome

Approved after developer implementation + two rework iterations (artifact README link depth; snapshot script template depth). Verifier: VERIFIED (final).

## Deliverables

### `docs/reports/` tree

```
docs/reports/
  2026-09-03-docs-reports-restructure.md   (this report)
  injury/
    injury_ablation_baseline.md
    injury_phase2_results.md
  ml_draw/
    draw_driver_analysis.md
    draw_formula_report.txt
  ml_residual/
    baseline_after_dc_tune.md
    log_loss_0.99_roadmap.md
    phase6_market_anchored.md
    phase6_restore_phase1.md
```

### New status doc

- `docs/project_status.md` — Phase 6.1 ship snapshot (best shrink ~1.0022, gate MISSED); aligned with `production_profile.md`.

### Link / script path updates

- Core: `production_profile.md`, `ARCHITECTURE.md` (docs layout), `product/probability_calculations.md`, `AGENTS.md`
- Moved reports: relative depths for artifacts/config/models and cross-category links
- Scripts: `analyze_draw_drivers.py`, `generate_draw_report_pdf.py`, `snapshot_post_dc_baseline.py` (path + `../../` depth), `eval_draw_adjustment_oos.py`
- Artifact: `artifacts/baseline_post_dc_tune_20250902/README.md`

### Intentionally left

- `docs/prompts/`
- `docs/away_drivers_roadmap_update_8622bb6a.plan.md` (stale sync-copy path)
- `production_profile.md` at docs root
- Historical roadmap prose / `scripts/` style links inside moved residual roadmap (pre-existing class of issue)

### Non-blocking notes

- Working tree may also contain broader `ARCHITECTURE.md` `src/` path rewrites, `stryktipset-predictions.md`, and `README.md` edits outside the minimal docs-move scope (pre-existing or adjacent; not reverted).
- No git commit performed for this task.
