# 2026-09-05 — Why Premier League (39) is stable and Allsvenskan (180) flips

**Status:** PLAN ONLY. Do not train. Do not change production.  
**Question:** Why does league **39** keep beating market across slices, while league **180** wins on the 492 tuning slice and loses on the official 518 gate?  
**Output of this work:** a written answer, then a go/no-go on specialist models. Not a new HGB.

Ship today: global HGB, blend **70/30**, shrink **α=0.7**, official 518 **1.0046**. Canonical CSV: `artifacts/dc_rho_mle_promotion/dataset.csv` (2589 rows; do not rebuild). Existing evals: `gate_eval_slices.json` (518), `gate_eval_492_tuning.json` (492).

This is the diagnostic that was recommended **instead of** training league-specific HGBs.

---

## 1. Goal

Explain the 180 sign flip in plain language. Do **not** train a new model unless this diagnostic later says so (it will not train anything in this sitting).

Success is a sentence like:

- “180 flips because holdout Allsvenskan is X”, or
- “180 flips because DC is unstable on the later window”, or
- “180 flips because 518’s Allsvenskan mix is a different year / favourite / outcome set than 492.”

Then decide (section 6). Do not invent a sixth hypothesis after seeing plots.

---

## 2. Facts already known (do not rediscover)

Reuse the two JSON evals. Do not re-run `eval_residual_ml` just to reprint these cells.

| Slice | League | n | Market | DC | Blend | Shrink@0.7 | Blend vs market |
|-------|--------|--:|-------:|---:|------:|-----------:|-----------------|
| Official 518 | 39 Premier League | 135 | 0.9580 | 0.9566 | **0.9538** | **0.9526** | wins (biggest n≥15 league win) |
| Official 518 | 180 Allsvenskan | 158 | 1.0529 | 1.0796 | **1.0542** | 1.0503 | **loses** |
| Tuning 492 | 39 | 144 | 0.9513 | 0.9584 | **0.9491** | — | wins |
| Tuning 492 | 180 | 201 | 1.0782 | 1.0859 | **1.0733** | — | **wins** (why 180 was allowlisted) |

Full CSV (official time-split of **all** rows, then filter by league — not a per-league 80/20):

- **39:** 680 rows (545 before the 518 cut / 135 in 518 val)
- **180:** 853 rows (695 / 158)

Other context (do not redo):

- Recency-weighted HGB helped 39 (518 shrink@0.7 **0.9503** vs ship **0.9526**) and killed pooled/2026. Not promoted.
- P0 league-gated blend left 39 on 70/30, but the global HGB retrain **hurt** 39 ML (518 shrink@0.7 **0.9563** vs **0.9526**). KILL. Production still global 70/30.
- DC params (live, do not re-fit): 39 ρ **−0.0145**; 180 ρ **−0.1086** (`docs/production_profile.md`).

Visible from the table already (confirm on subsets, do not treat as the answer yet):

- 518 Allsvenskan market LL is **easier** than 492 (1.0529 vs 1.0782), but blend loses the race.
- 180 DC−market gap is **larger** on 518 (+0.0267) than on 492 (+0.0076).
- 39 blend wins on both slices. 39 DC is slightly **better** than market on 518 and slightly worse on 492; blend still wins.

---

## 3. Why 492 ≠ 518 + a few rows

Do **not** treat 518 as “492 plus holdout.” They are different recent windows.

Code path (`config/eval_protocol.py`, `src/calc/residual_ml/filters.py` `select_backtest_rows`, `src/utils/time_split.py`):

| Slice | How rows are chosen | n |
|-------|---------------------|--:|
| **Default 492** | Drop holdout draws **4951–4960 first** (`max_draw=TUNING_DRAW_MAX=4950`), **then** last 20% by `(match_date, match_id)` | 492 |
| **Official 518** | Last 20% of the **full** CSV (`--include-holdout` → no draw cap) | 518 |
| **Holdout-only** | Draws **4951–4960** only (~129 rows in the wiki). Sanity check — **not** the ship metric | count from CSV |

Constants: `HOLDOUT_DRAW_MIN=4951`, `TUNING_DRAW_MAX=4950`, `VALIDATION_FRACTION=0.20`. Wiki: [`docs/wiki.md`](../wiki.md) (Slice).

Because the time-split runs **after** the holdout filter, 492 and 518 are **not nested**:

- Some 492 rows fall **before** the 518 cut once holdout rows are added back.
- Some 518 rows are holdout (and possibly other late rows that were not in the 492 val).

Measure overlap. Do not assume `518 = 492 ∪ holdout`.

Row key: `match_id` (assert unique in the CSV). League key: `league_external_id` (int in the loader; compare as `str(...)` in `{"39", "180"}`).

---

## 4. Hypotheses (pre-registered)

Test these. Do not add a shopping list after seeing plots.

1. **Composition.** 518’s 180 rows are a different mix than 492’s 180: year, draw/date range, 1/X/2 rate, market-favourite strength / odds range.
2. **Holdout 4951–4960.** Allsvenskan-heavy, or unusually hard/easy vs market, enough to flip the 180 cell by itself.
3. **DC quality.** 180 DC LL vs market on 492 vs 518 (and vs 39 on the same row sets). If 180 DC collapses only on the later window, that is the story.
4. **Slice arithmetic.** How many 39 / 180 rows are in `492∩518`, `518−492`, `492−518`. If the flip lives only in `518−492`, say so.
5. **Outcome mix.** Realized 1/X/2 rates vs market / DC / blend predicted shares on those same sets.

---

## 5. Steps (one sitting, cheap)

No dataset rebuild. No HGB retrain. No blend/shrink/production change.

### Step A — Row inventory

For leagues **39** and **180**, on these sets:

`492` · `518` · `492∩518` · `518−492` · `492−518` · holdout `4951–4960` (all-rows, no time-split)

Print: n, `match_date` min/max, `draw_number` min/max, year counts (2025/2026), label counts (1/X/2).

### Step B — Market vs DC vs blend LL on the same sets

Reuse `score_baseline_log_losses` (same function as eval). Report market / DC / blend LL and n scored (DC may be fewer if null). Optional: shrink@0.7 on 518 subsets only if already cheap; **do not** retune α.

### Step C — Stop if holdout is the whole flip

The flip is “holdout explains 180” when **all** of these hold:

- On `492∩518` league 180: blend **still beats** market (same sign as 492).
- On `518−492` league 180 **or** holdout-only league 180: blend **loses** to market (same sign as 518).
- The 518 180 cell is then just those unique/holdout rows dragging a still-good overlap.

If that is true: **write the answer and stop.** Skip Step D. Do not train an Allsvenskan HGB.

If the overlap set **already flips**, holdout is not the whole story — continue to DC/composition on `492∩518` vs `518−492`.

### Step D — Favourite / match-type (only if A–C do not explain it)

Pre-registered split, not a hunt:

- Favourite = argmax of `(p_home_market_norm, p_draw_market_norm, p_away_market_norm)`.
- Bins: `p_fav ≥ 0.50` vs `< 0.50`; and home-favourite vs away-favourite.

Stop after one table. Do not add more bins.

---

## 6. Then decide (after the write-up, not now)

| Finding | Decision |
|---------|----------|
| Flip is composition and/or holdout (Step C or a clean mix story) | **Do not** train an Allsvenskan HGB |
| 39 blend (and shrink@0.7 on 518) beats market on **492, 518, and both years** inside those windows | Optional **later** PL-only model; bar locked vs official 518 league-39 shrink@0.7 **0.9526**. Not this plan. |
| Neither is a clean story | **Stop specialist models.** Keep the global HGB. |

Do not allowlist-shop. Do not treat a 492-only 180 win as a reason to specialize.

---

## 7. Success

A written answer in `docs/reports/` dated the day the diagnostic is run (not this plan). One primary cause. Fill:

```
180 flips because: …
39 stays green because: …
Decision: no Allsvenskan HGB / optional later PL-only with bar 0.9526 / stop specialists.
```

---

## 8. Out of scope

- League-gated blend redo
- Recency-weighted HGB redo
- Allowlist shopping from 518 (or any new allowlist)
- Training Premier League or Allsvenskan HGBs **as part of this plan**
- Dataset rebuild, DC re-fit, α retune, production copy
- Git commit / push
- New product behavior or live scoring changes

---

## 9. Commands

From repo root. No `PYTHONPATH`. Existing slice evals (already run; re-run only if JSON is missing):

```bash
env -u PYTHONPATH python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --json artifacts/dc_rho_mle_promotion/gate_eval_492_tuning.json

env -u PYTHONPATH python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --include-holdout \
  --json artifacts/dc_rho_mle_promotion/gate_eval_slices.json
```

Inventory + LL on custom sets (stdin script; cwd = repo root so `src.*` imports resolve):

```bash
env -u PYTHONPATH python <<'PY'
from collections import Counter
from pathlib import Path

from src.calc.residual_ml import load_dataset_rows, select_backtest_rows
from src.calc.residual_ml.evaluation import score_baseline_log_losses
from src.calc.residual_ml.filters import filter_rows_by_draw
from config.eval_protocol import HOLDOUT_DRAW_MIN, TUNING_DRAW_MAX

LEAGUES = ("39", "180")
csv_path = Path("artifacts/dc_rho_mle_promotion/dataset.csv")
all_rows = load_dataset_rows(csv_path)
ids = [row["match_id"] for row in all_rows]
assert len(ids) == len(set(ids)), "match_id is not unique"

rows_492, _ = select_backtest_rows(all_rows, max_draw=TUNING_DRAW_MAX)
rows_518, _ = select_backtest_rows(all_rows)
holdout = filter_rows_by_draw(
    all_rows, min_draw=HOLDOUT_DRAW_MIN, max_draw=None
)
assert all(int(row["draw_number"]) >= HOLDOUT_DRAW_MIN for row in holdout)

def by_id(rows):
    return {row["match_id"]: row for row in rows}

map_492, map_518 = by_id(rows_492), by_id(rows_518)
sets = {
    "492": rows_492,
    "518": rows_518,
    "492∩518": [map_492[i] for i in map_492.keys() & map_518.keys()],
    "518−492": [map_518[i] for i in map_518.keys() - map_492.keys()],
    "492−518": [map_492[i] for i in map_492.keys() - map_518.keys()],
    "holdout": holdout,
}

def league_rows(rows, league):
    return [row for row in rows if str(row.get("league_external_id")) == league]

def inventory(rows):
    dates = [str(row.get("match_date") or "") for row in rows]
    draws = [int(row["draw_number"]) for row in rows]
    years = Counter(date[:4] if len(date) >= 4 else "?" for date in dates)
    labels = Counter(str(row.get("label")) for row in rows)
    return {
        "n": len(rows),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "draw_min": min(draws) if draws else None,
        "draw_max": max(draws) if draws else None,
        "years": dict(years),
        "labels": dict(labels),
    }

print(f"pooled 492={len(rows_492)} 518={len(rows_518)} "
      f"intersect={len(sets['492∩518'])} "
      f"518-only={len(sets['518−492'])} 492-only={len(sets['492−518'])} "
      f"holdout={len(holdout)}")

for league in LEAGUES:
    print(f"\n=== league {league} ===")
    for name, rows in sets.items():
        subset = league_rows(rows, league)
        inv = inventory(subset)
        losses = score_baseline_log_losses(subset) if subset else {}
        print(name, inv)
        for baseline, (loss, count) in losses.items():
            loss_s = "—" if loss is None else f"{loss:.4f}"
            print(f"  {baseline}: {loss_s} (n={count})")
PY
```

Optional holdout pooled sanity (no per-league table):

```bash
env -u PYTHONPATH python -m src.scripts.backtest_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --include-holdout \
  --min-draw 4951 \
  --all-rows
```

Write findings to `docs/reports/YYYY-MM-DD-pl-vs-allsvenskan-flip.md` on the day you run it. Do not claim this plan file is that result.

---

## 10. Kill / stop rules

- **Stop** when Step C fires or when one of the five hypotheses is a clean written answer.
- **Do not** train PL or Allsvenskan HGBs in this work.
- **Do not** rebuild CSV, re-fit DC, retune α, or copy `sweep_best/`.
- **Do not** compare 492 LL to 518 LL as if they were the same rows.
- **Do not** use holdout-only LL as a ship metric.
- Timebox: one sitting. If the story is still muddy after A–D, the decision is **stop specialist models**, not “try another cut.”
