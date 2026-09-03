#!/usr/bin/env python3
"""Generate PDF report: how draw (X) probability is calculated in this repo.

```bash
PYTHONPATH=src python src/scripts/generate_draw_report_pdf.py
# default output: docs/draw_calculation_report.pdf
```
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from utils.repo_paths import repo_root, resolve_repo_path

DEFAULT_OUTPUT = repo_root() / "docs" / "draw_calculation_report.pdf"


class DrawReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 8, "Stryktipset - Draw (X) probability calculation", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str) -> None:
        self.ln(4)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 13)
        self.multi_cell(0, 7, title)
        self.ln(2)

    def subsection(self, title: str) -> None:
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def mono(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Courier", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, f"  - {text}")


def build_report() -> FPDF:
    pdf = DrawReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "How Draw (X) Probability Is Calculated")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        "This report describes the end-to-end path from market odds and Dixon-Coles "
        "engine output to the final draw probability used in Stryktipset 1X2 predictions. "
        "It focuses on the explicit draw-adjustment layer (Phase 3) and includes "
        "worked numeric examples.",
    )

    pdf.section_title("1. Pipeline overview")
    pdf.body(
        "For each coupon match, probabilities are built in a fixed order. "
        "Only steps 4-6 change the draw (X) beyond what market and DC already imply."
    )
    pdf.mono(
        "Market odds  -->  normalize\n"
        "Dixon-Coles  -->  p_home_dc, p_draw_dc, p_away_dc\n"
        "Conditional blend  -->  p_*_blend (pre-draw)\n"
        "Draw adjustment  -->  p_*_blend (draw layer)\n"
        "HGB residual ML  -->  optional logit deltas vs blend\n"
        "Shrink toward market  -->  final 1X2"
    )
    pdf.body(
        "Code entry points: calc/probability_manager.py (live), "
        "calc/residual_ml/dataset.py (training CSV), config/draw_adjustment.json (coefficients)."
    )

    pdf.section_title("2. Market baseline")
    pdf.body(
        "Raw Stryktipset odds p_home_market, p_draw_market, p_away_market are "
        "normalized so they sum to 1.0 (overround removed proportionally)."
    )
    pdf.subsection("Example")
    pdf.mono(
        "Raw odds:  Home 1.80  Draw 3.50  Away 4.20\n"
        "Implied:   1/1.80=0.556  1/3.50=0.286  1/4.20=0.238  (sum=1.079)\n"
        "Normalized market:\n"
        "  p_home_market_norm = 0.556/1.079 = 0.515\n"
        "  p_draw_market_norm = 0.286/1.079 = 0.265\n"
        "  p_away_market_norm = 0.238/1.079 = 0.220"
    )

    pdf.section_title("3. Dixon-Coles engine")
    pdf.body(
        "Classic DC fits attack/defence strengths per league and produces model "
        "probabilities p_home_dc, p_draw_dc, p_away_dc (also normalized to sum to 1)."
    )
    pdf.subsection("Example (illustrative)")
    pdf.mono(
        "  p_home_dc = 0.42\n"
        "  p_draw_dc = 0.30\n"
        "  p_away_dc = 0.28"
    )

    pdf.section_title("4. Conditional blend (before draw adjustment)")
    pdf.body(
        "Market and DC are mixed in probability space (not logit space). "
        "Default weights are 70% market / 30% DC; config/blend_weights.json can "
        "shift weights when |market_vs_dc| is large, DC fit is poor, or injury data "
        "is missing (has_availability=0)."
    )
    pdf.subsection("Formula")
    pdf.mono(
        "p_draw_blend_pre = w_market * p_draw_market_norm + w_dc * p_draw_dc\n"
        "(same for home and away, then renormalize if needed)"
    )
    pdf.subsection("Example (w_market=0.7, w_dc=0.3)")
    pdf.mono(
        "p_draw_blend_pre = 0.7 * 0.265 + 0.3 * 0.30\n"
        "                 = 0.1855 + 0.09\n"
        "                 = 0.2755\n"
        "\n"
        "Full blend vector (same weights on each outcome):\n"
        "  p_home_blend_pre = 0.7*0.515 + 0.3*0.42 = 0.486\n"
        "  p_draw_blend_pre = 0.2755\n"
        "  p_away_blend_pre = 0.7*0.220 + 0.3*0.28 = 0.238"
    )

    pdf.section_title("5. Draw adjustment (explicit draw layer)")
    pdf.body(
        "This is the shipped Phase 3.2 rule in calc/draw_adjustment.py. "
        "It nudges only the draw probability in logit space, then renormalizes "
        "home / draw / away so they still sum to 1. It can be turned off via "
        '"enabled": false in config/draw_adjustment.json.'
    )
    pdf.subsection("Formula")
    pdf.mono(
        "delta = beta0 + sum(beta_i * feature_i)\n"
        "logit(p_draw') = logit(p_draw_blend) + delta\n"
        "p_draw' = inv_logit(...)\n"
        "Keep p_home and p_away at blend values; set p_draw = p_draw';\n"
        "Renormalize: p_k = p_k / (p_home + p_draw' + p_away)"
    )
    pdf.body(
        "where logit(p) = ln(p / (1-p)) and inv_logit is the inverse. "
        "Missing features use 0.0 (configurable). Coefficients come from "
        "interpretable discovery (L1 logistic on historical data), then a sparse "
        "refit checked on validation draw Brier / log loss."
    )
    pdf.subsection("Current config features (when enabled)")
    pdf.bullet("away_npxg_for (negative beta: lower draw when away attacks more)")
    pdf.bullet("away_short_rest (positive: more draw when away on short rest)")
    pdf.bullet("away_npxg_against")
    pdf.bullet("combined_low_scoring_rate")
    pdf.bullet("congestion_difference")
    pdf.bullet("intercept (beta0) ~ 0.106 in current config")

    pdf.subsection("Worked example")
    pdf.body(
        "Suppose after blend: p_home=0.486, p_draw=0.276, p_away=0.238. "
        "Only away_short_rest=1 (away played recently); all other features=0."
    )
    pdf.mono(
        "beta0 = 0.106\n"
        "beta_away_short_rest = 0.204\n"
        "delta = 0.106 + 0.204*1 = 0.310\n"
        "\n"
        "logit(0.276) = ln(0.276/0.724) = -0.962\n"
        "logit + delta = -0.962 + 0.310 = -0.652\n"
        "p_draw' = 1/(1+exp(0.652)) = 0.343\n"
        "\n"
        "Before renormalize: (0.486, 0.343, 0.238) sum = 1.067\n"
        "After renormalize:\n"
        "  p_home = 0.486/1.067 = 0.456\n"
        "  p_draw = 0.343/1.067 = 0.321  (+4.5 pp vs blend)\n"
        "  p_away = 0.238/1.067 = 0.223"
    )
    pdf.body(
        "Home and away probabilities fall slightly because draw mass increased "
        "and the triplet must sum to 1."
    )

    pdf.section_title("6. HGB residual model (optional)")
    pdf.body(
        "If RESIDUAL_ML_ENABLED and a model file exists, HistGradientBoosting "
        "predicts logit deltas vs the draw-adjusted blend for all three outcomes. "
        "This is a separate ML model from draw discovery; it learns residuals, "
        "not a standalone draw formula."
    )
    pdf.mono(
        "final_logit_k = logit(p_k_blend) + delta_k_ML\n"
        "then softmax / normalize to 1X2"
    )
    pdf.body(
        "Training targets use p_*_blend columns in dataset.csv (after draw adj "
        "when enabled at rebuild time). For a fair ablation without draw formula, "
        "set draw_adjustment enabled=false and rebuild the dataset before retraining."
    )

    pdf.section_title("7. Shrink toward market")
    pdf.body(
        "After ML, final probabilities can be pulled toward market odds: "
        "p_final = (1-alpha)*p_ML + alpha*p_market (per outcome). "
        "Typical production alpha is 0.5 (RESIDUAL_ML_FINAL_SHRINK_TO_MARKET). "
        "This affects draw as well as home/away."
    )

    pdf.section_title("8. Discovery vs production draw rule")
    pdf.body(
        "Phase 3.1 (scripts/analyze_draw_drivers.py) runs L1 logistic / elastic net "
        "on historical rows to find which features move P(draw) beyond p_draw_blend. "
        "That analysis does not run in production."
    )
    pdf.body(
        "Phase 3.2 ships a small fixed rule (at most 8 terms) in JSON. "
        "scripts/eval_draw_adjustment_oos.py refits betas and enables the config "
        "only if validation draw metrics improve vs blend-only."
    )

    pdf.section_title("9. Where draw probability is stored")
    pdf.bullet("Live: STMatchProbabilityResult.probabilities['X']")
    pdf.bullet("Dataset CSV: p_draw_blend (after adj), p_draw_blend_pre_draw (before)")
    pdf.bullet("Evaluation: binary draw log loss uses label=='X' vs p_draw")

    pdf.section_title("10. Quick reference commands")
    pdf.mono(
        "# Discover drivers (analysis only)\n"
        "PYTHONPATH=src python src/scripts/analyze_draw_drivers.py \\\n"
        "  --dataset data/residual_ml/dataset.csv \\\n"
        "  --write-doc docs/reports/ml_draw/draw_driver_analysis.md\n"
        "\n"
        "# OOS gate + optional config update\n"
        "PYTHONPATH=src python src/scripts/eval_draw_adjustment_oos.py --write-config\n"
        "\n"
        "# Rebuild CSV with draw layer baked in\n"
        "PYTHONPATH=src python -u src/scripts/build_residual_ml_dataset.py\n"
        "\n"
        "# Disable draw adj: set enabled=false in config/draw_adjustment.json, then rebuild"
    )

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        4,
        "Generated by scripts/generate_draw_report_pdf.py - "
        "see calc/draw_adjustment.py and docs/reports/ml_draw/draw_driver_analysis.md for details.",
    )
    return pdf


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path (default: docs/draw_calculation_report.pdf)",
    )
    args = parser.parse_args()
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = build_report()
    pdf.output(str(output_path))
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
