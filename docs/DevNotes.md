# Developer Notes (DEVNOTES.md)
_Repository: Value Investing Tools_  
_Last updated: 2025-10-09_

This file serves as a running log of technical and conceptual decisions across phases.  
It complements the ADRs and ROADMAP, providing context for key changes, rationale, and implementation nuances.

---
## 2025-10-10 — Clarifications Added (ADR-0001)
- Defined that “bigger = better” applies to **scores**, not raw metric values; direction handled per-metric via flags.
- Added explicit note on **Altman Z** threshold applicability (non-financial calibration) and corresponding data-health warning.

## 2025-10-09 — Fundamentals Redesign (ADR-0001)
- **Objective:** Rebuild the fundamentals module into a **MECE 4×4 framework** with balanced pillars and transparent scoring.
- **Pillars/Sub-Factors:**  
  - Profitability: ROE, Net Margin, FCF Margin, ROIC  
  - Growth: Revenue CAGR, Net Income CAGR, FCF CAGR, Total Assets CAGR  
  - Reinvestment: CapEx/Revenue, Fundamental Reinvestment Rate, Reinvestment/NOPAT, Retention Ratio  
  - Risk: Debt/Equity, Operating Leverage, **Altman Z**, **Beneish M**
- **Scoring Modes:**  
  1. Rank-based (default) — peer-relative.  
  2. Absolute thresholds (1–5 bands per ADR-0001 table).  
  3. Peer-set Z-score (dynamic per run).
- **Weighting Formula:**  
  `1.0 × Profitability + 0.7 × Reinvestment + 0.8 × Growth + 1.8 × Risk`  
  (Overrideable in function call.)
- **Data Health Logic:**  
  - Drop tickers with missing sub-factors.  
  - Warn if peer set < 5 tickers.
- **Output Contract:**  
  Return structured object:  
  `{method, tickers, pillar_scores, overall_scores, subfactor_scores, data_health}`  
  plus DataFrames + Matplotlib figures.
- **Plotting Extensions:**  
  - Annual series per sub-factor.  
  - YoY growth plots (for Growth metrics).  
  - Multi-ticker overlays supported.
- **Outcome:**  
  Clear, balanced scoring methodology aligned with practitioner fundamentals analysis.

---

## 2025-10-09 — Integration of Altman Z and Beneish M (ADR-0001)
- **Change:** Both scores relocated inside the **Risk pillar** as direct sub-factors.
- **Reasoning:**  
  - They measure intrinsic financial and accounting risk.  
  - Keeping them within Risk avoids fragmentation and duplication.  
- **Implementation:**  
  - Each computed per ticker per time window.  
  - Values normalised according to threshold bands.  
  - Visualization retained via factor-level plots.

---

## 2025-10-09 — Piotroski F-Score as Standalone Module (ADR-0002)
- **Objective:** Provide an empirical benchmark alongside fundamentals output.
- **Design:**  
  - `piotroski_fscore(ticker|statements)` returns total (0–9) and individual binary flags.  
  - `plot_piotroski_fscore(result_dict)` visualises all nine signals + total.  
  - Optional reference panel added to orchestrator output.  
- **Signals Implemented:** ROA > 0, ΔROA > 0, CFO > 0, CFO > ROA, ΔLeverage < 0, ΔCurrent Ratio > 0, No New Shares, ΔGross Margin > 0, ΔAsset Turnover > 0.
- **Outcome:**  
  Stand-alone, non-overlapping credibility module integrated cleanly into reporting.

---

## 2025-10-09 — Roadmap Reordering
- **Old:** Valuation-first sequence.  
- **New:** Fundamentals-first sequence for faster tangible output and testability.
- **Phase Priorities:**
  1. Fundamentals 4×4 framework (ADR-0001)
  2. Piotroski F-Score (ADR-0002)
  3. Orchestrator presentation (fundamentals + peers)
  4. Profiles (valuation only)
  5. Forecast object + valuation engine
  6. MCP/HTTP integration

---

## 2025-10-09 — Data Quality Rules
- All sub-factors computed over consistent years where possible.  
- Growth uses CAGR; others use averages.  
- Missing data triggers exclusion of ticker.  
- Data health warnings summarised for user visibility.

---

## 2025-10-09 — Plotting Standardisation
- All visualisation functions should:
  - Return both Matplotlib figures and underlying DataFrames.  
  - Use consistent colour themes (muted, non-commercial palette).  
  - Support multiple tickers per chart.  
  - Label each chart with factor name, units, and time window.
- Growth plots include YoY change charts in addition to raw values.

---

## 2025-10-09 — Fallback Valuation Defaults
- ERP: 6.0 %  
- Risk-free rate: 4.5 %  
- Fallback terminal growth: 2.0 %  
- FCF averaging window: 3 years  
- Purpose: provide conservative baseline assumptions until profiles are introduced.

---

## 2025-10-09 — Planned Next Steps
1. Implement and test the new `compute_fundamentals_score()` schema.  
2. Add Altman Z and Beneish M calculations directly into Risk pillar.  
3. Finalise and validate threshold bands.  
4. Implement Piotroski F-Score and visualisation.  
5. Update orchestrator and notebooks to reflect new outputs.  
6. Prepare documentation and examples for Phase 2 & 3 completion.

---

## Maintenance Notes
- Continue tracking all architectural changes in ADRs (one ADR per major decision).  
- Each new module must include:
  - Function signature and parameters in docstring.  
  - Example input/output JSON snippet.  
  - Reference to applicable ADR.  
- For dependencies: prefer code adaptation with attribution over new package imports.  
- Keep all plots deterministic (seeded) for reproducibility.
