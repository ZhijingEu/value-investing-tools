# Value Investing Tools – Development Roadmap

_Last updated: 2025-10-09_

This roadmap reflects the architectural direction and sequencing agreed in ADR-0001 (Fundamentals 4×4 Framework) and ADR-0002 (Piotroski F-Score Module).  
The implementation approach is **modular and progressive** — starting with fundamentals and peer comparison, then extending to valuation, forecasting, and multi-agent integration.

Status clarification (2026-02-22): this document is an aspirational roadmap. Completion criteria listed below are targets unless explicitly validated in code/tests.

---

## Phase 1 — Stabilise Defaults & Documentation
**Goal:** Ensure current functions run consistently before introducing new frameworks.

- Validate and document valuation defaults  
  - Risk-free rate = 4.5 %  
  - Equity risk premium = 5.5 % (≈ 6 %)  
  - Fallback growth = 2 %  
  - FCF look-back window = 3 years  
- Update main README  
  - Clarify difference between *Implied Value* vs *DCF Scenarios*  
  - Add quick-reference “Defaults & Profiles” section  
- Minor smoke-tests using MDLZ example  

✅ **Output:** Documentation refresh and working baseline.

---

## Phase 2 — Fundamentals Revamp (ADR-0001)
**Goal:** Implement the new **MECE 4×4 Fundamentals Scorecard** with three scoring modes and weighted composite.

- Create unified data schema with four pillars × four sub-factors each  
  - *Profitability, Growth, Reinvestment, Risk*  
  - Risk pillar now includes **Altman Z** and **Beneish M**  
- Add scoring modes  
  - **Rank-based** (default)  
  - **Absolute thresholds** (1–5 bands per ADR-0001 table)  
  - **Peer-set Z-score** (dynamic per run)  
- Support multi-ticker input  
- Implement `data_health` reporting (warn if N < 5 tickers)  
- Drop tickers with missing sub-factors  
- Return unified structured object + DataFrames + Matplotlib figures  
- Extend plotting  
  - Annual sub-factor series  
  - YoY growth charts (for Growth metrics)  
  - Multi-ticker overlays  

**Artifacts / Completion Criteria**
- Contract test: tests/test_fundamentals_contract.py: Asserts the return schema keys: method, pillar_scores, overall_scores, subfactor_scores, data_health. and verifies tie-ranking average and N<5 warning present.
- Golden CSVs: tests/golden/MDLZ_peer_set_input.csv (5–6 tickers) and tests/golden/MDLZ_fundamentals_output.json (one canonical run).
- Threshold sanity test: Parametrized unit test that feeds synthetic metrics to hit each absolute band (1–5) and checks scores.
- Plot smoke test:Save one PNG per chart type; assert image created and non-zero size (no visual diffing yet).
- `tests/test_directionality_flags.py`: verify inversion logic for metrics where lower is better (e.g., D/E, Beneish M).
- `tests/test_altman_financial_warning.py`: ensure data_health warning triggers for Financials sector tickers.

✅ **Output:** `compute_fundamentals_score()` and `compute_fundamentals_actuals()` upgraded; end-to-end peer comparison working.

---

## Phase 3 — Piotroski F-Score Module (ADR-0002)
**Goal:** Add a credible standalone reference score without overlapping the 4×4 framework.

- Implement `piotroski_fscore(ticker|statements)` returning total (0–9) + component flags  
- Add `plot_piotroski_fscore()` visualisation  
- Integrate optional display into orchestrator output  
- Document academic references and formulae in README  
- Unit-test sample tickers for consistency  

**Artifacts / Completion Criteria**
- Component flags test: Synthetic two-period inputs that flip each of the 9 signals one by one; assert total equals sum of flags.
- Docs note: Source references + any licenses in README section “Attribution”.

✅ **Output:** F-Score appears as optional reference panel alongside the Fundamentals summary.

---

## Phase 4 — Presentation & Orchestrator (Up to Peers)
**Goal:** Produce consolidated reports for fundamentals and peer analysis only.

- Standardise orchestrator flow  
  - **Absolute → Peer → Reference (Piotroski)**  
- Include Data Health section and per-metric annual charts  
- Harmonise JSON / DataFrame / figure outputs for easy downstream use  
- Stop execution before any valuation logic  

✅ **Output:** One-call notebook or MCP tool generates a practitioner-ready fundamentals report.

**Artifacts / Completion Criteria**
- End-to-end notebook: notebooks/peer_report_demo.ipynb that runs Absolute → Peer → (Optional) Piotroski, saving outputs.
- JSON schema check:schemas/fundamentals_report.schema.json and test that validates orchestrator output.

---

## Phase 5 — Profiles for Valuation Modules
**Goal:** Introduce **Conservative / Moderate / Speculative** configuration profiles for valuation-related tools only.

- Apply profiles to  
  - `compare_to_market_ev()`  
  - `compare_to_market_cap()`  
  - `dcf_three_scenarios()`  
- Profiles adjust WACC, growth, terminal-g assumptions; fundamentals remain profile-agnostic.  
- Add profile metadata to output JSON for traceability.  

**Artifacts / Completion Criteria**
- `tests/test_profiles_echo.py` verifies `valuation_profile` metadata (name + effective params) in outputs.

✅ **Output:** Valuation modules parameterised for scenario analysis.

---

## Phase 6 — Forecast Object & Valuation Engine
**Goal:** Build a clean separation between data, forecast assumptions, and valuation computation.

- Define a reusable `Forecast` dataclass (schema)  
- Implement forecast generators (manual entry + growth-based)  
- Add detailed PV breakdown tables (per year + terminal component)  
- Retain compatibility with DCF and implied EV functions  

**Artifacts / Completion Criteria**
- `tests/test_forecast_validator.py` rejects non-monotonic years, NaNs, length mismatches, missing fields.
- `tests/test_valuation_audit_table.py` validates PV and terminal value against a hand-computed toy case (within tolerance).

✅ **Output:** Valuation engine with auditable forecast inputs.

---

## Phase 7 — Valuation Wrappers & Charts
**Goal:** Modernise comparison utilities and visualisations.

- Refactor  
  - `compare_to_market_ev()`  
  - `compare_to_market_cap()`  
  - `dcf_three_scenarios()`  
- Add charts  
  - Market vs Intrinsic EV  
  - DCF vs Price bands  
  - Sensitivity plots (g vs WACC)  
- Integrate profile controls.  

**Artifacts / Completion Criteria**
- `tests/test_compare_to_market_ev_integration.py` end-to-end integration with Forecast engine and profile echo present.
- `tests/test_dcf_three_scenarios_plots.py` chart smoke tests (non-zero image size, expected titles/labels).
- `tests/golden/valuation_output.json` snapshot for one canonical ticker (regression guard).

✅ **Output:** Valuation visualisation suite aligned with forecast engine.

---

## Phase 8 — End-to-End Examples & Docs
**Goal:** Publish reference notebooks and user documentation.

- Case 1 – Staples sector peer comparison  
- Case 2 – Tech sector with manual forecasts  
- Write narrative: **Absolute → Peer → Valuation**  
- Update README with screenshots and function map.  

✅ **Output:** Two ready-to-run Jupyter notebooks + illustrated README.

---

## Phase 9 — MCP Integration & HTTP/SSE Server
**Goal:** Connect tools to agentic workflows (Claude Desktop / ChatGPT Dev Mode).

- Update `server.py` (stdio MCP) to include new fundamentals and F-Score tools  
- Add `server_http.py` (HTTP / SSE) for future ChatGPT integration  
- Optional: configure ngrok / localtunnel exposure for testing  

**Artifacts / Completion Criteria**
-  Tool manifest test: Both stdio and HTTP handlers return identical JSON for the same input.

✅ **Output:** Agents can invoke fundamentals and valuation functions locally or via HTTP.

---

## Phase 10 — Calibration & Back-Testing
**Goal:** Validate threshold bands and scoring weights.

- Back-test quartile performance on historical data  
- Review threshold distributions and adjust band limits  
- Document findings and future enhancements  

✅ **Output:** Calibrated and empirically supported scoring system.
