# Value Investing Tools - Development Roadmap

_Last updated: 2026-02-22_

This roadmap tracks target architecture from ADR-0001 through ADR-0006.
It is a planning document, not a completion log.
Product positioning and scope guardrails live in `docs/NORTH_STAR.md`.

## How to Read This Roadmap
This file tracks **development phases** only. Phases describe planning sequence and design decisions, not release readiness.
For “what’s safe to use now,” rely on the README’s **Status & Readiness** labels (`Stable`, `Preview`, `Experimental`).

## Phase Status Conventions
- **Planned**: not started
- **Active**: in progress
- **Completed**: delivered to main
- **Parked**: on hold, not dropped

## Current Implementation Snapshot

Implemented in the current repository:
- `compute_fundamentals_actuals()` and `compute_fundamentals_scores()` (12-metric model).
- Valuation flows: `dcf_three_scenarios()`, `compare_to_market_ev()`, `compare_to_market_cap()`.
- MCP stdio server in `server.py`.

Not implemented yet (still planned):
- ADR-0001 full 4x4 fundamentals framework (`compute_fundamentals_score` singular API).
- ADR-0002 Piotroski module (`piotroski_fscore`, `plot_piotroski_fscore`).
- ADR-0004 valuation profile system and `valuation_profile` metadata.
- ADR-0005 `Forecast` dataclass pipeline.
- ADR-0006 HTTP/SSE server companion (`server_http.py`) and shared manifest tests.

Status labels used below:
- `Planned`: target work not complete in current tree.
- `In Progress`: partially implemented.
- `Completed`: delivered and validated with tests/docs.

Status clarification (2026-02-22): this document is an aspirational roadmap. Completion criteria listed below are targets unless explicitly validated in code/tests.

---

## Phase 1 - Stabilise Defaults & Documentation
Status: `In Progress`

Goal:
- Ensure valuation defaults and usage docs are consistent.

Targets:
- Validate baseline defaults in code and README.
- Add quick-reference defaults section.
- Add minimal smoke checks for one sample ticker flow.

Deliverable:
- Documentation and defaults aligned to implementation.

---

## Phase 2 - Fundamentals Revamp (ADR-0001)
Status: `Planned`

Goal:
- Implement MECE 4x4 framework with rank/absolute/z-score modes.

Targets:
- 4 pillars x 4 sub-factors.
- Risk pillar includes Altman Z and Beneish M.
- Unified response schema with data-health diagnostics.
- Plotting support for sub-factors and overlays.

Completion criteria:
- `tests/test_fundamentals_contract.py`
- threshold/directionality tests
- Financials-sector Altman warning test
- golden snapshot for canonical peer set

---

## Phase 3 - Piotroski F-Score (ADR-0002)
Status: `Planned`

Goal:
- Add standalone Piotroski reference module.

Targets:
- `piotroski_fscore(ticker|statements)`
- `plot_piotroski_fscore(...)`
- Optional orchestrator integration

Completion criteria:
- Component-flag unit tests (all 9 signals)
- README references and attribution section updates

---

## Phase 4 - Fundamentals + Peer Presentation Layer
Status: `Planned`

Goal:
- Produce consolidated fundamentals-first reports.

Targets:
- Standardized flow: Absolute -> Peer -> (Optional) Piotroski
- Harmonized JSON/DataFrame/figure outputs
- Data-health section in report outputs

Completion criteria:
- notebook demo
- schema validation test for report contract

---

## Phase 5 - Valuation Profiles (ADR-0004)
Status: `Planned`

Goal:
- Add Conservative / Moderate / Speculative profiles for valuation modules.

Targets:
- Profile argument support in:
  - `compare_to_market_ev()`
  - `compare_to_market_cap()`
  - `dcf_three_scenarios()`
- Echo effective assumptions in outputs.

Completion criteria:
- profile metadata contract test

---

## Phase 6 - Forecast Object (ADR-0005)
Status: `Planned`

Goal:
- Separate forecast schema from valuation engines.

Targets:
- `Forecast` dataclass
- forecast validation helper
- valuation functions accept ticker or forecast object

Completion criteria:
- forecast validation tests
- valuation audit-table test on toy case

---

## Phase 7 - Valuation Wrapper Hardening
Status: `In Progress`

Goal:
- Improve reliability and diagnostics in EV/DCF wrappers.

Targets:
- tighten error handling and notes
- plot smoke checks and integration checks
- add regression tests for valuation bridges

Completion criteria:
- integration tests for EV/cap/DCF wrappers
- chart smoke tests with non-zero artifacts

---

## Phase 8 - End-to-End Examples and Documentation
Status: `Planned`

Goal:
- Publish end-to-end notebook examples and screenshots.

Targets:
- staples peer workflow notebook
- tech workflow notebook with manual assumptions
- update README function map and examples
- provider roadmap: add an FMP (Financial Modeling Prep) provider behind the adapter interface once demand is validated

Completion criteria:
- reproducible notebooks with saved outputs

---

## Phase 9 - MCP Transport Expansion (ADR-0006)
Status: `Planned`

Goal:
- Support both stdio and HTTP/SSE transports with consistent contracts.

Targets:
- keep `server.py` as stdio default
- add `server_http.py` with mirrored endpoints
- maintain a shared manifest for transport mapping

Completion criteria:
- contract parity tests between stdio and HTTP/SSE responses

---

## Phase 10 - Calibration and Backtesting
Status: `Planned`

Goal:
- Empirically calibrate scoring thresholds and weights.

Targets:
- backtest quartiles over historical data
- review threshold distributions by sector
- publish calibration notes and versioned updates

Completion criteria:
- reproducible calibration scripts and report artifact
Output target: Calibrated and empirically supported scoring system.

Notes (summary of the calibration plan):
- Define a reproducible backtest dataset and label universe (time window, sector coverage, survivorship handling).
- Compute score distributions by sector and market-cap buckets to evaluate stability.
- Backtest quartiles against forward returns and drawdown metrics.
- Tune thresholds/weights conservatively and version any changes.
- Publish a calibration report and scripts for reproducibility.
