# Open Tasks (Public Summary)

This file tracks high-level planned work. Detailed working notes live in `docs/internal/` and are not published.

## Current Focus (Next 1-3 Sprints)
1. Run a provider field check for Piotroski F-Score inputs, especially long-term debt history and shares outstanding history.
2. Implement a partial-capable Piotroski F-Score only if the provider field check supports graceful degradation.
3. Decide whether to expose Piotroski through the MCP server immediately or keep it library-first until response contracts are stable.

## Upcoming (Later)
- Consider runtime refresh detection for `data/damodaran_macro.json` so long-running MCP servers can pick up updated macro defaults without restart.
- Revisit ADR-0001 4x4 fundamentals framework after the trust-release work is complete.
- Introduce Forecast object pipeline (ADR-0005) and validation helpers.
- HTTP/SSE MCP companion server (ADR-0006) with parity tests.
- Add valuation profiles (ADR-0004) and profile metadata in outputs.
- End-to-end example notebooks and screenshots.
- Calibration and backtesting of scoring thresholds (see Roadmap Phase 10 summary).

## Recently Completed
- Externalized macro valuation defaults into `data/damodaran_macro.json` and surfaced provenance in valuation assumptions.
- Added calculation hygiene guardrails for ratio denominators, CAGR endpoints, and WACC effective-tax-rate handling.
- Added assumptions/provenance manifests to valuation rows and orchestrator outputs.

## Maintenance
- Keep this summary aligned with Roadmap and ADR statuses after major updates.
