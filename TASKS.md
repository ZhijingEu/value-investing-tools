# Open Tasks (Public Summary)

This file tracks high-level planned work. Detailed working notes live in `docs/internal/` and are not published.

## Current Focus (Next 1-3 Sprints)
1. Review the July trust-release changes in real MCP usage and confirm the new manifest/data-health outputs are easy for LLM clients to interpret.
2. Park larger feature work until the manifest, macro defaults, and calculation-hygiene changes have been user-tested.
3. Choose the next small reversible batch after testing feedback.

## Upcoming (Later)
- Consider runtime refresh detection for `data/damodaran_macro.json` so long-running MCP servers can pick up updated macro defaults without restart.
- Revisit ADR-0001 4x4 fundamentals framework after the trust-release work is complete.
- Introduce Forecast object pipeline (ADR-0005) and validation helpers.
- HTTP/SSE MCP companion server (ADR-0006) with parity tests.
- Add valuation profiles (ADR-0004) and profile metadata in outputs.
- End-to-end example notebooks and screenshots.
- Calibration and backtesting of scoring thresholds (see Roadmap Phase 10 summary).
- Reconsider Piotroski F-Score only if a provider adapter can supply all nine canonical inputs reliably; see ADR-0002.

## Recently Completed
- Externalized macro valuation defaults into `data/damodaran_macro.json` and surfaced provenance in valuation assumptions.
- Added calculation hygiene guardrails for ratio denominators, CAGR endpoints, and WACC effective-tax-rate handling.
- Added assumptions/provenance manifests to valuation rows and orchestrator outputs.
- Deferred Piotroski F-Score to avoid shipping a non-canonical or proxy-based implementation under the Piotroski name.

## Maintenance
- Keep this summary aligned with Roadmap and ADR statuses after major updates.
