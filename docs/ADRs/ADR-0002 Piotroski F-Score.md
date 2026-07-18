# ADR-0002: Piotroski F-Score Deferral

## Status
Deferred as of 2026-07-18

## Context
Piotroski F-Score is a recognized accounting-quality screen built from nine binary financial-statement signals. Its usefulness comes from being canonical and comparable: a 0-9 score means the same thing across implementations only when all nine criteria are calculated faithfully.

VIT's default data path is intentionally low-friction and local-agent friendly, using yfinance without user API keys or subscriptions. That design supports the repo's zero-setup value proposition, but it does not reliably provide every input needed for a canonical Piotroski implementation across tickers.

The main blockers are:
- historical shares outstanding / share issuance data for the no-dilution signal;
- consistent long-term debt tagging across companies;
- occasional gaps or alternate labels for current assets, current liabilities, gross profit, and revenue;
- the need to align annual reporting periods cleanly across income statement, balance sheet, and cash-flow data.

## Decision
Do not implement Piotroski F-Score in the current yfinance-only default workflow.

VIT should not ship a custom or modified F-Score under the Piotroski name. A partial or proxy-based score would overlap with the existing fundamentals scoring framework while reducing methodological clarity.

Piotroski may be reconsidered only if one of the following changes:
- a provider adapter supplies all nine canonical inputs with reliable annual history;
- an optional SEC EDGAR/XBRL provider is implemented with robust ticker-to-CIK mapping, concept aliases, fiscal-year alignment, and duplicate/restatement handling;
- an optional FMP or equivalent provider is added, with clear API-key handling and documented coverage limits.

## Canonical Criteria

| # | Signal | Required Inputs | Current Default Data Risk |
|---|---|---|---|
| 1 | ROA > 0 | Net income, beginning/average assets | Low |
| 2 | CFO > 0 | Operating cash flow | Low |
| 3 | Delta ROA > 0 | Two years of net income and assets | Low |
| 4 | CFO > net income | Operating cash flow, net income | Low |
| 5 | Leverage down | Long-term debt, assets, two years | Medium |
| 6 | Current ratio up | Current assets, current liabilities, two years | Medium |
| 7 | No new shares issued | historical shares outstanding / issuance | High |
| 8 | Gross margin up | Gross profit or revenue and COGS, two years | Medium |
| 9 | Asset turnover up | Revenue, assets, two years | Low |

## Consequences
- Avoids presenting a non-canonical score as if it were Piotroski F-Score.
- Preserves VIT's positioning as an auditable, reproducible toolkit rather than a loose collection of heuristic screens.
- Keeps focus on the existing fundamentals, peer, valuation, data-health, and provenance layers.
- Leaves a clear re-entry path if provider coverage changes.

## Related Notes
If future work adds SEC EDGAR/XBRL support, treat Piotroski as a provider-validation project first and a scoring project second. The first deliverable should be a field-availability report, not the score itself.
