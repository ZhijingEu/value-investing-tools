# ADR-0002: Piotroski F-Score – Stand-Alone Reference Module

## Status
Proposed (Draft, Not Yet Implemented in code as of 2026-02-22)

## Context
Following ADR-0001, the Altman Z-Score and Beneish M-Score are now integrated directly into the **Risk** pillar of the 4×4 fundamentals framework.  
However, the **Piotroski F-Score** remains a valuable, empirically tested signal of financial strength among value stocks.

Integrating the F-Score into the 4×4 framework would blur conceptual boundaries and reduce interpretability.  
Instead, it will be implemented as an **independent reference function and visualization module** that complements, but does not affect, the composite fundamentals score.

---

## Decision
Implement a dedicated Piotroski F-Score module that reproduces all **nine binary signals** exactly as defined in Piotroski (2000).

### Function Signatures
```python
piotroski_fscore(ticker_or_statements) -> {"total": int(0..9), "signals": {...}}
plot_piotroski_fscore(result_dict) -> matplotlib.figure.Figure


## Model Overview

| # | Signal | Category | Description | Good Condition (Score = 1) |
|---|---------|-----------|--------------|-----------------------------|
| 1 | ROA > 0 | Profitability | Positive Return on Assets | ROA > 0 |
| 2 | Δ ROA > 0 | Profitability | Improving ROA vs prior year | ROA increase |
| 3 | CFO > 0 | Profitability | Positive operating cash flow | CFO > 0 |
| 4 | CFO > ROA | Profitability | Quality of earnings | CFO > ROA |
| 5 | Δ Leverage < 0 | Leverage | Lower leverage (YoY) | Debt reduction |
| 6 | Δ Current Ratio > 0 | Liquidity | Improved short-term liquidity | Ratio increase |
| 7 | No new shares issued | Leverage | No equity dilution | Shares unchanged |
| 8 | Δ Gross Margin > 0 | Efficiency | Improving margins | Margin increase |
| 9 | Δ Asset Turnover > 0 | Efficiency | Better asset utilisation | Turnover increase |

Total F-Score = sum of nine signals (0 – 9).
Higher values indicate stronger fundamentals and financial health.

---

## Implementation Notes

- Compute from the same financial statements used in the fundamentals module.
- Use the latest two reporting periods for all “Δ” comparisons.
- Output both the total score and the individual component flags for transparency.
- plot_piotroski_fscore() visualises each binary signal (bar or traffic-light chart) plus the aggregate score.
- Integrate into notebooks and orchestrator outputs as an optional “Reference Module – Piotroski F-Score.”

---

## Consequences
- Adds a credible, easily interpreted benchmark without overlapping with the 4×4 framework.
- Enhances diagnostic insight for value-oriented investors.
- Minimal maintenance overhead (inputs already available from existing parsers).
- Provides a consistent extension point for future empirical or academic score modules.


