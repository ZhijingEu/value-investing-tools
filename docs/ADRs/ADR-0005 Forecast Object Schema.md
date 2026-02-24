# ADR-0005: Forecast Object Schema – Standardised Structure for Forward Estimates

## Status
Proposed (Draft)

## Context
Valuation functions within Value Investing Tools require **future financial projections** to estimate intrinsic value through Discounted Cash Flow (DCF) and related methods.  
Currently, each function (e.g., `dcf_three_scenarios()`, `compare_to_market_ev()`) handles forecast data independently, leading to:

- Duplicated logic for growth extrapolation and discounting.  
- Inconsistent time horizons and key fields (revenue, FCF, reinvestment, etc.).  
- Limited ability to ingest **manual user forecasts** for “what-if” analysis.  

This ADR defines a unified **Forecast object schema** to represent forward financial estimates consistently across modules.

---

## Decision
Create a reusable, lightweight data container called **`Forecast`** that encapsulates all projected inputs required for valuation calculations.

### Class Definition (Dataclass Example)

```python
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict

@dataclass
class Forecast:
    ticker: str
    start_year: int
    end_year: int
    forecast_years: pd.DataFrame        # year-indexed: revenue, EBIT, NOPAT, reinvestment, FCF
    assumptions: Dict[str, float]       # growth_rate, terminal_g, wacc, tax_rate, capex_pct, etc.
    source: str = "generated"           # 'generated' | 'manual'
    notes: Optional[str] = None
```
Each valuation function can then accept a Forecast object directly or generate one internally.

---

## Required Fields (per Year)

| Field | Description | Example Source |
|--------|--------------|----------------|
| **Revenue** | Projected total revenue | Extrapolated from CAGR or user input |
| **EBIT** | Earnings before interest and taxes | Derived from margin assumptions |
| **NOPAT** | EBIT × (1 − Tax Rate) | Calculated |
| **Reinvestment** | CapEx + ΔWorking Capital | Derived |
| **Free Cash Flow (FCF)** | NOPAT − Reinvestment | Core valuation metric |
| **Terminal Value (TV)** | Optional; computed from final FCF × (1+g)/(WACC−g) | Calculated automatically |

---

## Usage Pattern

### Option 1 – Generated Forecast (Automated)
```
fc = generate_forecast(
    ticker="MSFT",
    growth_rate=0.06,
    horizon=5,
    wacc=0.085,
    tax_rate=0.21,
    source="generated"
)
dcf_three_scenarios(forecast=fc)
```
### Option 2 – Manual Forecast (User-Defined)
```
custom_df = pd.DataFrame({
    "year": [2025, 2026, 2027],
    "revenue": [150e9, 162e9, 175e9],
    "ebit": [60e9, 66e9, 72e9],
    "nopat": [47e9, 51e9, 56e9],
    "reinvestment": [15e9, 17e9, 18e9],
    "fcf": [32e9, 34e9, 38e9],
}).set_index("year")

fc = Forecast(
    ticker="MSFT",
    start_year=2025,
    end_year=2027,
    forecast_years=custom_df,
    assumptions={"growth_rate":0.06, "terminal_g":0.02, "wacc":0.085},
    source="manual"
)
dcf_three_scenarios(forecast=fc)
```

---

## Implementation
Add Forecast class to core valuation module (valuation_utils.py).

Implement helper functions:
- generate_forecast() → auto-builds from latest actuals.
- validate_forecast() → checks completeness & consistency.

Modify all valuation functions (dcf_three_scenarios, compare_to_market_ev, etc.) to accept either:
- a raw ticker (→ generates Forecast internally), or
- an explicit Forecast object (manual path).

Ensure all orchestrator, MCP, and plotting utilities can read/write this schema to JSON.

---

## Alternatives Considered
- Continue passing raw numeric arguments (growth, margin, years) – Rejected (fragmented and non-reusable).
- Adopt an external forecasting library (e.g., Prophet, Statsmodels) – Rejected (adds heavy dependencies and inconsistent outputs).
- Use plain dictionaries instead of dataclass – Rejected (less self-documenting and type-unsafe).

---

## Consequences
- Provides a single, auditable structure for all forward financial data.
- Enables both manual user input and auto-generated forecasts.
- Simplifies testing, caching, and agentic interactions (MCP/LLM).
- Serves as the integration bridge between fundamentals, valuation, and scenario tools.

---

## References
- McKinsey & Co., Valuation: Measuring and Managing the Value of Companies, 8th ed., 2020.
- Aswath Damodaran, Investment Valuation, 3rd ed., Wiley, 2012.
