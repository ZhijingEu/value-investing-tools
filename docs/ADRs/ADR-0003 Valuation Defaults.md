# ADR-0003: Valuation Defaults - ERP, Fallback Growth, and FCF Window

## Status
Accepted (Implemented in code, 2026-02-23)

## Context
Valuation calculations within the Value Investing Tools library - particularly  
`compare_to_market_ev()`, `compare_to_market_cap()`, and `dcf_three_scenarios()` -  
require several key parameters that materially affect output:

- **Equity Risk Premium (ERP)**  
- **Risk-Free Rate**  
- **Fallback Growth Rate (g)**  
- **Free Cash Flow (FCF) Averaging Window**

These values were previously scattered across code blocks or notebooks with no consistent convention,  
leading to differences between scenario runs and lower reproducibility.

This ADR standardises default inputs and explicitly documents their rationale.

---

## Decision
Establish a single set of **conservative, globally applied valuation defaults**.  
Defaults are fixed in code but can be overridden via function arguments.

| Parameter | Default | Description | Rationale |
|------------|----------|-------------|------------|
| **Risk-Free Rate** | 4.18 % | 10-year US Treasury yield (Damodaran Jan 1, 2026 reference). | Anchors discount rate baseline. |
| **Equity Risk Premium (ERP)** | 4.23 % | Implied excess return required by equity investors. | Based on Damodaran implied ERP (Jan 1, 2026). |
| **Fallback Growth Rate (g)** | 2.0 % | Used when terminal or forecast growth data are unavailable. | Approximates long-term real GDP growth. |
| **FCF Averaging Window** | 3 years | Period over which historical free cash flows are averaged. | Smooths volatility given yfinance's 4-year data limit. |

**Discount Rate (WACC)** is derived as  
`WACC = Risk-Free + Beta x ERP + Cost of Debt x (1 - TaxRate)`  
using these defaults when specific inputs are absent.

---

## Implementation
- Embed these defaults in all valuation-related functions as keyword arguments:  
  ```python
  def dcf_three_scenarios(...,
      risk_free_rate: float = 0.0418,
      equity_risk_premium: float = 0.0423,
      fallback_growth: float = 0.02,
      fcf_years: int = 3,
      ...
  )

- Expose the same parameters in ```compare_to_market_ev()``` and ```compare_to_market_cap()``` for consistency.
- Document defaults in function docstrings and README.
- Add a defaults_config() helper returning a dictionary of current defaults for transparency.
- Ensure that orchestrator tools and MCP endpoints echo defaults in JSON outputs.

---

## Alternatives Considered

- Dynamic ERP via API or market data.
- Rejected - adds dependencies and instability; project aims for reproducibility.

- No central defaults (manual each run).
- Rejected - increases user error and inconsistency.

- Aggressive (growth-tilted) defaults.
- Rejected - violates conservative valuation ethos.

---

## Consequences
- Establishes consistent, auditable valuation assumptions across all modules.
- Enables reproducible cross-company comparisons.
- Provides a simple override mechanism for advanced users.
- Serves as foundation for ADR-0004 (Profiles), which parameterises these defaults per scenario type.

---

## References
- Aswath Damodaran, implied ERP update (Jan 1, 2026).
- McKinsey & Co., Valuation: Measuring and Managing the Value of Companies, 8th ed., 2020.

