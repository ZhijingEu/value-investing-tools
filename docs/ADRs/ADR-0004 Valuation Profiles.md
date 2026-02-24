# ADR-0004: Valuation Profiles – Conservative, Moderate, and Speculative

## Status
Proposed (Draft)

## Context
Valuation outcomes are highly sensitive to assumptions such as discount rate (WACC), growth rate (g), and reinvestment expectations.  
To make these assumptions transparent and reproducible, Value Investing Tools introduces **three valuation profiles** that scale key parameters in predictable ways.

These profiles apply **only to valuation modules**:
- `dcf_three_scenarios()`
- `compare_to_market_ev()`
- `compare_to_market_cap()`

Fundamentals scoring and peer comparison remain **profile-agnostic**.

---

## Decision
Adopt three predefined valuation profiles:

| Profile | Description | Intended Use |
|----------|--------------|---------------|
| **Conservative** | Pessimistic assumptions (higher discount, lower growth). | Defensive or cyclical sectors; high uncertainty. |
| **Moderate** | Balanced assumptions around baseline defaults. | Typical case; default if none specified. |
| **Speculative** | Optimistic assumptions (lower discount, higher growth). | High-growth or early-stage firms. |

### Parameter Scaling

Each profile adjusts the baseline defaults from ADR-0003:

| Parameter | Conservative | Moderate | Speculative |
|------------|---------------|-----------|--------------|
| **Risk-Free Rate** | +0.5 % | Base (4.5 %) | –0.5 % |
| **Equity Risk Premium (ERP)** | +1.0 % | Base (5.5 %) | –1.0 % |
| **Fallback Growth Rate (g)** | –0.5 % | Base (2.0 %) | +1.0 % |
| **FCF Averaging Window** | 3 yrs | 3 yrs | 2 yrs |
| **Terminal Growth Cap** | ≤ WACC – 0.75 % | ≤ WACC – 0.5 % | ≤ WACC – 0.25 % |

### Example Usage

```python
dcf_three_scenarios(
    ticker="MDLZ",
    profile="conservative"  # or "moderate" / "speculative"
)
```

The function adjusts parameters automatically according to the selected profile.

---

## Implementation
1. Profile Mapping

```
valuation_profiles = {
    "conservative": {"rf_adj": +0.005, "erp_adj": +0.01, "g_adj": -0.005, "fcf_years": 3, "terminal_gap": 0.0075},
    "moderate": {"rf_adj": 0.0, "erp_adj": 0.0, "g_adj": 0.0, "fcf_years": 3, "terminal_gap": 0.005},
    "speculative": {"rf_adj": -0.005, "erp_adj": -0.01, "g_adj": +0.01, "fcf_years": 2, "terminal_gap": 0.0025},
}
```
2. Function Argument – Each valuation function should accept profile="moderate" by default.
3. Parameter Derivation – Effective WACC and terminal growth are adjusted automatically based on profile values.
4. Audit Trail – Include profile metadata in the output JSON:
```
"valuation_profile": {
    "name": "conservative",
    "effective_rf": 0.050,
    "effective_erp": 0.065,
    "effective_g": 0.015
}
```
5. Documentation – Add profile definitions to README and notebook examples.

---

## Alternatives Considered
- Continuous sliders instead of discrete profiles – Rejected (adds complexity and reduces reproducibility).
- Apply profiles to fundamentals scoring – Rejected (scores must stay constant for comparability).
- Introduce more than three profiles – Rejected (unnecessary granularity).

---

## Consequences
- Simplifies scenario analysis and comparison.
- Ensures consistent valuation assumptions across modules.
- Keeps fundamentals scoring independent of valuation settings.
- Provides transparent metadata for audit and reproducibility.

---

## References
- McKinsey & Co., Valuation: Measuring and Managing the Value of Companies, 8th ed., 2020.
- Aswath Damodaran, Investment Valuation, 3rd ed., Wiley, 2012.
