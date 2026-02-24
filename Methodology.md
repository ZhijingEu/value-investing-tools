# Methodology & Assumptions

This document explains the theoretical foundations, key model choices, and default assumptions used by VIT. It complements the README (usage) with rationale and citations so results are auditable.

This document explains the default numeric assumptions and guardrails used by VIT. It is intentionally concise and focused on what is defensible via external sources vs. what is engineering guardrail logic for stability.

## Scope & Time Sensitivity
Some defaults (risk-free rate and equity risk premium) are market-dependent and should be refreshed periodically. The values below are anchored to public references as of early 2026.

## Market-Based Defaults (Cited)
These defaults are used in `VALUATION_DEFAULTS` and in MCP tool defaults. They can be overridden per call.

- Risk-free rate (`risk_free_rate`): **4.18%**
  - Rationale: 10-year U.S. Treasury yield at 4.18% (Damodaran reference).
  - Source: Damodaran update citing the 10-year Treasury yield at 4.18% (Jan 1, 2026).
  - See: https://www.edwardconard.com/macro-roundup/using-nominal-cash-flow-projections-the-irr-for-the-sp-500-was-8-41-on-jan-1-subtracting-the-10-year-treasury-rate-4-18-yields-an-erp-of-4-23-damodaran-notes-the-current-erp-is-alm/?filters=macro-roundup-database&view=detail

- Equity risk premium (`equity_risk_premium`): **4.23%**
  - Rationale: Implied ERP from Damodaran (Jan 1, 2026).
  - Source: Same Damodaran update as above.
  - See: https://www.edwardconard.com/macro-roundup/using-nominal-cash-flow-projections-the-irr-for-the-sp-500-was-8-41-on-jan-1-subtracting-the-10-year-treasury-rate-4-18-yields-an-erp-of-4-23-damodaran-notes-the-current-erp-is-alm/?filters=macro-roundup-database&view=detail

- Default corporate tax rate (`tax_rate_default` in WACC): **21%**
  - Rationale: U.S. federal corporate tax rate under TCJA.
  - Source: IRS guidance.
  - See: https://www.irs.gov/irm/part21/irm_21-007-004r

If you want to apply different markets or timeframes, override these defaults per call (see examples below).

## Engineering Guardrails (Heuristics)
These values are stability guardrails rather than universally accepted finance standards. They help avoid pathological outputs (division by zero, extreme growth assumptions, or unstable WACC).

Examples:
- Terminal growth cap: `g <= WACC - terminal_growth_gap` (default gap = 0.5%).
- Additional terminal growth cap: `g <= risk_free_rate + terminal_growth_rfr_spread` (default spread = 0.5%), applied when `terminal_growth_cap_mode="min_wacc_rfr"`.

Note: The risk-free anchored cap is a stability guardrail aligned with stable-growth guidance (terminal growth should not persistently exceed long-run nominal growth). Use overrides to match your market or research standard.
- Growth floor: `-5%`.
- Revenue CAGR haircut: `0.8` (conservative adjustment).
- Scenario multipliers for Low/Mid/High growth: `0.6 / 1.0 / 1.3`.
- Cost of debt clamps: min 2%, max 15%, fallback 4%.
- Peer comparability thresholds: coverage 60%, dispersion rules by metric.
- Fundamentals scoring thresholds and weights (see `vitlib/fundamentals.py`).

These are intentionally overrideable so you can align them to your own research standards.

## Override Mechanisms
VIT exposes override hooks at the library and MCP layers:

- **Valuation overrides** via `assumptions_overrides` (dict) on:
  - `dcf_three_scenarios`, `dcf_implied_enterprise_value`, `compare_to_market_ev`, `compare_to_market_cap`, `dcf_sensitivity_grid`
- **Scoring overrides** via `scoring_overrides` (dict) on:
  - `compute_fundamentals_scores`, `full_fundamentals_table`
- **Peer diagnostics overrides** via `diagnostics_overrides` (dict) on:
  - `peer_multiples`

## Overrides Quick Reference
Use these keys in the override dicts to tune guardrails without modifying core logic.

| Override Dict | Supported Keys (non-exhaustive) | Affects |
|---|---|---|
| `assumptions_overrides` | `terminal_growth_gap`, `terminal_growth_rfr_spread`, `terminal_growth_cap_mode`, `growth_floor`, `fcf_cagr_bounds`, `rev_cagr_bounds`, `revenue_cagr_haircut`, `wacc_spread_low`, `wacc_spread_high`, `scenario_growth_multipliers`, `cost_of_debt_fallback`, `cost_of_debt_min`, `cost_of_debt_max`, `tax_rate_default`, `tax_rate_cap`, `ev_fcf_multiple_warn_high`, `ev_fcf_multiple_warn_low`, `premium_band_small`, `premium_band_large`, `cov_moderate` | DCF growth bounds, WACC bounds, EV/FCF warnings, premium interpretation bands, volatility notes |
| `scoring_overrides` | `thresholds`, `weights`, `data_incomplete_threshold`, `recommendation_cutoffs` | Fundamentals scoring thresholds, factor weights, data completeness gate, recommendation logic |
| `diagnostics_overrides` | `coverage_warn_threshold`, `metric_specs` | Peer comparability diagnostics coverage/dispersion thresholds |

### Example
```python
assumptions_overrides = {
    "terminal_growth_gap": 0.0075,
    "growth_floor": -0.03,
    "revenue_cagr_haircut": 0.7,
    "premium_band_small": 7.5,
    "premium_band_large": 25.0,
}

df = dcf_implied_enterprise_value(
    "MSFT",
    assumptions_overrides=assumptions_overrides,
)
```

This preserves the existing methodology while making the guardrails explicit and auditable.
