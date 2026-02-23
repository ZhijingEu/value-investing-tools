# Calibration and Backtest Plan

## Purpose
This plan defines how to calibrate scoring thresholds and weights in a reproducible way before promoting major model changes (for example ADR-0001 4x4 scoring).

## Scope
- Current model: `compute_fundamentals_scores` (12-metric pipeline).
- Future model: 4x4 framework (when implemented).
- Horizon: 1Y, 3Y, 5Y forward returns and drawdown behavior.

## Data and Hygiene Rules
- Point-in-time discipline: avoid look-ahead in fundamentals and prices.
- Universe control: define sector, market-cap, and liquidity filters up front.
- Survivorship bias control: include delisted names where possible.
- Timestamp all assumption sets with `analysis_report_date`.

## Core Experiments
1. Quartile sort backtest:
   - Rank by total score at rebalance date.
   - Compare top vs bottom quartile forward returns.
2. Factor contribution:
   - Test each factor score independently and in combination.
3. Threshold sensitivity:
   - Shift each absolute threshold band up/down and measure stability.
4. Sector robustness:
   - Evaluate performance dispersion by sector and regime.

## Evaluation Metrics
- Annualized return and volatility.
- Information ratio vs benchmark.
- Max drawdown and downside capture.
- Hit rate (top quartile outperforming bottom quartile).
- Turnover and capacity proxy (liquidity-aware).

## Acceptance Gates
- Results must beat random/shuffled baselines.
- No single metric should dominate all signal value.
- Performance should persist across at least two disjoint time windows.
- Documented failure cases are required (not only favorable runs).

## Deliverables
- `docs/calibration_report.md` summarizing outcomes and caveats.
- Versioned threshold/weight table with rationale.
- Test fixtures for any adopted threshold changes.

## References
- Fama & French (1993), common risk-factor framework.
- Piotroski (2000), accounting-signal based value strategy evidence.
- Damodaran valuation notes (stable growth and discount-rate discipline).
