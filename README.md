# ValueInvestingTools
Value Investing Tools VIT is a Python library and MCP server for Claude that supports value investing principles based fundamental equity analysis: fetch and summarize financial data, benchmark against peers, run DCF valuations with scenarios, estimate Enterprise Value and Equity Value, and save charts/CSVs for reproducible workflows.

Important Disclosure - This code was co-developed with review and refactoring support of ChatGPT-5 and Claude Sonnet-4

## Table of Contents
1. [Why this exists](#1-why-this-exists)
2. [How to use this library](#2-how-to-use-this-library)
3. [Installation](#3-installation)
4. [Data conventions & assumptions](#4-data-conventions--assumptions)
5. [Company Intrinsic Performance Factors](#5-company-intrinsic-performance-factors)
6. [Multi-company Comparative Fundamentals](#6-multi-company-comparative-fundamentals)
7. [Peer Multiples Comparisons](#7-peer-multiples-comparisons)
8. [Implied Value from Historical Performance (EV & Equity)](#8-Implied-Value-from-Historical-Performance-(EV-&-Equity))
9. [Scenario-Based Intrinsic Value (DCF with Terminal Growth))](#9-Scenario-Based-Intrinsic-Value-(DCF-with-Terminal-Growth))
10. [Data Health Reporting](#10-data-health-reporting)
11. [Summary of Functions](#11-summary-of-functions)
12. [Automated Valuation Orchestrator](#12-automated-valuation-orchestrator--estimate_company_value)
13. [Notes, Roadmap, Contributing](#13-notes-roadmap-contributing)
14. [License](#14-license)

# 1. Why this exists
ValueInvestingTools (VIT) is a Python library for fundamental equity analysis that supports workflows such as valuation scenarios, observed vs. implied EV, peer benchmarking, and exportable visuals/data. 

VIT fetches statements, builds clean data sets, and runs DCF and peer-multiple valuations with ready-to-plot outputs. 

This repo also ships an MCP server so Claude Desktop can call VIT functions as tools to perform calculations, generate charts & artifacts and saves outputs for reproducible workflows (Note that there is a separate readme within this repo for the MCP Server Refer to the **MCP Server Setup Guide** → [VIT-MCP_Server_SetUp_README.md](https://github.com/ZhijingEu/value-investing-tools/blob/main/VIT-MCP_Server_SetUp_README.md)

Most free data sources (e.g., Yahoo Finance) provide quick snapshots, but they:
- Mix TTM and point-in-time values inconsistently.
- Omit critical ratios (ROE, reinvestment rates, multi-year CAGRs).
- Hide the calculation methods, making reproducibility difficult.

ValueInvestingTools addresses these gaps by:
- Computing metrics directly from statements (income, balance sheet, cash flow).
- Returning everything as decimals for programmatic consistency (0.15 = 15%).
- Offering outputs as Pandas DataFrames (default) or JSON-serializable dicts (for LLMs/agents).
- Adding health/status notes so users see when values were missing, approximated, or based on short coverage.

# 2. How to use this library
This toolkit follows a stepwise workflow that mirrors how analysts typically think about a company:

1. Company snapshot → look at TTM vs historical averages.
2. Peer group comparisons → put one company's numbers in context.
3. Scoring fundamentals (0–5) → normalize metrics to directly compare across firms.
4. Visualize → explore results with clustered bars, boxplots, and time-series plots.
5. Drill down → inspect trends in a single metric for a single company.
6. Extend into valuation → move beyond raw fundamentals:
   - 6.1 Benchmark valuation multiples (PE, PS, EV/EBITDA) against peers.
   - 6.2 Run DCF scenarios (Low/Mid/High) using perpetuity growth methods.
   - 6.3 Estimate implied Enterprise Value from historical growth + FCF.
   - 6.4 Translate EV → Equity → per-share value to see if the market looks rich or cheap.

This flow reflects how the library is organized: start with intrinsic factors, add comparative context, then use valuation models to connect fundamentals to price.

# 3. Installation & Quick Start
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate # Windows
pip install -r requirements.txt
```

```python
import ValueInvestingTools as vit

# 1) Single-company snapshot: TTM vs historical averages
summary = vit.fundamentals_ttm_vs_average("MSFT")
print(summary)

# 2) Multi-company fundamentals (absolute values)
tickers = ["MSFT", "AAPL", "GOOG"]
actuals = vit.compute_fundamentals_actuals(tickers, basis="annual")

# 3) Scored fundamentals (0–5 per metric, with factor rollups and total)
scores = vit.compute_fundamentals_scores(actuals, basis="annual")

# 4) Visualize: clustered scores across tickers
vit.plot_scores_clustered(scores, basis="annual", include_total=True)

# 5) Drill into a single metric over time for one company
vit.plot_single_metric_ts("MSFT", "ROE", family="Profitability", basis="annual")
```

# 4. Data conventions & assumptions

## Conventions

- **Units**: Ratios, margins, and growth rates are decimals (0.18 = 18%). Format as % only for display.
- **Basis switch & suffixing**:
  - `basis="annual"` → multi-year averages, public labels end in `-Ave`.
  - `basis="ttm"` → trailing twelve months, labels end in `-TTM`.
- **Special cases**:
  - CAGRs (revenue, earnings) are always annual, since multi-year growth cannot be defined on a TTM basis.
  - PEG and Beta are always snapshot (TTM), regardless of basis flag.
- **Scoring**: 0–5 where 5 = favorable/strong/safer.
- **Outputs**: DataFrames by default; use `as_df=False` for JSON-friendly dicts.
- **CSV persistence**: Opt-in with `save_csv=True` (default is no files).

## Assumptions & Data Sources

- **Source**: Yahoo Finance (via yfinance Python library).
- **Range**: 3–5 most recent fiscal years plus TTM where relevant.
- **Statements**: Income statement, balance sheet, cash flow.
- **Estimates**: PEG ratio uses forward EPS growth from Refinitiv (via Yahoo).
- **Peer-normalization**: Not applied in scoring – all cutoffs are absolute, not relative.
- **Growth Rate Methodology** assumes FCF-First Approach where growth assumptions prioritize Free Cash Flow CAGR over revenue CAGR when calculating terminal values to reflect that shareholder value derives from cash generation, not revenue expansion. Revenue growth that requires proportional increases in capital expenditure or working capital provides less value than efficient cash conversion. (More info within Section 9)

## Interpretation caveats

These tools are not a substitute for deep qualitative analysis or due diligence. They:
- Emphasize long-term fundamentals, not near-term catalysts or sentiment.
- Do not account for qualitative drivers like management quality, regulation, or competitive dynamics.
- Treat average historical FCF and growth as proxies for perpetuity assumptions, which may not hold.
- May understate reinvestment for intangible-heavy sectors (e.g., SaaS, pharma).

Outputs should be treated as a consistent analytical baseline, not an investment recommendation.

# 5. Company Intrinsic Performance Factors

## Theory
Every company has a set of intrinsic drivers that shape its long-term value, regardless of short-term market noise. 

While there are other predictive scoring systems like F-Score or Z-Score, VIT provides a heuristic/diagnostic framework that helps to structure the comparison between the financial performance of companies along the lens of four key drivers with transparent thresholds.

These drivers fall into four broad categories:
- **Profitability** – efficiency in generating returns on capital and margins.
- **Growth** – ability to scale revenues and earnings over time.
- **Reinvestment** – discipline and capacity to reinvest earnings to compound value.
- **Risk** – balance sheet strength and exposure to volatility.

By scoring these factors consistently across companies and time, this creates a deterministic baseline of financial quality that can be compared across industries and cycles.

Unlike relative rankings (z-scores), this framework uses absolute thresholds so that:
- A score of 5 always means "excellent" in real financial terms, not just "better than peers."
- Scores are stable across time and do not drift if industry peers get weaker or stronger.
- A single company can be scored in isolation without requiring a peer set.

## Factor Components & Weights

| **Factor** | **Sub-metrics** | **Weighting** |
|------------|-----------------|---------------|
| Profitability | ROE (30%), Net Margin (25%), Operating Margin (25%), ROA (20%) | 100% |
| Growth | Revenue CAGR (40%), Earnings CAGR (40%), PEG Ratio (20%) | 100% |
| Reinvestment | Reinvestment Rate (60%), Capex / Revenue Ratio (40%) | 100% |
| Risk | Debt/Equity (40%), Beta (40%), Current Ratio (20%) | 100% |

## Weight Rationales

- **ROE (30%)**: Efficient use of shareholder capital; empirically linked to long-term returns.
- **Net & Operating Margins (25% each)**: Capture efficiency and pricing power; especially relevant in low-margin sectors.
- **ROA (20%)**: Broad measure of asset utilization; critical for asset-heavy industries.
- **Revenue & Earnings CAGR (40% each)**: Strong correlation with shareholder value creation; signal scalability and discipline.
- **PEG (20%)**: Provides a reality check by relating valuation to growth; forward-looking but noisy.
- **Reinvestment Rate (60%)**: Direct measure of compounding potential (earnings retained for growth).
- **Capex Ratio (40%)**: Captures capital intensity; high values can indicate growth investment or inefficiency.
- **Debt/Equity (40%)**: Core solvency ratio; high leverage reduces flexibility.
- **Beta (40%)**: Systematic risk proxy; widely used in cost of equity (CAPM).
- **Current Ratio (20%)**: Liquidity test; lower weight due to sector variability.

## Scoring Thresholds & Mechanics

| Metric | Direction | t1 | t2 | t3 | t4 |
|--------|-----------|----|----|----|----|
| ROE | ↑ | 0.05 | 0.10 | 0.15 | 0.20 |
| Net Margin | ↑ | 0.05 | 0.10 | 0.15 | 0.20 |
| Operating Margin | ↑ | 0.05 | 0.10 | 0.15 | 0.20 |
| ROA | ↑ | 0.03 | 0.06 | 0.10 | 0.15 |
| Revenue CAGR | ↑ | 0.00 | 0.05 | 0.10 | 0.15 |
| Earnings CAGR | ↑ | 0.00 | 0.05 | 0.10 | 0.15 |
| PEG Ratio | ↓ | 3.00 | 2.00 | 1.50 | 1.00 |
| Reinvestment Rate | ↑ | 0.10 | 0.25 | 0.50 | 0.75 |
| Capex Ratio | ↓ | 0.10 | 0.05 | 0.02 | 0.00 |
| Debt/Equity | ↓ | 2.00 | 1.50 | 1.00 | 0.50 |
| Beta | ↓ | 2.00 | 1.50 | 1.20 | 0.80 |
| Current Ratio | ↑ | 0.80 | 1.00 | 1.50 | 2.00 |

## Interpretation
- Crossing a threshold increases the score by 1.
- Example (higher-is-better): ROE of 12% → score of 3.
- Example (lower-is-better): Beta of 1.3 → score of 3.

### Recommendation Logic
The total score and factor breakdown can be mapped into diagnostic categories:
- **Elite Performer (≥18, all factor scores ≥4)** – broad-based strength across profitability, growth, reinvestment, and risk.
- **Resilient Core (14–16.9)** – strong fundamentals with minor weaknesses.
- **Uneven Fundamentals (11–13.9)** – patchy performance; some strengths offset by critical weaknesses.
- **Weak Fundamentals (<11)** – structural fragility.

### Guardrails
- Any factor score = 1 → flagged as a critical weakness.
- Risk score <2.5 → flagged regardless of total score.
- Reinvestment score <2.0 → flagged as unsustainable.
- Excessive imbalance between factors (high dispersion) → flagged.
- Data completeness <75% → flagged.

## Options & defaults
**Function**: `compute_fundamentals_actuals(tickers, basis="annual", save_csv=False, as_df=True)`

**basis**:
- `"annual"` for multi-year averages (default).
- `"ttm"` for trailing 12 months (point-in-time).

**Important caveats**:
- CAGRs (revenue, earnings) are computed only on an annual basis.
- PEG and Beta are always snapshot/TTM measures, regardless of basis flag.
- `save_csv`: False by default; if True, saves raw statements per ticker.
- `as_df`: True returns a DataFrame; False returns JSON-serializable dicts.

## Compute
```python
import ValueInvestingTools as vit

# Annual averages for one company
df = vit.compute_fundamentals_actuals(["MSFT"], basis="annual")
print(df.T)

# Scores (0–5) and factor rollups
scores = vit.compute_fundamentals_scores(df, basis="annual", merge_with_actuals=False)
print(scores.filter(regex="^score_|_score$").T)

# Average vs TTM comparison table
summary = vit.fundamentals_ttm_vs_average("MSFT")
print(summary)
```

## Visualize
```python
# Profitability time series
vit.plot_single_metric_ts("MSFT", "ROE", family="Profitability", basis="annual")

# Family of metrics (e.g., Reinvestment) across tickers
vit.plot_multi_tickers_multi_metrics_ts(
    ["MSFT","AAPL","GOOG"],
    family="Reinvestment-basic",
    metrics=["ReinvestmentRate","CapexRatio"],
    basis="annual"
)

# Clustered scores (factor rollups across companies)
scores = vit.compute_fundamentals_scores(["MSFT","AAPL","GOOG"], basis="annual")
vit.plot_scores_clustered(
    scores,
    metrics=["profitability_score","growth_score","reinvestment_score","risk_score"],
    include_total=True, sort_by="avg"
)
```

## Interpretation Guide
The outputs are scored tables and visuals. Here's how to read them:

**Metric-level scores (0–5)**:
- Example: ROE = 12% → Score = 3 (solid but not elite).
- Example: Beta = 1.3 → Score = 3 (moderate volatility).

**Factor rollups**:
- Profitability score = average of ROE, margins, ROA.
- Growth score = average of revenue CAGR, earnings CAGR, PEG.
- Reinvestment score = reinvestment rate + capex ratio.
- Risk score = debt/equity, beta, current ratio.

**Total score (out of 20)**:
- MSFT: Profitability 4.0 + Growth 3.0 + Reinvestment 3.5 + Risk 4.0 = 14.5 → "Resilient Core."

**Visuals**:
- Clustered bar charts: highlight relative strengths within one company across factors.
- Time series: show whether fundamentals are improving or deteriorating.

**Key point**: High total score is positive, but interpretation depends on context.
- A company may score high on growth but low on risk → more volatile.
- Another may score high on profitability but low on reinvestment → mature, less compounding ahead.

While this example uses one company (MSFT), the same framework extends seamlessly to peer-group comparisons – making it easy to spot whether one company's strengths and weaknesses stand out against competitors.

# 6. Multi-company Comparative Fundamentals

## Theory
Looking at one company in isolation provides insights into its intrinsic performance, but valuation and investment decisions are inherently relative. Competitors in the same sector, geography, or market cap range face similar macro conditions, so peer comparison highlights whether a company's fundamentals are strong, average, or lagging.

Key principles:
- Keep basis consistent (annual vs TTM) across all companies in a peer set.
- Choose peers deliberately: industry, size, product mix, and region all matter.
- Focus on directional gaps (is one company consistently stronger on margins, leverage, growth?).
- Scores and rollups make it easier to compare across dimensions, but raw metrics are equally important for nuance.

## Options & defaults
- **Function**: `compute_fundamentals_actuals(tickers, basis="annual", save_csv=False, as_df=True)`
- **tickers**: list of tickers (minimum 2 recommended).
- **basis**: `"annual"` (multi-year averages) or `"ttm"` (point-in-time).
- **save_csv**: save per-ticker statements if True.
- **as_df**: True returns DataFrame; False returns JSON-serializable dicts.

## Compute
```python
import ValueInvestingTools as vit

tickers = ["MSFT", "AAPL", "GOOG"]

# Absolute fundamentals across companies
actuals = vit.compute_fundamentals_actuals(tickers, basis="annual")

# Score fundamentals for the peer set
scores = vit.compute_fundamentals_scores(actuals, basis="annual")

print(actuals.head())
print(scores.filter(regex="score").head())
```

## Visualize
```python
# Compare reinvestment metrics across companies
vit.plot_multi_tickers_multi_metrics_ts(
    ["MSFT","AAPL","GOOG"],
    family="Reinvestment-basic",
    metrics=["ReinvestmentRate","CapexRatio"],
    basis="annual"
)

# Compare score rollups across companies
vit.plot_scores_clustered(
    scores,
    metrics=["profitability_score","growth_score","reinvestment_score","risk_score"],
    include_total=True, sort_by="avg"
)
```

## Interpretation Guide
When comparing multiple companies:
- **Absolute metrics** (e.g., ROE, margins, debt ratios) reveal who performs best in raw terms.
- **Scored metrics** standardize the values on a 0–5 scale, making it easy to see outliers at a glance.
- **Rollups** allow for quick high-level assessment (e.g., Company A stronger on profitability, Company B weaker on risk).
- **Visualizations**:
  - Multi-ticker line plots show trend divergence over time.
  - Clustered bar plots show relative positioning across factors.

**Note**: Scoring is absolute, not relative – a high score for one company does not lower the score for another. This ensures stability over time but means peers may all score high (or low) if the industry is generally strong (or weak).

# 7. Peer Multiples Comparisons

## 7.1 Peer Multiples Calculation

### Theory
Relative valuation through multiples is one of the most widely used tools in equity analysis. Instead of projecting cash flows or building detailed forecasts, multiples provide a market-based benchmark: how do similar companies trade relative to their earnings, sales, or operating cash flows?

This library summarizes peer valuation bands (25th, 50th, and 75th percentiles) so you can anchor a company's valuation against a conservative, median, and bullish context. Note that the PE, PS, EV/EBITDA here are sourced as TTM ratios (latest trailing-twelve-months), not multi-year averages (unlike the earlier function for fundamental factors analysis).

But the method is only as good as the peer group chosen.

Selecting the right peer set is critical to getting a meaningful benchmark. Poorly chosen peers can distort the analysis and lead to misleading conclusions.

When selecting peers, consider:
- Industry & business model – companies should operate in the same or closely related industries.
- Company size & market share – large-cap vs small-cap firms often trade at different multiples.
- Geographic exposure – local vs global footprint changes growth, margin, and risk profiles.
- Financial metrics – capital intensity, margin structure, reinvestment needs.
- Customer base & product portfolio – degree of overlap in markets served.
- Competitive positioning – companies with durable moats vs those with commoditized products.

While peer selection is outside the scope of this library, the analyst must exercise judgment to ensure apples-to-apples comparison.

### Options & defaults
**Function**: `peer_multiples(tickers: list[str], *, target_ticker: str, include_target: bool = False, as_df: bool = True) -> dict[str, Any]`

- **target_ticker**: ticker of the company being benchmarked.
- **tickers**: list of peer tickers.

You can opt-in to `include_target=True` if you want the target to be counted in the peer stats (not generally recommended for valuation comparability).

### Outputs

- **peer_comp_detail**: per-ticker snapshot for all tickers passed in (including target_ticker if present). Contains raw ratios such as pe_ratio, ps_ratio, ev_to_ebitda, plus any per-share valuation fields your build provides.
- **peer_multiple_bands_wide**: rows = PE / PS / EV_EBITDA, cols = Min / P25 / Median / P75 / Max / Average. Computed from peers only when include_target=False; includes the target if include_target=True.
- **peer_comp_bands**: long per-share valuation bands (e.g., PE_Min, PE_P25, etc.), computed from peers only when include_target=False.

### Compute
```python
import ValueInvestingTools as vit

tickers = ["AAPL","GOOGL","AMZN","MSFT"]
# Target required; default behavior excludes target from stats
res = vit.peer_multiples(tickers, target_ticker="MSFT", include_target=False)
```

### Visualize
```python
# Access detail (all tickers) and peer-only bands
detail = res["peer_comp_detail"]               # includes MSFT row
bands  = res["peer_multiple_bands_wide"]       # quartiles over peers (MSFT excluded)

# Boxplot with overlay of target, using detail + bands
fig, ax = vit.plot_peer_metric_boxplot(
    peer_comp_detail=detail,
    peer_multiple_bands_wide=bands,
    metric="EV_EBITDA",
    target_ticker="MSFT",
    include_target_in_stats=False,  # match the call above
)
```
### Notes
- Default exclusion of the target from peer stats is a defensible comp approach (prevents the subject from biasing its benchmark).
- You may include the target in peer stats for presentations or small peer sets by setting include_target=True.
- plot_peer_metric_boxplot has been updated to overlay the target value alongside the peer distribution, even if the target was excluded from the stats.

### Interpretation Guide
**Lo / Med / Hi bands**:
- P25 (Lo) → conservative benchmark.
- P50 (Med) → fair/median estimate.
- P75 (Hi) → optimistic benchmark.

**Relative positioning**:
- If a target consistently sits below P25, it may be undervalued or fundamentally weaker than peers.
- If consistently above P75, it may be richly valued or justifiably premium (due to moat, growth, margins).

**Context matters**:
- High-growth firms may command higher multiples.
- Mature firms may trade at discounts despite solid fundamentals.

**Key point**: Multiples are diagnostic, not definitive. They work best as:
- A sanity check against DCF results.
- A way to see whether the market values a company consistently with peers.
- A quick screen to identify outliers that warrant deeper analysis.

## 7.2 Extract Price Snapshots & Averages

Many workflows want simple average price anchors for the last 1/30/90/180 days.

### Function
`historical_average_share_prices(tickers, *, analysis_report_date=None, save_csv=False, as_df=True)`

### What it does
Returns average prices over 1d, 30d, 90d, 180d windows (ending today) for one or more tickers. Internally it leverages the library's price snapshot helper, so you get consistent behavior anywhere snapshots are used. Columns: Ticker, avg_price_1d, avg_price_30d, avg_price_90d, avg_price_180d, price_asof, Notes

```python
import ValueInvestingTools as vit

vit.historical_average_share_prices(["MSFT","AAPL"])
```

Note the CSV export is opt-in via `save_csv=True`; files saved under `./output`

## 7.3 Conversion of peer multiples into per-share prices

You can now convert peer multiple bands straight into implied per-share prices for your target.

### Function
`price_from_peer_multiples(peer_multiples_output, *, ticker=None, analysis_report_date=None, save_csv=False, as_df=True)`

### What it does
- Takes the output of `peer_multiples(...)` and returns P25 / P50 / P75 per-share price bands for PE, PS, and EV/EBITDA. It infers the target ticker when possible (or you can pass ticker= explicitly)
- For EV/EBITDA, Enterprise Value is converted to Equity Value per share using the latest shares, net debt, cash & equivalents, and minority interests carried in the peer snapshot, then mapped to P25/Median/P75. The inputs used are echoed back in Inputs_Used for auditability

```python
import ValueInvestingTools as vit

peers = vit.peer_multiples(["AAPL","GOOGL","MSFT","AMZN"], target_ticker="NVDA")
vit.price_from_peer_multiples(peers)
```

If your peer_multiples result has the target excluded from the peer stats (recommended), bands are still computed correctly from peers only while overlaying the target where needed.

# 8. Implied Value from Historical Performance (EV & Equity)

## Theory
Whereas the scenario-based DCF (Section 9) projects forward with explicit assumptions, the Implied EV module looks backward: it uses historical averages of free cash flow and growth as if they were steady-state inputs into a perpetuity growth model.

Therefore it is important to note that this function **does not introduce a separate terminal growth input**; instead it uses an averaged FCF base and historical CAGR (with guardrails) and discounts at WACC.

The question it answers is: "If we assume the company's average historical FCF and growth continue indefinitely, what enterprise value (EV) does that imply – and how does it compare to today's market EV?"

The calculation is based on the Perpetuity Growth Method:

$$EV_{\text{implied}} = \frac{FCF_{\text{avg}} \cdot (1+g)}{WACC - g}$$

Where:
- $FCF_{\text{avg}}$ = average historical free cash flow (last 3–5 years)
- $g$ = average historical growth rate
- $WACC = R_f + \beta \cdot ERP$ (using yfinance beta, required else raises error)
- $g$ is capped at $WACC - 0.5\%$ to avoid unrealistic perpetuity assumptions

**Backward-Looking Limitations**: This approach assumes historical FCF and growth patterns represent sustainable steady-state conditions. This assumption is particularly problematic for:
- Cyclical businesses (captures arbitrary cycle positions)
- Companies undergoing strategic transitions
- Industries facing disruption or regulatory change
- Firms with significant one-time items in historical FCF

## Options & defaults
**Balanced (default)** — conservative-leaning, history-anchored:
- Risk-free rate: **4.5%**
- Equity risk premium: **6.0%** 
- Fallback FCF growth: **2.0%**, with guardrail **g ≤ WACC − 0.5%**
- FCF base window: **3 years** average by default
- DCF horizon: **3 years**, then terminal (DCF only)

- **More Conservative (quick stress test, optional)** — use when rates/risks feel elevated:
- Keep the above, and optionally set `years=1/2/3`, or use `cf_base="recent_weighted"` / `recency_weight=0.6` (see function args) to emphasize latest years.

## Market Comparison

The function `compare_to_market_ev()` compares Implied EV vs Observed EV (from Yahoo Finance):
- If Observed EV > Implied EV, the market is pricing in more growth/lower risk.
- If Observed EV < Implied EV, the market is pricing in less growth/more risk (or undervaluing the company).

Output includes:
- Observed_EV
- EV_Implied
- Premium_% (positive = richer market valuation)
- Notes (health/status messages).

## Compute
```python
import ValueInvestingTools as vit

# Implied EV from historical FCF + growth
implied_ev = vit.dcf_implied_enterprise_value("AAPL", window_years=5)

# Compare to market EV
cmp = vit.compare_to_market_ev("AAPL", implied_ev)
print(cmp)

# Visualize
fig, ax = vit.plot_ev_observed_vs_implied(cmp)
```

## Equity Bridge (EV → Equity → Per-share)

Once an implied EV is computed, it can be reconciled to equity value using standard adjustments:

$$\text{Equity} = EV - \text{Net Debt} + \text{Cash \& Equivalents} - \text{Minority Interest}$$

Then:

$$\text{Per-Share Implied} = \frac{\text{Equity}}{\text{Shares Outstanding}}$$

**Note**: If Yahoo Finance has partial data - the function falls back to `sharesOutstanding × currentPrice` when `marketCap` is missing; otherwise leaves the field `None` and logs context in `Notes`.

Implemented via `compare_to_market_cap`

`compare_to_market_cap(ticker_or_evdf, *, years=None, risk_free_rate=0.045, equity_risk_premium=0.060, growth=None, target_cagr_fallback=0.02, use_average_fcf_years=None, volatility_threshold=0.5, as_df=True, analysis_report_date=None)`

**Purpose**: Compares Implied Equity Value (derived from implied EV) with Observed Market Capitalization from Yahoo Finance.

**Inputs**: 
- Pass either a **ticker string** (mirrors `compare_to_market_ev` options), **or** the **single-row DataFrame** returned by `compare_to_market_ev`.
- Optional knobs match your EV comparison workflow (WACC, growth, averaging window, etc.).

**Output**: Single-row `pd.DataFrame` with the columns listed above.

## Compute
```python
df_cap = compare_to_market_cap("AAPL", years=5, use_average_fcf_years=5)
print(df_cap[["Ticker","Observed_MarketCap","Equity_Implied","Premium_%"]])

# Example B – from an existing EV comparison:
df_ev   = compare_to_market_ev("AAPL", years=5)
df_cap2 = compare_to_market_cap(df_ev)

# Visualize
fig, ax = plot_market_cap_observed_vs_implied_equity_val(df_cap, save_path="aapl_equity_vs_mktcap.png")
```

## Interpretation Guide
**Primary Use Case**: Best used as a **sanity check** alongside forward-looking DCF and peer multiples, not as a standalone valuation method.

**Market Comparison Context**:
- Premium >20%: Market expects significantly higher growth/lower risk than historical patterns
- Premium 5-20%: Market modestly optimistic vs. historical performance  
- Premium -5% to +5%: Market pricing roughly consistent with historical patterns
- Discount -20% to -5%: Market modestly pessimistic vs. historical performance
- Discount <-20%: Market expects significantly lower growth/higher risk, or potential undervaluation

**Key Limitations to Remember**:
- **Historical FCF Baseline**: Simple averaging can be misleading for volatile businesses; normalization helps but cannot eliminate all cyclical effects
- **Growth Rate Challenges**: Historical FCF growth is often unstable; revenue growth used as proxy with 20% haircut
- **WACC Sensitivity**: Same sensitivity issues as forward DCF; debt cost estimates may be imprecise
- **Business Evolution**: Historical patterns may not reflect current competitive position or strategic direction

**Quality Indicators**:
- Coefficient of Variation <0.3: Relatively stable historical FCF
- CoV 0.3-0.5: Moderate volatility (flagged)
- CoV >0.5: High volatility (flagged with recommendation to shorten window)

**When Results Are Most Reliable**:
- Mature, stable businesses with predictable cash flows
- Companies with consistent capital allocation patterns
- Industries with stable competitive dynamics

**When to Use Alternative Methods**:
- High-growth companies (historical growth not representative)
- Cyclical businesses (timing of cycle affects results significantly)
- Companies with major recent strategic changes

# 9. Scenario-Based Intrinsic Value (DCF with Terminal Growth)

## Theory
Discounted Cash Flow (DCF) estimates enterprise value by discounting a finite stream of cash flows plus a terminal value:

**Terminal value (Perpetuity Growth Method)**
$TV = \frac{FCF_{N} \cdot (1+g)}{WACC - g}$

**DCF-implied EV**
$EV_{\text{implied}} = \sum_{t=1}^{N} \frac{FCF_t}{(1+WACC)^t} \;+\; \frac{TV}{(1+WACC)^N}$

This library uses a three-scenario approach (Low / Mid / High) to bracket uncertainty without false precision.

## Key Methodology

This runs **DCF scenarios (Low/Mid/High)** with an explicit **terminal growth** model. Defaults are slightly tightened vs earlier versions:
- **Equity Risk Premium** 6.0%
- **Fallback FCF growth** 2.0% 
- **FCF window** 3 years by default.
- The terminal‐growth guardrail `g ≤ WACC − 0.5%`.

**WACC calculation** that includes both equity and debt costs:
$WACC = \frac{E}{V} \times R_e + \frac{D}{V} \times R_d \times (1-T)$

Where E/V is equity weight, D/V is debt weight, Re is cost of equity, Rd is cost of debt, and T is tax rate. For companies with no debt, this reduces to the cost of equity (Rf + β·ERP).

Uses market value of equity, book value of debt, estimated cost of debt from interest expense, and effective tax rate. Falls back to cost of equity (Rf + β·ERP) for companies with minimal debt.

**Growth Rate Prioritization**: 
1. FCF CAGR (if available and within -30% to +30% bounds)
2. Revenue CAGR with conservative 0.8x adjustment 
3. Fallback rate (default 3%)

**Scenario Risk-Growth Alignment**: 
- Low growth → Low WACC (lower risk)
- Mid growth → Mid WACC 
- High growth → High WACC (higher risk)

**FCF Baseline**: Uses normalized historical FCF with outlier removal and recent-period weighting, rather than simple averaging.

**Growth Constraints**: Growth rates capped at WACC - 0.5% and floored at -5% for stability.

Returns DataFrame with columns: Scenario, Growth_Used, WACC_Used, Per_Share_Value for 3 scenarios

**Impact on Results**: The WACC correction typically lowers the discount rate for leveraged companies, resulting in higher enterprise values compared to the previous cost-of-equity-only approach. Highly leveraged companies will see the most significant valuation changes.

### Methodological Advantages Over Standard Approaches Using Revenue CAGR

**FCF Growth Prioritization**: Unlike most valuation libraries that default to revenue CAGR, this library prioritizes FCF CAGR for terminal growth assumptions. This distinction is critical because:

**Revenue vs. FCF Growth Divergence**:
- Revenue growth doesn't guarantee shareholder value creation
- Companies can grow revenue while destroying value through margin compression, excessive capital requirements, or poor working capital management
- FCF growth directly reflects cash available to shareholders after all operating needs and reinvestment

**Real-World Impact Examples**:
- **Capital-intensive businesses**: Revenue growth requiring proportional capex increases shows lower FCF growth, leading to more conservative (realistic) valuations
- **Mature optimizers**: Companies improving working capital efficiency may show FCF growth exceeding revenue growth
- **High-growth transitions**: Early-stage companies with negative FCF despite revenue growth receive appropriate valuation constraints

**Validation Framework**: FCF CAGR is bounded (-30% to +30%) because it can be highly volatile due to lumpy capital expenditures and cyclical patterns. When FCF growth is extreme or unreliable, the model falls back to revenue CAGR with a conservative 0.8x adjustment, then to the specified fallback rate.

**Competitive Differentiation**: Most libraries use revenue CAGR because it's more stable and predictable. This library accepts the complexity of FCF-based growth to produce valuations more aligned with long-term shareholder value creation.

**When This Matters Most**:
- Manufacturing, utilities, and other capital-intensive industries
- Companies undergoing business model transitions
- Mature companies with changing reinvestment needs
- Any situation where top-line growth doesn't translate to bottom-line cash generation

## Options & defaults (quick reference)
- `years=5` horizon (terminal at year 5).
- `risk_free_rate=0.045, equity_risk_premium=0.06`.
- `target_cagr_fallback=0.02` if peer/target growth is unavailable.
- `peer_tickers=[...]` optionally seeds growth from peer FCF CAGRs (P25/P50/P75).
- Growth always capped at WACC − 0.5%.
- Requires beta; missing beta → error.

**Advanced Controls for Transformational Companies**
- `fcf_window_years=None` → Use all available FCF data (default). Set to 1-3 to limit historical window for companies with recent business model changes.
- `manual_baseline_fcf=None` → Calculate from historical data (default). Override with current run-rate FCF for rapid growth companies.
- `manual_growth_rates=None` → Use peer/target growth logic (default). Override with [low, mid, high] custom growth rates for AI/biotech/other high-growth sectors.

## Compute
```python
import ValueInvestingTools as vit

df = vit.dcf_three_scenarios(
    ticker="META",
    peer_tickers=["AAPL","GOOG","MSFT"],  # optional but recommended for better growth seeding
    years=5,
    risk_free_rate=0.045,
    equity_risk_premium=0.060,
    target_cagr_fallback=0.02
)
print(df)

# Advanced controls for transformational companies (e.g., NVDA, high-growth AI)
df_advanced = vit.dcf_three_scenarios(
    ticker="NVDA",
    fcf_window_years=2,                      # Use only recent 2 years
    manual_baseline_fcf=72_000_000_000,      # Use current $72B run-rate
    manual_growth_rates=[0.20, 0.30, 0.40], # AI-appropriate growth rates
    risk_free_rate=0.04,                     # Current lower rates
    equity_risk_premium=0.045
)
print(df_advanced)
```

## Interpretation guide
**WACC Sensitivity**: DCF values are highly sensitive to WACC assumptions. A 1% change in WACC can alter enterprise values by 15-25%. The debt cost estimation uses approximations that may not reflect true borrowing costs.

**FCF Quality Matters**: 
- Companies with volatile or cyclical FCF patterns receive volatility warnings
- Historical FCF may not represent sustainable run-rate if business mix has changed
- One-time items in historical FCF can distort baseline assumptions

**Growth Rate Limitations**:
- FCF growth can diverge significantly from revenue growth due to margin changes and capital intensity
- Historical growth may not predict future performance, especially for mature or disrupted industries
- Peer-based growth seeding assumes similar business models and competitive positions

**Scenario Interpretation**:
- Wide spreads between scenarios indicate high assumption sensitivity
- Negative per-share values suggest either data quality issues or fundamental business challenges
- Values significantly above/below market price warrant investigation of underlying assumptions

**When to Exercise Caution**:
- EV/FCF multiples >50x or <5x (flagged automatically)
- Companies undergoing major transitions or with lumpy capital cycles
- Highly cyclical businesses where historical averages may not represent normalized performance

### Advanced Controls Usage Guide

**When to Use fcf_window_years**:
- Companies with major business model shifts (limit to post-transformation years)
- Cyclical businesses at different cycle points (use peak-to-peak or trough-to-trough)
- Post-acquisition integration (exclude pre-deal periods)
- Example: `fcf_window_years=1` for NVDA to capture only post-AI boom performance

**When to Use manual_baseline_fcf**:
- Rapid growth companies where historical averages understate current capability
- Companies with recent one-time FCF impacts that distort averaging
- Seasonal businesses using annualized current quarter performance
- Example: `manual_baseline_fcf=72_000_000_000` for NVDA's current $72B run-rate

**When to Use manual_growth_rates**:
- AI/biotech companies requiring non-traditional growth assumptions
- Turnaround situations with expected inflection points
- Industries with known expansion cycles (e.g., semiconductor upcycles)
- Example: `manual_growth_rates=[0.15, 0.25, 0.35]` for AI infrastructure companies

**Advanced Controls Best Practices**:
- Use sparingly and document assumptions clearly
- Validate against peer multiples and market expectations
- Consider shorter forecast horizons (years=3) for high-uncertainty companies
- Always review resulting EV/FCF multiples for reasonableness (flagged automatically if >50x or <5x)

## 9.1 Extract Historical growth metrics (Revenue/Earnings/FCF) Helper Function

This helper computes CAGRs from the annual statements, with safe fallbacks and clear labeling of the observed period.

### Function
`historical_growth_metrics(tickers, *, min_years=3, analysis_report_date=None, save_csv=False, as_df=True)`

### What it Does
- Computes Revenue CAGR, Net Income (Earnings) CAGR, and Free Cash Flow (FCF) CAGR from the most recent clean annual points.
- Uses classic CAGR: (last/first)^(1/n)−1
- Where n is the year span between first and last observation. Returns Period_Start_Year and Period_End_Year so you can see the window used. FCF CAGR is only computed when both endpoints are positive (avoids undefined growth through/around zero).

### Usage
```python
import ValueInvestingTools as vit
vit.historical_growth_metrics(["MSFT","GOOGL","META"])
```
- Works with a single ticker or a list. min_years guards against too-short windows.
- The function reads annual income & cash-flow statements and builds the FCF series directly from Yahoo's "Free Cash Flow" row when present; if the first/last values are not both positive you'll see FCF_CAGR = None by design.

# 10. Data Health Reporting

## Theory
Any quantitative toolkit is only as reliable as its input data. Financial statements from Yahoo Finance (via yfinance) often have:
- Missing or incomplete items (e.g., dividends, minority interest, beta).
- Short coverage windows (e.g., only 3 years of data available).
- Noisy or inconsistent fields across tickers.

To make results auditable, most functions in this library return a health/status payload. These payloads flag when values were approximated, missing, or truncated, so analysts (or LLM agents) know when outputs should be interpreted with caution.

Therefore most functions in this library emit **data quality / completeness notes** inline with their outputs. This ensures that missing or incomplete inputs are always visible and traceable.

## Inline health fields
- **`data_incomplete`** → Boolean flag, per row (per ticker where applicable).
- **`Notes`** → String or list of short messages describing gaps, fallbacks, or assumptions.

Examples:
- `compute_fundamentals_actuals` → `"Revenue series too short (<3 yrs)"`, `"FCF CAGR undefined (non-positive endpoints)"`.
- `historical_average_share_prices` → `"No data to compute 180d average"`.
- `compare_to_market_cap` → `"Minority Interest missing; treated as 0"`.
- `dcf_three_scenarios` → `"Historical FCF highly volatile (CoV > threshold)"`.

These fields are attached at the **row level** (so multi-ticker outputs have different notes per ticker).

## Orchestrator consolidation
The `orchestrator_function` automatically collects health notes from all sub-functions and normalizes them into a **list of blocks**. Each block has:

```json
{
  "source": "compute_fundamentals_actuals",
  "ticker": "MSFT",
  "data_incomplete": true,
  "notes": [
    "Revenue series too short (<3 yrs)",
    "FCF CAGR undefined (non-positive endpoints)"
  ]
}
```

## Rendering health reports
Three helper functions convert health reports into human-friendly formats:
- `health_to_tables(health_report)` → Normalizes into a pandas.DataFrame with columns: Source | Ticker | Data Incomplete | Notes
- `health_to_markdown(health_report)` → Renders the same as a Markdown table.
- `save_health_report_excel(health_report, path="...")` → Saves both the normalized table and the raw JSON to Excel.

These helpers accept both:
- the new schema (list of blocks, per-function/per-ticker), and
- the legacy schema (dict with notes_for_llm, method_status, etc.) for backward compatibility.

## Notes - How to Access
Notes are usually returned as a column (e.g., in dcf_three_scenarios) or as a dict alongside the main DataFrame. Examples you might see:
- "TTM constructed from last 4 quarters"
- "Growth capped at WACC−0.5%"
- "Revenue CAGR based on 3 annual points only"
- "Beta missing → cannot compute WACC"
- "Net debt missing → approximated from latest balance sheet"

These make the analysis explainable – you can see not just the numbers, but also the assumptions and shortcuts applied under the hood.

## Usage example
```python
out = orchestrator_function(
    target_ticker="NVDA",
    peer_tickers=["NVDA","MSFT","META","GOOGL"],
)

# View structured health notes
print(health_to_markdown(out["data_health_report"]))

# Save to Excel
save_health_report_excel(out["data_health_report"], path="output/NVDA_health.xlsx")
```

## Interpretation Guide
- High missing_pct (>20%) → metric may not be reliable across tickers.
- Low n_unique (e.g., 1) → values may be stale or incomplete (e.g., beta not reported).
- Short CAGR windows → flagged in notes, meaning growth scores may be noisy.
- Zero/NaN scores → not an error, but a signal that data was unavailable or flagged for caution.

⚠️ **Best practice**: Always review health notes alongside metrics before drawing conclusions. A "perfect-looking" score may hide weak underlying coverage.

# 11. Summary of Functions

This section provides a quick reference to the public functions in the library. Ordered by workflow stage (from single-ticker fundamentals → peer comparisons → valuation → visualization → health). Each row shows the task, the function(s), a plain-language purpose, key options, and a brief note on what kind of health/notes flags you may encounter. Internal helpers (e.g. _fetch_timeseries_for_plot) are intentionally excluded.

| Task | Function(s) | Purpose | Key Options | Notes / Data Health |
|------|-------------|---------|-------------|---------------------|
| Single-company snapshot | `fundamentals_ttm_vs_average` | Compare TTM vs multi-year averages for one ticker | `basis="annual"` or `"ttm"` | Notes on missing items, short history |
| Multi-company fundamentals | `compute_fundamentals_actuals` | Compute raw metrics (profitability, growth, reinvestment, risk) | `basis`, `save_csv`, `as_df` | Missing values flagged; CAGR needs ≥3 years |
| Price snapshots | `historical_average_share_prices` | Return average prices over 1d/30d/90d/180d windows | `save_csv`, `as_df` | Returns Ticker, avg_price_1d/30d/90d/180d, price_asof, Notes |
| Fundamental scoring | `compute_fundamentals_scores` | Apply 0–5 scoring, roll up to factors and total | `basis`, `merge_with_actuals` | Includes Notes/data_incomplete when merge_with_actuals=True |
| Peer multiples | `peer_multiples` | Per-share valuation bands from P25/P50/P75 multiples | `basis="ttm"`, peers list | Peer choice critical; NaNs flagged if ratios missing |
| Peer multiples → price | `price_from_peer_multiples` | Convert peer multiple bands (PE/PS/EV-EBITDA) into implied per-share values | `ticker` (optional), `save_csv`, `as_df` | Uses latest shares, debt, cash, MI for EV→Equity bridge; echoes Inputs_Used for auditability |
| Growth metrics | `historical_growth_metrics` | Compute Revenue, Earnings, and FCF CAGRs with clear start/end years | `min_years`, `save_csv`, `as_df` | FCF CAGR requires ≥3 annual points and positive endpoints; Notes field explains gaps |
| Implied EV | `dcf_implied_enterprise_value` | Compute EV from normalized historical FCF & growth with enhanced validation | `window_years`, `risk_free_rate`, `equity_risk_premium`, `growth`, `volatility_threshold` | Prioritizes FCF CAGR over revenue CAGR; normalized FCF baseline; volatility warnings; reasonableness checks for EV/FCF multiples |
| Compare EVs | `compare_to_market_ev` | Compare implied vs observed EV with enhanced market data validation | `ticker`, `years`, `growth`, `volatility_threshold` | Nuanced premium interpretation (5%, 20% thresholds); fallback EV estimation from market cap + debt - cash |
| Equity bridge | `implied_equity_value_from_ev` | Translate EV → equity → per-share with robust balance sheet handling | `ticker`, `implied_ev`, `notes_from_ev` | Has multiple fallback sources for debt/cash data; per-share calculation added; enhanced validation warnings for negative equity and high leverage |
| DCF (scenario-based) | `dcf_three_scenarios` | Compute Low/Mid/High DCF per-share values using proper WACC with advanced controls for transformational companies | `years`, `peer_tickers`, `risk_free_rate`, `equity_risk_premium`, `target_cagr_fallback`, `fcf_window_years`, `manual_baseline_fcf`, `manual_growth_rates` | Proper WACC including debt costs; prioritizes FCF CAGR; corrected risk-growth relationship; enhanced FCF normalization; NEW advanced controls for high-growth/transformational companies; EV/FCF multiple warnings |
| Visualize fundamentals | `plot_single_metric_ts`, `plot_multi_tickers_multi_metrics_ts` | Time-series plots (single/multi metrics) | `family`, `metrics`, `basis` | Missing values → gaps in chart |
| Visualize scores | `plot_scores_clustered` | Clustered bar charts of scores across tickers | `metrics`, `include_total` | Uses scored DataFrame; flags carry through |
| Visualize multiples | `plot_peer_metric_boxplot` | Boxplot + target overlay using peer_comp_detail + peer_multiple_bands_wide | `metric`, `target_ticker` | Target shown even if excluded from stats |
| Visualize DCF | `plot_dcf_scenarios_vs_price` | Plot Low/Mid/High DCF scenario values | `ticker`, scenario DataFrame | Health notes appear in legend/labels |
| Data health reporting | `health_to_tables`, `health_to_markdown`, `save_health_report_excel` | Normalize and render health notes across all functions | `as_df`, `sort`, output path | New schema supports per-ticker blocks (Source, Ticker, Data Incomplete, Notes) |
| Automated orchestrator | `orchestrator_function` | Run complete workflow (fundamentals → scores → multiples → EV → Equity → DCF) | `target_ticker`, `peer_tickers`, `basis`, `years`, WACC/growth options, `save_csv` | Returns structured dict with all outputs; data_health_report is list of blocks (per-function and per-ticker) |

**Tip**: Always check the Notes / Data Health outputs. They are designed to surface issues (e.g., missing beta, truncated growth windows, NaNs) that might otherwise be hidden in the numbers.

# 12. Automated Valuation Orchestrator

The `orchestrator_function` provides a single-call way to run the **entire valuation workflow**:

- Computes historical price anchors (1d/30d/90d/180d).
- Extracts growth metrics (Revenue/Earnings/FCF CAGRs).
- Computes fundamentals (actuals + scored factors).
- Runs peer multiples and converts to implied per-share prices.
- Compares implied vs observed Enterprise Value.
- Compares implied equity value vs market capitalization.
- Runs DCF (three scenarios: Low/Mid/High).
- Aggregates data health notes into a standardized schema (list of per-function or per-ticker blocks).

This provides a single-call snapshot suitable for:
- Screening many companies quickly.
- Producing audit-friendly summary tables.
- Feeding downstream apps or LLM agents that need the "big picture" at once.

## Options & defaults
```python
orchestrator_function(
    target_ticker: str,
    peer_tickers: list[str],
    *,
    include_target_in_peers: bool = False,
    years: int = 5,
    risk_free_rate: float = 0.045,
    equity_risk_premium: float = 0.060,
    growth: float | None = None,
    target_cagr_fallback: float = 0.02,
    use_average_fcf_years: Optional[int]=3,
    volatility_threshold: float = 0.5,
    basis: str = "annual",
    save_csv: bool = False,
    output_dir: str = "output",
    analysis_report_date: str | None = None,
) -> dict[str, Any]
```

## Output
**Output keys**:
- `analysis_report_date` → ISO string
- `ticker` → target ticker
- `peer_tickers` → list of peers (with target always included)
- `avg_price_1d` / `30d` / `90d` / `180d` → price anchors
- `growth_metrics` → DataFrame of CAGRs
- `fundamentals_actuals` / `fundamentals_scores`
- `peer_multiples` / `price_from_peer_multiples`
- `ev_vs_market` / `market_cap_vs_equity`
- `dcf_scenarios`
- `data_health_report` → list of dicts with fields:
  - `source` (e.g., "compute_fundamentals_actuals")
  - `ticker` (if applicable)
  - `data_incomplete` (True/False/None)
  - `notes` (list of strings)

## Compute
```python
import ValueInvestingTools as vit

res = vit.orchestrator_function(
    target_ticker="NVDA",
    peer_tickers=["NVDA","MSFT","META","GOOGL"],
    years=5
)

# Show health notes in Markdown
print(vit.health_to_markdown(res["data_health_report"]))

# Save to Excel
vit.save_health_report_excel(res["data_health_report"], path="output/NVDA_health.xlsx")
```

## Navigating the outputs
The result is a nested structure (dict or dict-of-DataFrames depending on as_df). Typical keys:
```python
print(res.keys())
# dict_keys(['fundamentals','scores','peer_multiples',
#            'dcf_scenarios','implied_ev','equity_value','health'])

res["fundamentals"]     # → raw metrics DataFrame.
res["scores"]           # → scored fundamentals DataFrame (0–5 with rollups).
res["peer_multiples"]   # → dict of PE / PS / EV-EBITDA valuation bands.
res["dcf_scenarios"]    # → Low / Mid / High DCF per-share values.
res["implied_ev"]       # → single float (implied EV).
res["equity_value"]     # → DataFrame (equity bridge to per-share implied value).
res["health"]           # → list of diagnostic notes collected from all sub-functions.
```

## Interpretation Guide
Use this when you need a full audit trail in one call. Ideal for:
- Dashboards / reports (export each key to CSV).
- LLM or agent pipelines (one function call returns the entire valuation picture).
- Outputs can be verbose; analysts should still inspect sub-sections rather than relying on the aggregate blindly.

**Not a Black Box**: While convenient for screening, the orchestrator should not be used as a "set and forget" valuation tool. Always inspect individual components:

- **Review Health Notes**: The consolidated health field flags data quality issues, volatile inputs, and methodological limitations
- **Cross-Validate Results**: Compare DCF scenarios with peer multiples and implied EV for consistency
- **Assess Assumption Sensitivity**: Wide valuation ranges across methods indicate high uncertainty

**When Orchestrator Results May Be Misleading**:
- High FCF volatility (CoV >0.5) makes both DCF and implied EV unreliable
- Missing beta or debt data forces fallback assumptions
- Peer set includes companies with different business models or capital structures
- Target company undergoing major transitions not reflected in historical data

**Best Practices**:
1. Start with orchestrator for initial screening
2. Dive deeper into individual functions for material investment decisions  
3. Always read and understand health/notes flags before drawing conclusions
4. Consider multiple valuation approaches and business context before making investment decisions

# 13. Notes, Roadmap, Contributing

## Notes
- This library is intended for educational and analytical use only. It does not constitute investment advice.
- Outputs should always be interpreted alongside qualitative analysis (management quality, competitive dynamics, regulatory environment).
- Historical financials are taken from Yahoo Finance via yfinance; occasional gaps or inconsistencies are expected.

## Roadmap
Planned improvements for future revisions include:
- Extended health diagnostics (coverage scoring, sector-specific caveats).
- Optional alternative data sources beyond Yahoo Finance.
- Richer visualization (heatmaps for sensitivity grids, factor-radar plots).
- Additional factor modules (earnings quality, accruals, working capital intensity).
- CLI wrapper for running orchestrated workflows from the command line.

## Contributing
Contributions are welcome!
- Fork the repo, create a feature branch, and submit a pull request.
- Please align code style with existing modules and include basic automated test coverage (`unittest` or `pytest`).
- Suggestions for new metrics, visualization types, or health checks are especially appreciated.

### Local validation commands
```bash
python -m py_compile server.py ValueInvestingTools.py vit/__init__.py
python -m unittest discover -s tests -p "test_*.py" -v
```

# 14. License
This project is licensed under the MIT License – you are free to use, modify, and distribute it, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.

**
