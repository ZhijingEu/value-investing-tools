# ADR-0001: Fundamentals Scoring – MECE 4×4 Framework, Three Scoring Modes, Weighted Composite

## Status
Proposed (Draft, Not Yet Implemented in code as of 2026-02-22)

## Context
The fundamentals scoring module is being re-engineered to ensure:
- Factors are **MECE** (mutually exclusive, collectively exhaustive).  
- Each pillar carries the same number of sub-factors (4) to prevent bias.  
- The system supports **absolute** and **peer-relative** scoring.  
- An explicit **weighted composite** reflects the McKinsey value-driver intuition.  

Earlier versions over-relied on Beta as a risk proxy and mixed uneven sub-factor counts.  
This revision establishes a balanced 4×4 framework and formal scoring options.

---

## Decision
Adopt a **4 pillars × 4 sub-factors** framework with **three scoring modes** and a **weighted composite**.

### Pillars & Sub-Factors (MECE, 4 each)
**Profitability**
- Return on Equity (ROE)  
- Net Margin  
- Free Cash Flow Margin  
- Return on Invested Capital (ROIC)

**Growth**
- Revenue CAGR (3–5 yrs)  
- Net Income CAGR  
- Free Cash Flow CAGR  
- Total Asset CAGR

**Reinvestment**
- CapEx / Revenue  
- Fundamental Reinvestment Rate = YoY Revenue Growth ÷ ROIC  
- Reinvestment / NOPAT  
- Retention Ratio = 1 – Dividend Payout Ratio

**Risk**
- Debt/Equity Ratio  
- Operating Leverage = %Δ EBIT ÷ %Δ Revenue  
- **Altman Z-Score**  
- **Beneish M-Score**

> Unless noted, sub-factors use time-period averages; Growth uses CAGR; Altman Z and Beneish M use latest 2–3 periods.

---

## Scoring Modes
All share a “bigger = better” convention (Note please read below clarification)

1. **Rank-based (default)** – within a peer set of N tickers, assign rank N (best)…1 (worst).  
   - Ties → average ranks (e.g., 6, 5, 4, 2.5, 2.5, 1).  
   - Warn if N < 5 in data-health output.  
2. **Absolute thresholds (1–5 bands)** – see table below.  
3. **Peer-set Z-score** – computed across peer set; invert where lower = better.

### Clarification: “Bigger Is Better” Refers to Scores, Not Raw Metrics

The phrase “bigger is better” throughout this ADR applies **only to the scoring output**
(Rank, Absolute Threshold, or Z-Score), not the underlying raw financial metrics.

Some metrics are beneficial when larger (e.g., ROE, Net Margin, FCF Margin),
while others are preferable when smaller (e.g., Debt/Equity, CapEx/Revenue, Beneish M).

Each sub-factor carries a `higher_is_better` flag so that:
- Rank scores invert when a metric is “lower is better.”
- Absolute thresholds flip the lookup direction.
- Z-scores are sign-adjusted (multiplied by –1) where needed.

This guarantees that all **scores** remain on a consistent “higher = better” scale,
even when the raw metric direction differs.

---

## Threshold Bands (for Absolute 1–5 Mode)

| Sub-Factor | 1 (Weak) | 2 | 3 | 4 | 5 (Strong) | Dir |
|-------------|-----------|---|---|---|-------------|----|
| ROE | < 5% | 5–10% | 10–15% | 15–20% | > 20% | ↑ |
| Net Margin | < 3% | 3–6% | 6–10% | 10–15% | > 15% | ↑ |
| FCF Margin | < 2% | 2–5% | 5–8% | 8–12% | > 12% | ↑ |
| ROIC | < 4% | 4–6% | 6–9% | 9–12% | > 12% | ↑ |
| Revenue CAGR | < 0% | 0–3% | 3–6% | 6–9% | > 9% | ↑ |
| Net Income CAGR | < 0% | 0–3% | 3–6% | 6–9% | > 9% | ↑ |
| FCF CAGR | < 0% | 0–3% | 3–6% | 6–9% | > 9% | ↑ |
| Total Asset CAGR | < 0% | 0–2% | 2–4% | 4–6% | > 6% | ↑ |
| CapEx / Revenue | > 10% | 8–10% | 6–8% | 4–6% | < 4% | ↓ |
| Fundamental Reinv. Rate | < 0.3× | 0.3–0.6× | 0.6–0.9× | 0.9–1.2× | > 1.2× | ↑ |
| Reinv. / NOPAT | > 100% | 80–100% | 60–80% | 40–60% | < 40% | ↓ |
| Retention Ratio | < 0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | > 0.9 | ↑ |
| Debt/Equity | > 2.5× | 2.0–2.5× | 1.5–2.0× | 1.0–1.5× | < 1.0× | ↓ |
| Operating Leverage | < 0.3× | 0.3–0.6× | 0.6–0.9× | 0.9–1.2× | > 1.2× | ↑ |
| Altman Z | < 1.8 | 1.8–2.2 | 2.2–2.9 | 2.9–3.5 | > 3.5 | ↑ |
| Beneish M | > –1.5 | –1.7 to –1.5 | –1.9 to –1.7 | –2.1 to –1.9 | < –2.1 | ↓ |

### Note on Altman Z Applicability

The absolute-band thresholds for **Altman Z** are calibrated for **non-financial corporates**
(industrials, consumer, tech, etc.).  
They are **not appropriate for banks, insurance companies, or other financial institutions**
whose balance-sheet structures differ fundamentally.

When analyzing financial-sector tickers:

- Prefer **Rank** or **peer-set Z-Score** modes over absolute bands.
- The system will emit a `data_health.warning` such as:
  “Altman Z absolute thresholds calibrated for non-financials; use Rank/Z for financials.”
  
---

## Pillar & Overall Scores
- Pillar = avg of its 4 sub-factors (using selected mode).  
- Overall (defaults, overrideable in-function):  
  `overall = 1.0*Profitability + 0.7*Reinvestment + 0.8*Growth + 1.8*Risk`

---

## API / UX Design
- Keep function names (for backward compatibility).  
- Accept multiple tickers.  
- Return structured object + DataFrames + Matplotlib figures.  
- Missing data ⇒ drop ticker, record in data_health.  
- Data Health warns if peer set < 5.  

---

## Consequences
- Balanced, MECE, production-ready fundamentals framework.  
- Risk pillar now anchors on Altman Z and Beneish M (internal sub-factors).  
- Remaining reference model = Piotroski F-Score (see ADR-0002).  

