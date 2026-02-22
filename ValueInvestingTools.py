# ValueInvestingTools.py Rev 49.2

from __future__ import annotations

import os
import json
import math
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import re
import inspect

# -----------------------------
# General helpers
# -----------------------------
def _today_iso() -> str:
    return dt.date.today().isoformat()

def _sanitize_ticker(t: str) -> str:
    # Basic normalizer for common quirks (e.g., BRK.B -> BRK-B); safe no-op otherwise.
    return t.replace(".", "-").strip().upper()

def _is_num(x) -> bool:
    try:
        return (x is not None) and np.isfinite(float(x))
    except Exception:
        return False

def _is_pos(x) -> bool:
    try:
        v = float(x)
        return np.isfinite(v) and v > 0
    except Exception:
        return False

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _pct_from_info(info: dict, key: str) -> Optional[float]:
    v = _safe_float(info.get(key))
    return v  # keep as decimal, no *100

def to_records(obj, analysis_report_date: Optional[str] = None, schema_version: str = "1.0", notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convert DataFrame / dict / list[dict] to an LLM-friendly JSON envelope with flat primitives.
    """
    if analysis_report_date is None:
        analysis_report_date = _today_iso()

    if isinstance(obj, pd.DataFrame):
        data = json.loads(obj.to_json(orient="records"))
    elif isinstance(obj, dict):
        # assume already serializable; wrap as a single record
        data = [obj]
    elif isinstance(obj, list):
        data = obj
    else:
        data = [{"value": str(obj)}]

    # replace NaN with None
    def _nan_to_none(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    if isinstance(data, list):
        data = [{k: _nan_to_none(v) for k, v in row.items()} if isinstance(row, dict) else row for row in data]

    return {
        "schema_version": schema_version,
        "analysis_report_date": analysis_report_date,
        "data": data,
        "notes": notes or []
    }


# Centralized valuation defaults (can be overridden per call).
VALUATION_DEFAULTS: Dict[str, float] = {
    "risk_free_rate": 0.045,
    "equity_risk_premium": 0.060,
    "target_cagr_fallback": 0.020,
    "fcf_window_years": 3,
    "terminal_growth_gap": 0.005,  # g <= WACC - gap
}


def valuation_defaults(
    *,
    as_of_date: Optional[str] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: Optional[float] = None,
    target_cagr_fallback: Optional[float] = None,
    fcf_window_years: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return a normalized assumptions payload for valuation outputs and audits.
    """
    return {
        "as_of_date": as_of_date or _today_iso(),
        "risk_free_rate": VALUATION_DEFAULTS["risk_free_rate"] if risk_free_rate is None else float(risk_free_rate),
        "equity_risk_premium": VALUATION_DEFAULTS["equity_risk_premium"] if equity_risk_premium is None else float(equity_risk_premium),
        "target_cagr_fallback": VALUATION_DEFAULTS["target_cagr_fallback"] if target_cagr_fallback is None else float(target_cagr_fallback),
        "fcf_window_years": VALUATION_DEFAULTS["fcf_window_years"] if fcf_window_years is None else int(fcf_window_years),
        "terminal_growth_gap": VALUATION_DEFAULTS["terminal_growth_gap"],
    }

# ==========================================================
# FUNDAMENTALS (actuals first; scores on demand; optional merge)
# (Derived from your FundamentalFactorsTool_Ver2.py)
# ==========================================================

# Scoring thresholds & weights (unchanged semantics)
_THRESHOLDS = {
    'roe': [0.05, 0.10, 0.15, 0.20],                 # decimals
    'profit_margin': [0.05, 0.10, 0.15, 0.20],       # decimals
    'op_margin': [0.05, 0.10, 0.15, 0.20],           # decimals
    'roa': [0.03, 0.06, 0.10, 0.15],                 # decimals
    'revenue_cagr': [0.00, 0.05, 0.10, 0.15],        # decimals
    'earnings_cagr': [0.00, 0.05, 0.10, 0.15],       # decimals
    'peg': [3, 2, 1.5, 1],                           # lower is better 
    'reinvestment_rate': [0.10, 0.25, 0.50, 0.75],   # decimals (higher is better)
    'capex_ratio': [0.10, 0.05, 0.02, 0.00],         # decimals (lower is better)
    'de_ratio': [2, 1.5, 1.0, 0.5],                  # lower is better 
    'beta': [2, 1.5, 1.2, 0.8],                      # lower is better 
    'current_ratio': [0.8, 1.0, 1.5, 2.0]            # higher is better 
}

_WEIGHTS = {
    'profitability': {'roe': 0.3, 'profit_margin': 0.25, 'op_margin': 0.25, 'roa': 0.2},
    'growth': {'revenue_cagr': 0.4, 'earnings_cagr': 0.4, 'peg': 0.2},
    'reinvestment': {'reinvestment_rate': 0.6, 'capex_ratio': 0.4},
    'risk': {'de_ratio': 0.4, 'beta': 0.4, 'current_ratio': 0.2}
}

_RAW_METRICS = [
    'roe','profit_margin','op_margin','roa',
    'revenue_cagr','earnings_cagr','peg',
    'reinvestment_rate','capex_ratio',
    'de_ratio','beta','current_ratio'
]

def _public_suffix_for(basis: Literal["annual","ttm"], metric_key: str) -> str:
    """
    Public label suffix rules:
    - CAGRs (revenue_cagr, earnings_cagr): always '-Ave' (historical window; no TTM analog)
    - PEG, Beta: always '-TTM' (point-in-time from yfinance.info)
    - ROE/ROA/margins, reinvestment_rate, capex_ratio, de_ratio, current_ratio: follow `basis`:
    - 'annual' → '-Ave' (multi-year average from annual statements)
    - 'ttm'    → '-TTM' (TTM or latest-Q derived)
    """
    if metric_key in ("revenue_cagr", "earnings_cagr"):
        return "-Ave"
    if metric_key in ("peg", "beta"):
        return "-TTM"
    return "-TTM" if basis == "ttm" else "-Ave"

def _make_rename_map(basis: Literal["annual","ttm"]) -> dict[str, str]:
    # base → public label (metrics)
    m = {
        "roe":               f"Profitability-ROE{_public_suffix_for(basis,'roe')}",
        "profit_margin":     f"Profitability-NetMargin{_public_suffix_for(basis,'profit_margin')}",
        "op_margin":         f"Profitability-OpMargin{_public_suffix_for(basis,'op_margin')}",
        "roa":               f"Profitability-ROA{_public_suffix_for(basis,'roa')}",
        "revenue_cagr":      f"Growth-RevenueCAGR{_public_suffix_for(basis,'revenue_cagr')}",    # always -Ave
        "earnings_cagr":     f"Growth-EarningsCAGR{_public_suffix_for(basis,'earnings_cagr')}",   # always -Ave
        "peg":               f"Growth-PEG{_public_suffix_for(basis,'peg')}",                      # always -TTM
        "reinvestment_rate": f"Reinvestment-ReinvestmentRate{_public_suffix_for(basis,'reinvestment_rate')}",
        "capex_ratio":       f"Reinvestment-CapexRatio{_public_suffix_for(basis,'capex_ratio')}",
        "de_ratio":          f"Risk-DebtEquityRatio{_public_suffix_for(basis,'de_ratio')}",
        "beta":              f"Risk-Beta{_public_suffix_for(basis,'beta')}",                      # always -TTM
        "current_ratio":     f"Risk-CurrentRatio{_public_suffix_for(basis,'current_ratio')}",
    }
    # score_* → public label
    s = {
        "score_roe":               f"score_Profitability-ROE{_public_suffix_for(basis,'roe')}",
        "score_profit_margin":     f"score_Profitability-NetMargin{_public_suffix_for(basis,'profit_margin')}",
        "score_op_margin":         f"score_Profitability-OpMargin{_public_suffix_for(basis,'op_margin')}",
        "score_roa":               f"score_Profitability-ROA{_public_suffix_for(basis,'roa')}",
        "score_revenue_cagr":      f"score_Growth-RevenueCAGR{_public_suffix_for(basis,'revenue_cagr')}",
        "score_earnings_cagr":     f"score_Growth-EarningsCAGR{_public_suffix_for(basis,'earnings_cagr')}",
        "score_peg":               f"score_Growth-PEG{_public_suffix_for(basis,'peg')}",
        "score_reinvestment_rate": f"score_Reinvestment-ReinvestmentRate{_public_suffix_for(basis,'reinvestment_rate')}",
        "score_capex_ratio":       f"score_Reinvestment-CapexRatio{_public_suffix_for(basis,'capex_ratio')}",
        "score_de_ratio":          f"score_Risk-DebtEquityRatio{_public_suffix_for(basis,'de_ratio')}",
        "score_beta":              f"score_Risk-Beta{_public_suffix_for(basis,'beta')}",
        "score_current_ratio":     f"score_Risk-CurrentRatio{_public_suffix_for(basis,'current_ratio')}",
    }
    m.update(s)
    return m

def _fetch_statements(ticker: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    tkr = yf.Ticker(ticker)
    info = tkr.info
    financials = tkr.financials
    cashflow = tkr.cashflow
    balance = tkr.balance_sheet

    years = []
    if isinstance(financials, pd.DataFrame): years = [c for c in financials.columns]
    if isinstance(cashflow, pd.DataFrame):   years = list(set(years) & set(cashflow.columns)) if years else list(cashflow.columns)
    if isinstance(balance, pd.DataFrame):    years = list(set(years) & set(balance.columns)) if years else list(balance.columns)
    period_range = f"{min(years).year}-{max(years).year}" if len(years) else "N/A"
    min_periods = len(years)

    return info, financials, cashflow, balance, period_range, min_periods

def _score_metric(value: Optional[float], metric: str) -> Optional[int]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    cuts = _THRESHOLDS[metric]
    if metric in ['peg', 'capex_ratio', 'de_ratio', 'beta']:  # lower is better
        if value <= cuts[3]: return 5
        if value <= cuts[2]: return 4
        if value <= cuts[1]: return 3
        if value <= cuts[0]: return 2
        return 1
    else:  # higher is better
        if value >= cuts[3]: return 5
        if value >= cuts[2]: return 4
        if value >= cuts[1]: return 3
        if value >= cuts[0]: return 2
        return 1

def _strategic_recommendation(row: pd.Series) -> Tuple[str, str]:
    factors = ['profitability_score', 'growth_score', 'reinvestment_score', 'risk_score']
    factor_scores = {f: row.get(f) for f in factors}
    if any(pd.isna(v) for v in factor_scores.values()):
        return "Inconclusive", "Missing factor scores."

    min_score = min(factor_scores.values())
    avg_pg = (row['profitability_score'] + row['growth_score']) / 2

    missing_count = sum(pd.isna(row[col]) for col in _RAW_METRICS)
    data_incomplete = (missing_count / len(_RAW_METRICS)) > 0.25

    total = row['total_score']
    if data_incomplete:
        return "Inconclusive", "More than 25% of input metrics missing."

    if total >= 17:
        if min_score == 1:
            return "Uneven Fundamentals", "At least one factor score is critically low."
        if factor_scores['risk_score'] < 2.5:
            return "Uneven Fundamentals", "Risk score below 2.5."
        if factor_scores['reinvestment_score'] < 2.0:
            return "Uneven Fundamentals", "Reinvestment score below 2.0."
        if avg_pg < 2.5:
            return "Uneven Fundamentals", "Weak profitability + growth despite high total."
        if total >= 18 and all(v >= 4 for v in factor_scores.values()):
            return "Elite Performer", "Exceptional balance across all four factors."
        return "Resilient Core", "Meets threshold with no red flags."

    if 14 <= total < 17:
        if factor_scores['risk_score'] < 2.0 or factor_scores['reinvestment_score'] < 2.0:
            return "Uneven Fundamentals", "Moderate total but risk/reinvestment weak."
        return "Resilient Core", "Moderate fundamentals with manageable risk."

    if 11 <= total < 14:
        return "Uneven Fundamentals", "Mixed fundamentals."

    return "Weak Fundamentals", "Low total score."

def compute_fundamentals_actuals(
    tickers: List[str],
    *,
    output_dir: str = "output",
    save_csv: bool = False,           
    as_df: bool = True,
    basis: Literal["annual","ttm"] = "annual",
    analysis_report_date: Optional[str] = None
):
    """
    Compute absolute (non-scored) fundamentals for tickers.

    Key behaviors:
    - Units: returns decimals (e.g., 0.35 = 35%).
    - Basis switch: `basis="annual"` (default) computes multi-year averages from annual statements; 
    `basis="ttm"` uses TTM / latest-Q derived figures where applicable.
    - Suffixing: public column names include '-Ave' or '-TTM' per `_public_suffix_for`.
    - CAGRs: full available annual window (>=3 points), always labeled '-Ave'.
    - CSV: writing is opt-in (`save_csv=False` by default).
    - Outputs: DataFrame by default; JSON-serializable via `as_df=False`.
    """
    suffix = "-Ave" if basis == "annual" else "-TTM" 
    analysis_report_date = analysis_report_date or _today_iso()
    rows = []

    for raw in tickers:
        ticker = _sanitize_ticker(raw)
        try:
            info, financials, cashflow, balance, period_range, min_periods = _fetch_statements(ticker)

            # persist raw statements
            if save_csv:
                tdir = os.path.join(output_dir, ticker)
                os.makedirs(tdir, exist_ok=True)
                if isinstance(financials, pd.DataFrame) and not financials.empty: financials.T.to_csv(os.path.join(tdir, "income_statement.csv"))
                if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:     cashflow.T.to_csv(os.path.join(tdir, "cashflow.csv"))
                if isinstance(balance, pd.DataFrame) and not balance.empty:       balance.T.to_csv(os.path.join(tdir, "balance_sheet.csv"))

            roe = _pct_from_info(info, 'returnOnEquity')
            roa = _pct_from_info(info, 'returnOnAssets')
            op_margin = _pct_from_info(info, 'operatingMargins')
            profit_margin = _pct_from_info(info, 'profitMargins')
            current_ratio = None

            # --- Profitability & Liquidity per basis ---
            try:
                prof_ts = compute_profitability_timeseries(
                    ticker, include_ttm=True if basis == "ttm" else False,
                    as_df=True, return_format="wide"
                )
                if basis == "annual":
                    # average across available annual rows (drop TTM if present)
                    prof_annual = prof_ts.drop(index="TTM", errors="ignore")
                    if not prof_annual.empty:
                        roe           = float(prof_annual["ROE"].mean(skipna=True))
                        profit_margin = float(prof_annual["NetMargin"].mean(skipna=True))
                        op_margin     = float(prof_annual["OpMargin"].mean(skipna=True))
                        roa           = float(prof_annual["ROA"].mean(skipna=True))
                else:
                    # TTM row
                    if isinstance(prof_ts, pd.DataFrame) and "TTM" in prof_ts.index:
                        roe           = _safe_float(prof_ts.loc["TTM","ROE"])
                        profit_margin = _safe_float(prof_ts.loc["TTM","NetMargin"])
                        op_margin     = _safe_float(prof_ts.loc["TTM","OpMargin"])
                        roa           = _safe_float(prof_ts.loc["TTM","ROA"])
            except Exception:
                pass

            # Current Ratio per basis
            try:
                liq_ts = compute_liquidity_timeseries(
                    ticker, include_ttm=True if basis == "ttm" else False,
                    as_df=True, return_format="wide"
                )
                if basis == "annual":
                    liq_annual = liq_ts.drop(index="TTM", errors="ignore")
                    if not liq_annual.empty:
                        current_ratio = float(liq_annual["CurrentRatio"].mean(skipna=True))
                else:
                    if isinstance(liq_ts, pd.DataFrame) and "TTM" in liq_ts.index:
                        current_ratio = _safe_float(liq_ts.loc["TTM","CurrentRatio"])
            except Exception:
                pass

            peg_raw = _safe_float(info.get('trailingPegRatio'))
            peg = peg_raw if (peg_raw is not None and math.isfinite(peg_raw) and peg_raw > 0) else None

            beta_raw = _safe_float(info.get('beta'))
            beta = beta_raw if (beta_raw is not None and math.isfinite(beta_raw)) else None

            # Debt/Equity multi-year average
            try:
                de_vals = []
                if isinstance(balance, pd.DataFrame) and not balance.empty:
                    # NOTE: use .iloc for positional column access (FutureWarning fix)
                    if 'Total Debt' in balance.index and 'Common Stock Equity' in balance.index:
                        total_debt_row = balance.loc['Total Debt']
                        common_eq_row = balance.loc['Common Stock Equity']
                        for i in range(balance.shape[1]):
                            total_debt_i = _safe_float(total_debt_row.iloc[i])
                            common_eq_i  = _safe_float(common_eq_row.iloc[i])
                            if _is_pos(total_debt_i) and _is_pos(common_eq_i):
                                de_vals.append(total_debt_i / common_eq_i)
                de_ratio = float(np.mean(de_vals)) if len(de_vals) >= 2 else None
            except Exception:
                de_ratio = None

            cr_info = _safe_float(info.get('currentRatio'))
            # only use the info snapshot if the per-basis computation failed
            if 'current_ratio' not in locals() or current_ratio is None:
                current_ratio = cr_info

            # --- TTM flows (from quarterlies) for reinvestment/capex; and point-in-time D/E ---
            try:
                tkr = yf.Ticker(ticker)
                is_q = tkr.quarterly_financials
                cf_q = tkr.quarterly_cashflow
                bs_q = tkr.quarterly_balance_sheet

                s_ni_q    = _alias_first(is_q, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])
                s_rev_q   = _alias_first(is_q, ["Total Revenue","Revenue"])
                s_div_q   = _alias_first(cf_q, ["Common Stock Dividend Paid"])
                s_capex_q = _alias_first(cf_q, ["Capital Expenditure","Capital Expenditures"])
                s_debt_q  = _alias_first(bs_q, ["Total Debt","Total Debt Net","Long Term Debt","Short Long Term Debt"])
                s_equ_q   = _alias_first(bs_q, ["Common Stock Equity","Total Stockholder Equity","Total Equity"])

                ni_ttm    = _sum_last_4_quarters(s_ni_q) if s_ni_q is not None else None
                rev_ttm   = _sum_last_4_quarters(s_rev_q) if s_rev_q is not None else None
                div_ttm   = _sum_last_4_quarters(s_div_q) if s_div_q is not None else None
                capex_ttm = _sum_last_4_quarters(s_capex_q) if s_capex_q is not None else None

                # Latest quarter stocks
                debt_latest   = None if s_debt_q is None or s_debt_q.empty else float(s_debt_q.iloc[0])
                equity_latest = None if s_equ_q  is None or s_equ_q.empty  else float(s_equ_q.iloc[0])

                reinvestment_rate_ttm = None
                if ni_ttm not in (None, 0) and div_ttm is not None:
                    reinvestment_rate_ttm = (float(ni_ttm) + float(div_ttm)) / float(ni_ttm)

                capex_ratio_ttm = None
                if capex_ttm is not None and rev_ttm not in (None, 0):
                    capex_ratio_ttm = abs(float(capex_ttm)) / float(rev_ttm)

                de_ratio_ttm = None
                if equity_latest not in (None, 0) and debt_latest is not None:
                    de_ratio_ttm = float(debt_latest) / float(equity_latest)

                # Select values by basis
                if basis == "ttm":
                    if reinvestment_rate_ttm is not None: reinvestment_rate = reinvestment_rate_ttm
                    if capex_ratio_ttm is not None:       capex_ratio       = capex_ratio_ttm
                    if de_ratio_ttm is not None:          de_ratio          = de_ratio_ttm
            except Exception:
                pass

            try:
                revenue_cagr = earnings_cagr = None
                if isinstance(financials, pd.DataFrame) and not financials.empty:
                    inc = financials.loc[["Total Revenue", "Net Income"]]
                    if inc.shape[1] >= 3:
                        revs = inc.loc["Total Revenue"].dropna().astype(float)
                        ni   = inc.loc["Net Income"].dropna().astype(float)

                        # Ensure chronological order: oldest → newest
                        years = [int(pd.to_datetime(c).year) for c in revs.index]
                        order = np.argsort(years)
                        revs = revs.iloc[order]
                        ni   = ni.iloc[order]

                        # Revenue CAGR
                        if len(revs) >= 3 and revs.iloc[0] > 0 and revs.iloc[-1] > 0:
                            n = len(revs) - 1
                            revenue_cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1/n) - 1.0

                        # Earnings CAGR
                        if len(ni) >= 3 and ni.iloc[0] > 0 and ni.iloc[-1] > 0:
                            n = len(ni) - 1
                            earnings_cagr = (ni.iloc[-1] / ni.iloc[0]) ** (1/n) - 1.0
            except Exception:
                revenue_cagr = earnings_cagr = None

            # Reinvestment rate avg
            try:
                reinvestment_rates = []
                if isinstance(cashflow, pd.DataFrame) and isinstance(financials, pd.DataFrame):
                    for i in range(min(cashflow.shape[1], financials.shape[1])):
                        if "Common Stock Dividend Paid" in cashflow.index:
                            dividends = cashflow.loc["Common Stock Dividend Paid"].iloc[i]
                        else:
                            dividends = 0
                        if "Net Income" in financials.index:
                            net_income = financials.loc["Net Income"].iloc[i]
                        else:
                            net_income = 0

                        dividends = 0 if pd.isna(dividends) else dividends
                        net_income = 0 if pd.isna(net_income) else net_income
                        if net_income != 0:
                            retained = net_income + dividends
                            reinvestment_rates.append((retained / net_income))
                reinvestment_rate_avg = float(np.mean(reinvestment_rates)) if len(reinvestment_rates) > 0 else None
            except Exception:
                reinvestment_rate_avg = None

            # Capex ratio avg
            try:
                capex_list, revenue_list = [], []
                if isinstance(cashflow, pd.DataFrame) and isinstance(financials, pd.DataFrame):
                    for i in range(min(cashflow.shape[1], financials.shape[1])):
                        capex = None
                        revenue = None
                        if 'Capital Expenditure' in cashflow.index:
                            capex = cashflow.loc['Capital Expenditure'].iloc[i]
                        if 'Total Revenue' in financials.index:
                            revenue = financials.loc['Total Revenue'].iloc[i]            
                        if capex is not None and revenue is not None and float(revenue) > 0:
                            capex_list.append(float(abs(capex)))   # negative CapEx (outflow) → use magnitude
                            revenue_list.append(float(revenue))
                capex_ratio_avg = (float(np.mean([c / r for c, r in zip(capex_list, revenue_list)]))
                                if len(capex_list) >= 1 else None)
            except Exception:
                capex_ratio_avg = None

            # --- Final per-basis selection ---
            # de_ratio already has a multi-year average; de_ratio_ttm was built earlier.
            if basis == "ttm":
                if 'reinvestment_rate_ttm' in locals() and reinvestment_rate_ttm is not None:
                    reinvestment_rate = reinvestment_rate_ttm
                else:
                    reinvestment_rate = locals().get('reinvestment_rate_avg')
                if 'capex_ratio_ttm' in locals() and capex_ratio_ttm is not None:
                    capex_ratio = capex_ratio_ttm
                else:
                    capex_ratio = locals().get('capex_ratio_avg')
                if 'de_ratio_ttm' in locals() and de_ratio_ttm is not None:
                    de_ratio = de_ratio_ttm
                # else keep de_ratio (average) as is
            else:
                # basis == "annual": use the averages if available
                reinvestment_rate = locals().get('reinvestment_rate_avg', locals().get('reinvestment_rate'))
                capex_ratio       = locals().get('capex_ratio_avg',       locals().get('capex_ratio'))
                # de_ratio already set to the multi-year average above

            # Period labeling that reflects basis
            period_range_out = period_range
            period_count_out = min_periods
            if basis == "ttm":
                period_range_out = "TTM (last 4Q)"
                period_count_out = 4

            rows.append({
                "ticker": ticker,
                "roe": roe, "profit_margin": profit_margin, "op_margin": op_margin, "roa": roa,
                "revenue_cagr": revenue_cagr, "earnings_cagr": earnings_cagr, "peg": peg,
                "reinvestment_rate": reinvestment_rate, "capex_ratio": capex_ratio,
                "de_ratio": de_ratio, "beta": beta, "current_ratio": current_ratio,
                "period_range": period_range_out, "period_count": period_count_out})

        except Exception as e:
            rows.append({
                "ticker": ticker, "roe": None, "profit_margin": None, "op_margin": None, "roa": None,
                "revenue_cagr": None, "earnings_cagr": None, "peg": None,
                "reinvestment_rate": None, "capex_ratio": None, "de_ratio": None,
                "beta": None, "current_ratio": None, "period_range": None, "period_count": 0,
                "_error": f"{type(e).__name__}: {e}"
            })

    df = pd.DataFrame(rows)

    # Optional data completeness flag, handy for LLM narratives
    df["data_incomplete"] = df[_RAW_METRICS].isna().sum(axis=1) / len(_RAW_METRICS) > 0.25
    
    # Apply public-facing labels with correct suffixes (per basis/metric capability)
    public = df.rename(columns=_make_rename_map(basis))    

    if as_df:
        return public
    return to_records(public, analysis_report_date=analysis_report_date)

def compute_fundamentals_scores(
    actuals: pd.DataFrame | List[Dict[str, Any]] | List[str],
    *,
    as_df: bool = True,
    merge_with_actuals: bool = False,
    analysis_report_date: Optional[str] = None,
    use_timeseries_averages_for_profitability: bool = False, 
    return_timeseries: bool = False,
    basis: Literal["annual","ttm"] = "annual"
):
    """
    Given actuals (DataFrame, list[dict], or list[str]), produce per-metric scores (score_*) and factor rollups.
    - Expects decimals (e.g., 0.35 for 35%).
    - If given a list of tickers, the function builds actuals with the same basis defaults.
    - Public score columns mirror the same '-Ave' / '-TTM' suffix rules as actuals.
    - Return either scores-only or merge with actuals via `merge_with_actuals`.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    # Accept either a DataFrame, list[dict], or list[str] of tickers
    if isinstance(actuals, list) and len(actuals) > 0 and isinstance(actuals[0], str):
        # Build actuals first when given tickers list
        df = compute_fundamentals_actuals(actuals, save_csv=False, as_df=True, basis=basis)
    else:
        df = pd.DataFrame(actuals).copy() if not isinstance(actuals, pd.DataFrame) else actuals.copy()

    # Back-fill raw metric columns from public-labelled columns (basis-aware)
    rmap = _make_rename_map(basis)  # raw_name -> public_label
    for raw, public in rmap.items():
        if raw in _RAW_METRICS and public in df.columns:
            if raw not in df.columns or df[raw].isna().all():
                df[raw] = df[public]

    # Ensure expected raw cols exist
    for col in _RAW_METRICS:
        if col not in df.columns:
            df[col] = np.nan

    # --- NEW (opt-in): replace TTM profitability/liquidity with annual averages for consistent horizon ---
    _timeseries_payload = {}  # will only be used if return_timeseries=True
    if use_timeseries_averages_for_profitability:
        if "ticker" not in df.columns:
            raise KeyError("compute_fundamentals_scores: 'ticker' column missing in actuals DataFrame.")

        avg_map = {}
        for t in df["ticker"]:
            try:
                prof_ts = compute_profitability_timeseries(t, include_ttm=False, return_format="wide")
                liq_ts  = compute_liquidity_timeseries(t, include_ttm=False, return_format="wide")
                avg_map[t] = {
                    "roe": float(prof_ts["ROE"].astype(float).mean(skipna=True)) if "ROE" in prof_ts.columns else np.nan,
                    "profit_margin": float(prof_ts["NetMargin"].astype(float).mean(skipna=True)) if "NetMargin" in prof_ts.columns else np.nan,
                    "op_margin": float(prof_ts["OpMargin"].astype(float).mean(skipna=True)) if "OpMargin" in prof_ts.columns else np.nan,
                    "roa": float(prof_ts["ROA"].astype(float).mean(skipna=True)) if "ROA" in prof_ts.columns else np.nan,
                    "current_ratio": float(liq_ts["CurrentRatio"].astype(float).mean(skipna=True)) if "CurrentRatio" in liq_ts.columns else np.nan,
                }
                if return_timeseries:
                    _timeseries_payload[t] = {
                        "profitability_timeseries": (
                            prof_ts.reset_index().rename(columns={"index":"Period"})
                            if prof_ts.index.name is not None else prof_ts.reset_index()
                        ),
                        "liquidity_timeseries": (
                            liq_ts.reset_index().rename(columns={"index":"Period"})
                            if liq_ts.index.name is not None else liq_ts.reset_index()
                        ),
                    }
            except Exception:
                avg_map[t] = {"roe": np.nan, "profit_margin": np.nan, "op_margin": np.nan, "roa": np.nan, "current_ratio": np.nan}

        # Override TTM/snapshot inputs with annual-average values before scoring
        for k_df, k_avg in [
            ("roe","roe"),
            ("profit_margin","profit_margin"),
            ("op_margin","op_margin"),
            ("roa","roa"),
            ("current_ratio","current_ratio"),
        ]:
            if k_df in df.columns:
                df[k_df] = df["ticker"].map(lambda tt: avg_map.get(tt, {}).get(k_avg, np.nan))

    # Per-metric scores
    for factor, mweights in _WEIGHTS.items():
        for metric in mweights:
            df[f"score_{metric}"] = df[metric].apply(lambda x: _score_metric(x, metric))

    # Factor rollups
    for factor, mweights in _WEIGHTS.items():
        cols = [f"score_{m}" for m in mweights]
        df[f"{factor}_score"] = df[cols].mul(list(mweights.values()), axis=1).sum(axis=1)

    df["total_score"] = df[["profitability_score","growth_score","reinvestment_score","risk_score"]].sum(axis=1)

    # Either return only the score table or merged
    _score_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("score_")]
    _scores_sum_cols = ["profitability_score", "growth_score", "reinvestment_score", "risk_score", "total_score"]
    scores_only_cols = _score_cols + [c for c in _scores_sum_cols if c in df.columns]

    out = df if merge_with_actuals else df[["ticker"] + scores_only_cols]
    # Apply public-facing labels to score_* columns (basis-aware)
    rename_map = _make_rename_map(basis)
    out = out.rename(columns={k: v for k, v in rename_map.items() if k.startswith("score_")})

    if as_df:
        if return_timeseries:
            return {"scores": out, "timeseries": _timeseries_payload}
        return out
    else:
        if return_timeseries:
            return {
                "scores": to_records(out, analysis_report_date=analysis_report_date),
                "timeseries": {
                    t: {
                        "profitability_timeseries": to_records(ts["profitability_timeseries"], analysis_report_date=analysis_report_date),
                        "liquidity_timeseries": to_records(ts["liquidity_timeseries"], analysis_report_date=analysis_report_date),
                    }
                    for t, ts in _timeseries_payload.items()
                },
            }
        return to_records(out, analysis_report_date=analysis_report_date)

def full_fundamentals_table(
    tickers: List[str],
    *,
    output_dir: str = "output",
    save_csv: bool = False,           # <--- NEW: default OFF
    as_df: bool = True,
    analysis_report_date: Optional[str] = None,
    include_scores_in_actuals: bool = True,
    basis: Literal["annual","ttm"] = "annual"
):
    """
    Convenience orchestrator: actuals → scores → recommendation.
    Returns a single table (actuals + scores + recommendation) by default.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    actuals = compute_fundamentals_actuals(tickers, output_dir=output_dir, save_csv=save_csv, as_df=True, analysis_report_date=analysis_report_date,basis=basis)
    scored = compute_fundamentals_scores(actuals, as_df=True, merge_with_actuals=True, analysis_report_date=analysis_report_date,basis=basis) if include_scores_in_actuals \
        else compute_fundamentals_scores(actuals, as_df=True, merge_with_actuals=False, analysis_report_date=analysis_report_date,basis=basis)

    df = scored.copy() if include_scores_in_actuals else actuals.merge(scored, on="ticker", how="left")

    # Recommendation, dispersion, data flag
    df["factor_dispersion"] = df[["profitability_score","growth_score","reinvestment_score","risk_score"]].std(axis=1)
    recs = df.apply(lambda r: _strategic_recommendation(r), axis=1)
    df[["recommendation","comments"]] = pd.DataFrame(recs.tolist(), index=df.index)

    # (only if save_csv):
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "fundamentals_summary.csv")
        df.to_csv(out_file, index=False)

    if as_df:
        return df
    return to_records(df, analysis_report_date=analysis_report_date)

def historical_average_share_prices(
    tickers: list[str] | str,
    analysis_report_date: str | None = None,
    save_csv: bool = False,
    as_df: bool = False,
    output_dir: str = "output",
):
    """
    Compute simple average share prices over 1/30/90/180 days for one or more tickers.

    Args:
        tickers: Ticker or list of tickers (Yahoo symbols).
        analysis_report_date: Optional label to embed in CSV filename (no effect on calc).
        save_csv: If True, writes a CSV under {output_dir}/AVG_PRICES/.
        as_df: If True, return a pandas DataFrame; else return list-of-dicts.
        output_dir: Root folder for artifacts if save_csv=True.

    Returns:
        DataFrame (if as_df=True) ELSE list[dict] with fields:
            - ticker
            - price_asof
            - avg_price_1d
            - avg_price_30d
            - avg_price_90d
            - avg_price_180d
            - notes
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    if isinstance(tickers, str):
        tickers = [tickers]

    rows: list[dict] = []
    for t in tickers:
        snap = _price_snapshots_ext(t)  # <-- dict with keys we need
        rows.append({
            "ticker": snap.get("ticker", t),
            "price_asof": snap.get("price_asof"),
            "avg_price_1d": snap.get("avg_price_1d"),
            "avg_price_30d": snap.get("avg_price_30d"),
            "avg_price_90d": snap.get("avg_price_90d"),
            "avg_price_180d": snap.get("avg_price_180d"),
            "notes": snap.get("notes", ""),
        })

    df = pd.DataFrame(rows)

    if save_csv:
        out_dir = Path(output_dir) / "AVG_PRICES"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = analysis_report_date or datetime.now().strftime("%Y%m%d")
        out_path = out_dir / f"avg_prices_{tag}.csv"
        df.to_csv(out_path, index=False)

    return df if as_df else df.to_dict(orient="records")

# Small alias resolver for Yahoo Finance statement rows
def _alias_first(df: pd.DataFrame, aliases: List[str]) -> Optional[pd.Series]:
    """
    Return the first matching row (as a Series) from df.index for any alias.
    Index match is exact, case-sensitive to align with yfinance keys.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for a in aliases:
        if a in df.index:
            return df.loc[a]
    return None

def _annual_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure annual statement has increasing columns by fiscal year (left→right)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    # yfinance usually returns most-recent on the left; enforce ascending by column name if they are dates/years
    try:
        # Columns are often Timestamps or datelike strings
        cols = list(df.columns)
        parsed = pd.to_datetime(cols, errors="coerce")
        order = np.argsort(parsed.values)
        return df.iloc[:, order]
    except Exception:
        return df

def _sum_last_4_quarters(s: pd.Series) -> Optional[float]:
    if s is None or s.empty:
        return None
    s = s.dropna().astype(float)
    if len(s) < 4:
        return None
    return float(s.iloc[:4].sum())  # yfinance quarterly has most-recent first

def _avg_pair(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (a + b) / 2.0

def compute_profitability_timeseries(
    ticker: str,
    *,
    include_ttm: bool = True,
    as_df: bool = True,
    return_format: str = "wide"
) -> pd.DataFrame:
    """
    Annual per-period metrics (+ optional TTM):
      - ROE        = Net Income / average(Equity_t, Equity_{t-1})
      - NetMargin  = Net Income / Total Revenue
      - OpMargin   = Operating Income / Total Revenue
      - ROA        = Net Income / average(Total Assets_t, Total Assets_{t-1})

    Returns:
      wide (default): index = [YYYY, ... , 'TTM' (optional)], columns = ROE, NetMargin, OpMargin, ROA, Notes
      long: columns = [Period, Metric, Value, Notes]
    """
    tkr = yf.Ticker(ticker)

    # Annual statements
    is_annual = _annual_sorted(tkr.financials)      # income statement
    bs_annual = _annual_sorted(tkr.balance_sheet)   # balance sheet

    # Map rows (aliases cover naming variance across tickers)
    s_ni = _alias_first(is_annual, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
    s_rev = _alias_first(is_annual, ["Total Revenue", "Revenue"])
    s_oi = _alias_first(is_annual, ["Operating Income", "Operating Income or Loss"])
    s_equity = _alias_first(bs_annual, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"])
    s_assets = _alias_first(bs_annual, ["Total Assets"])

    # Collect years (common set)
    cols = None
    for s in [s_ni, s_rev, s_oi, s_equity, s_assets]:
        if s is not None:
            cols = s.index if cols is None else cols.union(s.index)
    cols = list(cols) if cols is not None else []

    # Build annual table
    rows = []
    for i, c in enumerate(cols):
        def _val(series): return None if series is None or c not in series.index else (None if pd.isna(series[c]) else float(series[c]))
        ni = _val(s_ni); rev = _val(s_rev); oi = _val(s_oi); eq = _val(s_equity); assets = _val(s_assets)

        # Previous period values for averages
        c_prev = cols[i-1] if i > 0 else None
        eq_prev = None
        assets_prev = None
        if c_prev is not None:
            eq_prev = None if s_equity is None or c_prev not in s_equity.index else (None if pd.isna(s_equity[c_prev]) else float(s_equity[c_prev]))
            assets_prev = None if s_assets is None or c_prev not in s_assets.index else (None if pd.isna(s_assets[c_prev]) else float(s_assets[c_prev]))

        notes = []
        # Denominators
        avg_eq = _avg_pair(eq, eq_prev)
        avg_assets = _avg_pair(assets, assets_prev)

        def safe_div(num, den, den_name):
            if num is None or den is None:
                notes.append(f"{den_name} missing"); return np.nan
            if den == 0:
                notes.append(f"{den_name} = 0"); return np.nan
            return float(num) / float(den)

        roe = safe_div(ni, avg_eq, "avg_equity")
        roa = safe_div(ni, avg_assets, "avg_assets")
        net_margin = safe_div(ni, rev, "revenue")
        op_margin = safe_div(oi, rev, "revenue")

        rows.append({
            "Period": str(pd.to_datetime(c).year if not isinstance(c, str) else c),
            "ROE": roe,
            "NetMargin": net_margin,
            "OpMargin": op_margin,
            "ROA": roa,
            "Notes": "; ".join(sorted(set([n for n in notes if n])))
        })

    df_annual = pd.DataFrame(rows)

    # Optional TTM (from quarterly)
    if include_ttm:
        is_q = tkr.quarterly_financials
        bs_q = tkr.quarterly_balance_sheet
        s_ni_q = _alias_first(is_q, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
        s_rev_q = _alias_first(is_q, ["Total Revenue", "Revenue"])
        s_oi_q  = _alias_first(is_q, ["Operating Income", "Operating Income or Loss"])
        s_equity_q = _alias_first(bs_q, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"])
        s_assets_q = _alias_first(bs_q, ["Total Assets"])

        notes = []
        ttm_ni = _sum_last_4_quarters(s_ni_q) if s_ni_q is not None else None
        ttm_rev = _sum_last_4_quarters(s_rev_q) if s_rev_q is not None else None
        ttm_oi  = _sum_last_4_quarters(s_oi_q)  if s_oi_q  is not None else None
        latest_eq = None if s_equity_q is None or s_equity_q.empty else float(s_equity_q.iloc[0])
        latest_assets = None if s_assets_q is None or s_assets_q.empty else float(s_assets_q.iloc[0])

        def safe_div_ttm(num, den, den_name):
            if num is None or den is None:
                notes.append(f"{den_name} missing for TTM"); return np.nan
            if den == 0:
                notes.append(f"{den_name} = 0 for TTM"); return np.nan
            return float(num) / float(den)

        roe_ttm = safe_div_ttm(ttm_ni, latest_eq, "equity_latestQ")
        roa_ttm = safe_div_ttm(ttm_ni, latest_assets, "assets_latestQ")
        net_margin_ttm = safe_div_ttm(ttm_ni, ttm_rev, "revenue_TTM")
        op_margin_ttm  = safe_div_ttm(ttm_oi, ttm_rev, "revenue_TTM")

        df_annual = pd.concat([df_annual, pd.DataFrame([{
            "Period": "TTM",
            "ROE": roe_ttm,
            "NetMargin": net_margin_ttm,
            "OpMargin": op_margin_ttm,
            "ROA": roa_ttm,
            "Notes": "; ".join(sorted(set([n for n in notes if n])))
        }])], ignore_index=True)

    # Format
    if return_format == "long":
        out = df_annual.melt(id_vars=["Period", "Notes"], var_name="Metric", value_name="Value")
        df_final = out[["Period", "Metric", "Value", "Notes"]]
    else:
        df_final = df_annual.set_index("Period")
    
    # Final return: DataFrame vs JSON records
    if as_df:
        return df_final
    else:
        return to_records(df_final, analysis_report_date=_today_iso())
    
def compute_liquidity_timeseries(
    ticker: str,
    *,
    include_ttm: bool = True,
    as_df: bool = True,
    return_format: str = "wide"
) -> pd.DataFrame:
    """
    Annual per-period:
      - CurrentRatio = Current Assets / Current Liabilities
    'TTM' row is really 'LatestQ' from the most recent quarterly balance sheet.
    """
    tkr = yf.Ticker(ticker)
    bs_annual = _annual_sorted(tkr.balance_sheet)

    s_ca = _alias_first(bs_annual, ["Total Current Assets", "Current Assets"])
    s_cl = _alias_first(bs_annual, ["Total Current Liabilities", "Current Liabilities"])

    cols = None
    for s in [s_ca, s_cl]:
        if s is not None:
            cols = s.index if cols is None else cols.union(s.index)
    cols = list(cols) if cols is not None else []

    rows = []
    for c in cols:
        def _val(series): return None if series is None or c not in series.index else (None if pd.isna(series[c]) else float(series[c]))
        ca = _val(s_ca); cl = _val(s_cl)
        notes = []
        if ca is None: notes.append("current_assets missing")
        if cl is None: notes.append("current_liabilities missing")
        if cl == 0:    notes.append("current_liabilities = 0")

        cr = np.nan
        if ca is not None and cl not in (None, 0):
            cr = float(ca) / float(cl)

        rows.append({
            "Period": str(pd.to_datetime(c).year if not isinstance(c, str) else c),
            "CurrentRatio": cr,
            "Notes": "; ".join(sorted(set([n for n in notes if n])))
        })

    df_annual = pd.DataFrame(rows)

    if include_ttm:
        bs_q = tkr.quarterly_balance_sheet
        s_ca_q = _alias_first(bs_q, ["Total Current Assets", "Current Assets"])
        s_cl_q = _alias_first(bs_q, ["Total Current Liabilities", "Current Liabilities"])

        ca_q = None if s_ca_q is None or s_ca_q.empty else float(s_ca_q.iloc[0])
        cl_q = None if s_cl_q is None or s_cl_q.empty else float(s_cl_q.iloc[0])
        notes = []
        if ca_q is None: notes.append("current_assets latestQ missing")
        if cl_q is None: notes.append("current_liabilities latestQ missing")
        if cl_q == 0:    notes.append("current_liabilities latestQ = 0")

        cr_q = np.nan
        if ca_q is not None and cl_q not in (None, 0):
            cr_q = float(ca_q) / float(cl_q)

        df_annual = pd.concat([df_annual, pd.DataFrame([{
            "Period": "LatestQ",
            "CurrentRatio": cr_q,
            "Notes": "; ".join(sorted(set([n for n in notes if n])))
        }])], ignore_index=True)

    if return_format == "long":
        df_final = (
            df_annual.rename(columns={"CurrentRatio": "Value"})[["Period", "Value", "Notes"]]
            .assign(Metric="CurrentRatio")[["Period","Metric","Value","Notes"]]
        )
    else:
        df_final = df_annual.set_index("Period")

    # Final return: DataFrame vs JSON records
    if as_df:
        return df_final
    else:
        return to_records(df_final, analysis_report_date=_today_iso())

# Helper regex function that picks up the averages regardless of exact spelling (hyphens/underscores/case).
def _pick(row: dict, patterns: list[str]):
    """
    Return the first numeric value in 'row' where the key matches any regex
    in 'patterns' (case-insensitive). Returns None if nothing matches.
    """
    if not isinstance(row, dict):
        return None
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for k, v in row.items():
            if isinstance(k, str) and rx.search(k):
                try:
                    return float(v)
                except Exception:
                    pass
    return None

def _coverage_from_series(s: pd.Series) -> str | None:
    if s is None or s.empty:
        return None
    yrs = [int(pd.to_datetime(c).year) for c in s.index if pd.notna(s[c])]
    if len(yrs) < 2:
        return None
    return f"{min(yrs)}–{max(yrs)} ({len(sorted(set(yrs)))}y)"

def fundamentals_ttm_vs_average(
    ticker: str,
    *,
    include_ttm: bool = True,
    as_df: bool = True,
    return_format: str = "wide"
) -> pd.DataFrame:
    """
    Return a tidy table comparing 'Average' (annual multi-year) vs 'TTM' (or LatestQ-derived) values.

    Notes:
    - 'Average_Value' uses the full available annual history (>=3 points for CAGR; >=1 for others), labeled with the actual Period window (e.g., '2020–2024').
    - 'TTM_Value' shows trailing-twelve-month (or latest-Q-derived) point estimate when supported (PEG, Beta are always TTM; CAGRs are Average only).
    - Units are decimals (0.25 = 25%).
    """
    # Profitability & Liquidity time series
    prof_ts = compute_profitability_timeseries(ticker, include_ttm=include_ttm, return_format="wide")
    liq_ts  = compute_liquidity_timeseries(ticker, include_ttm=include_ttm, return_format="wide")

    # Annual rows only for averages (exclude TTM/LatestQ)
    annual_prof = prof_ts.loc[[i for i in prof_ts.index if str(i).isdigit()]] if isinstance(prof_ts.index, pd.Index) else prof_ts
    annual_liq  = liq_ts.loc[[i for i in liq_ts.index if str(i).isdigit()]] if isinstance(liq_ts.index, pd.Index) else liq_ts

    # Averages across available annual periods
    prof_avg = annual_prof[["ROE","NetMargin","OpMargin","ROA"]].astype(float).mean(skipna=True)
    liq_avg  = annual_liq[["CurrentRatio"]].astype(float).mean(skipna=True)

    # TTM/snapshot values
    prof_ttm_row = prof_ts.loc["TTM"] if "TTM" in prof_ts.index else (prof_ts.iloc[[-1]] if len(prof_ts)>0 else pd.DataFrame())
    liq_ttm_row  = liq_ts.loc["LatestQ"] if "LatestQ" in liq_ts.index else (liq_ts.iloc[[-1]] if len(liq_ts)>0 else pd.DataFrame())

    def _safe_get(sr, k): 
        try: return float(sr[k])
        except: return None

    roe_ttm = _safe_get(prof_ttm_row.squeeze(), "ROE") if isinstance(prof_ttm_row, pd.Series) else (None if prof_ttm_row.empty else float(prof_ttm_row["ROE"].iloc[0]))
    nm_ttm  = _safe_get(prof_ttm_row.squeeze(), "NetMargin") if isinstance(prof_ttm_row, pd.Series) else (None if prof_ttm_row.empty else float(prof_ttm_row["NetMargin"].iloc[0]))
    om_ttm  = _safe_get(prof_ttm_row.squeeze(), "OpMargin")  if isinstance(prof_ttm_row, pd.Series) else (None if prof_ttm_row.empty else float(prof_ttm_row["OpMargin"].iloc[0]))
    roa_ttm = _safe_get(prof_ttm_row.squeeze(), "ROA")       if isinstance(prof_ttm_row, pd.Series) else (None if prof_ttm_row.empty else float(prof_ttm_row["ROA"].iloc[0]))
    cr_ttm  = _safe_get(liq_ttm_row.squeeze(), "CurrentRatio") if isinstance(liq_ttm_row, pd.Series) else (None if liq_ttm_row.empty else float(liq_ttm_row["CurrentRatio"].iloc[0]))

    # Pull TTM-only items (PEG, Beta) and historical-avg items you already compute in your fundamentals
    info = yf.Ticker(ticker).info or {}
    peg_ttm  = info.get("trailingPegRatio", None)
    beta_ttm = info.get("beta", None)

    # Try to reuse your already-computed averages from compute_fundamentals_actuals
    growth_rev_cagr = growth_earn_cagr = reinvest_rate_ave = capex_ratio_ave = de_ratio_ave = None

    try:
        f_actuals = compute_fundamentals_actuals([ticker], save_csv=False, as_df=True)
    except Exception:
        f_actuals = None

    if isinstance(f_actuals, pd.DataFrame) and not f_actuals.empty:
        row = f_actuals.iloc[0].to_dict()
    else:
        row = {}

    # tolerant lookups (handles different column spellings)
    growth_rev_cagr   = _pick(row, [r"growth.*revenue.*cagr.*(ave|avg)", r"\brevenue.*cagr.*(ave|avg)\b",
        r"\brevenue.*cagr\b", r"^revenue_cagr$"])
    growth_earn_cagr  = _pick(row, [r"growth.*earn.*cagr.*(ave|avg)",    r"\bearn.*cagr.*(ave|avg)\b",
        r"\bearn.*cagr\b", r"^earnings_cagr$"])
    reinvest_rate_ave = _pick(row, [r"reinvest.*rate.*(ave|avg)", r"\breinvestment.*rate.*(ave|avg)\b",
        r"\breinvest.*rate\b", r"^reinvestment_rate$"])
    capex_ratio_ave   = _pick(row, [r"capex.*ratio.*(ave|avg)", r"\bcapital.*expend.*ratio.*(ave|avg)\b",
        r"\bcapex.*ratio\b", r"^capex_ratio$"])
    de_ratio_ave      = _pick(row, [r"debt.*equity.*ratio.*(ave|avg)", r"\bD\/?E.*(ave|avg)\b",
        r"debt.*equity.*ratio", r"^de_ratio$"])
    
    # --- NEW: compute TTM for Reinvestment Rate, D/E, and Capex Ratio from quarterlies ---
    tkr = yf.Ticker(ticker)
    is_q = tkr.quarterly_financials
    cf_q = tkr.quarterly_cashflow
    bs_q = tkr.quarterly_balance_sheet

    # Alias picks
    s_ni_q    = _alias_first(is_q, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
    s_rev_q   = _alias_first(is_q, ["Total Revenue", "Revenue"])
    s_div_q   = _alias_first(cf_q, ["Common Stock Dividend Paid"])
    s_capex_q = _alias_first(cf_q, ["Capital Expenditure", "Capital Expenditures"])
    s_debt_q  = _alias_first(bs_q, ["Total Debt", "Total Debt Net", "Long Term Debt", "Short Long Term Debt"])
    s_equ_q   = _alias_first(bs_q, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"])

    # Sum last 4 quarters (TTM flows)
    ni_ttm    = _sum_last_4_quarters(s_ni_q) if s_ni_q is not None else None
    rev_ttm   = _sum_last_4_quarters(s_rev_q) if s_rev_q is not None else None
    div_ttm   = _sum_last_4_quarters(s_div_q) if s_div_q is not None else None
    capex_ttm = _sum_last_4_quarters(s_capex_q) if s_capex_q is not None else None

    # Latest quarter (point-in-time stocks)
    debt_latest   = None if s_debt_q is None or s_debt_q.empty else float(s_debt_q.iloc[0])
    equity_latest = None if s_equ_q  is None or s_equ_q.empty  else float(s_equ_q.iloc[0])

    # Reinvestment Rate (TTM):
    # dividends are usually negative cash flows; retained = NI + Dividends
    reinvestment_rate_ttm = None
    if ni_ttm not in (None, 0) and div_ttm is not None:
        reinvestment_rate_ttm = (float(ni_ttm) + float(div_ttm)) / float(ni_ttm)

    # D/E (TTM): point-in-time ratio from latest quarterly balance sheet
    de_ratio_ttm = None
    if equity_latest not in (None, 0) and debt_latest is not None:
        de_ratio_ttm = float(debt_latest) / float(equity_latest)

    # Capex Ratio (TTM): abs(CapEx_TTM) / Revenue_TTM
    capex_ratio_ttm = None
    if capex_ttm is not None and rev_ttm not in (None, 0):
        capex_ratio_ttm = abs(float(capex_ttm)) / float(rev_ttm)

    # Annual income statements for CAGR coverage
    is_annual = _annual_sorted(yf.Ticker(ticker).financials)
    s_rev = _alias_first(is_annual, ["Total Revenue","Revenue"])
    s_ni  = _alias_first(is_annual, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])

    rev_window = _coverage_from_series(s_rev)
    ni_window  = _coverage_from_series(s_ni)

    # Balance sheet for leverage coverage
    bs_annual = _annual_sorted(yf.Ticker(ticker).balance_sheet)
    s_debt = _alias_first(bs_annual, ["Total Debt","Long Term Debt","Short Long Term Debt"])
    s_equ  = _alias_first(bs_annual, ["Total Stockholder Equity","Common Stock Equity"])
    de_window = _coverage_from_series(s_debt)  # or overlap with equity if you want

    rows = [
        {"Metric": "Profitability-ROE",          "TTM_Value": roe_ttm, "Average_Value": float(prof_avg.get("ROE")) if "ROE" in prof_avg else None, "Window": f"{annual_prof.index.min()}–{annual_prof.index.max()}" if len(annual_prof)>0 else None, "Notes": ""},
        {"Metric": "Profitability-NetMargin",    "TTM_Value": nm_ttm,  "Average_Value": float(prof_avg.get("NetMargin")) if "NetMargin" in prof_avg else None, "Window": f"{annual_prof.index.min()}–{annual_prof.index.max()}" if len(annual_prof)>0 else None, "Notes": ""},
        {"Metric": "Profitability-OpMargin",     "TTM_Value": om_ttm,  "Average_Value": float(prof_avg.get("OpMargin")) if "OpMargin" in prof_avg else None, "Window": f"{annual_prof.index.min()}–{annual_prof.index.max()}" if len(annual_prof)>0 else None, "Notes": ""},
        {"Metric": "Profitability-ROA",          "TTM_Value": roa_ttm, "Average_Value": float(prof_avg.get("ROA")) if "ROA" in prof_avg else None, "Window": f"{annual_prof.index.min()}–{annual_prof.index.max()}" if len(annual_prof)>0 else None, "Notes": ""},
        {"Metric": "Growth-RevenueCAGR",     "TTM_Value": None,    "Average_Value": growth_rev_cagr,    "Window": rev_window, "Notes": ""},
        {"Metric": "Growth-EarningsCAGR",    "TTM_Value": None,    "Average_Value": growth_earn_cagr,   "Window": ni_window, "Notes": ""},
        {"Metric": "Growth-PEG-TTM",             "TTM_Value": float(peg_ttm) if peg_ttm is not None else None,  "Average_Value": None, "Window": "TTM", "Notes": ""},
        {"Metric": "Reinvestment-ReinvestmentRate",  "TTM_Value": reinvestment_rate_ttm, "Average_Value": reinvest_rate_ave, "Window": rev_window, "Notes": ""},
        {"Metric": "Reinvestment-CapexRatio",        "TTM_Value": capex_ratio_ttm,      "Average_Value": capex_ratio_ave,   "Window": rev_window, "Notes": ""},
        {"Metric": "Risk-Beta-TTM",                      "TTM_Value": float(beta_ttm) if beta_ttm is not None else None,"Average_Value": None, "Window": "TTM", "Notes": ""},
        {"Metric": "Risk-CurrentRatio",                  "TTM_Value": cr_ttm,  "Average_Value": float(liq_avg.get("CurrentRatio")) if "CurrentRatio" in liq_avg else None, "Window": f"{annual_liq.index.min()}–{annual_liq.index.max()}" if len(annual_liq)>0 else None, "Notes": ""},
        {"Metric": "Risk-DebtEquityRatio",           "TTM_Value": de_ratio_ttm,         "Average_Value": de_ratio_ave,      "Window": de_window, "Notes": ""},
    ]

    out = pd.DataFrame(rows)

    # return_format / as_df gate
    if return_format == "long":
        df_final = out[["Metric","TTM_Value","Average_Value","Window","Notes"]]
    else:
        df_final = (
            out.pivot_table(index="Metric",
                            values=["TTM_Value","Average_Value","Window","Notes"],
                            aggfunc="first")
            .reset_index()
        )

    if as_df:
        return df_final
    else:
        return to_records(df_final, analysis_report_date=_today_iso())

# ---- Backward-compatible facade (keeps your current server.py working) ----
def evaluate_fundamentals(tickers: List[str], output_dir: str = "output") -> str:
    """
    Returns JSON string (records) to preserve server.py behavior.
    """
    df = full_fundamentals_table(tickers, output_dir=output_dir, as_df=True, include_scores_in_actuals=True,basis="annual")
    # Important: the old facade renamed columns before CSV write; we matched that above.
    # Convert to JSON records to match prior return type
    return df.to_json(orient="records")

# ==========================================================
# VALUATION (peer multiples, DCF scenarios, Reverse-DCF)
# ==========================================================

def _data_health_init(target: str, peers: List[str]) -> Dict[str, Any]:
    return {
        "target": target,
        "inputs": {},
        "peer_set": {"tickers_provided": list(peers), "tickers_used": [], "excluded": []},
        "method_status": {
            "DCF": {"status": "unknown", "reasons": [], "valuations": {}},
            "PE": {"status": "unknown", "reasons": [], "scenarios_present": [], "missing": []},
            "PS": {"status": "unknown", "reasons": [], "scenarios_present": [], "missing": []},
            "EV_EBITDA": {"status": "unknown", "reasons": [], "scenarios_present": [], "missing": []},
        },
        "notes_for_llm": ""
    }

# =========================
# HEALTH REPORT RENDERERS
# =========================

def health_to_tables(
    health_report: Any,
    *,
    sort: bool = True,
    as_df: bool = True
):
    """
    Normalize a health report into a tidy table.

    Supports:
      1) New schema (recommended):
         [
           {"source": "compute_fundamentals_actuals", "ticker": "MSFT", "data_incomplete": False, "notes": ["..."]},
           {"source": "compare_to_market_cap",        "ticker": None,  "data_incomplete": None,  "notes": ["..."]},
           ...
         ]

      2) Legacy schema (older orchestrator style):
         {
           "inputs": {...},
           "peer_set": {"tickers_used": [...], "tickers_provided": [...]},
           "method_status": {"PE": {"status": "ok"}, ...},
           "notes_for_llm": "long concatenated string of notes ..."
         }

    Returns:
      - pandas.DataFrame by default (as_df=True)
      - list[dict] if as_df=False
    """
    rows: List[Dict[str, Any]] = []

    # -------- Case 1: New schema (list of blocks) --------
    if isinstance(health_report, list):
        for blk in health_report:
            if not isinstance(blk, dict):
                continue
            source = blk.get("source") or "(unknown)"
            ticker = blk.get("ticker")
            di     = blk.get("data_incomplete", None)

            notes_list = []
            if isinstance(blk.get("notes"), (list, tuple)):
                notes_list = [str(x).strip() for x in blk["notes"] if str(x).strip()]
            elif isinstance(blk.get("notes"), str):
                notes_list = [blk["notes"].strip()] if blk["notes"].strip() else []

            # Deduplicate within the block
            notes_str = " | ".join(sorted(set(notes_list)))

            rows.append({
                "Source": source,
                "Ticker": ticker,
                "Data Incomplete": di,
                "Notes": notes_str
            })

    # -------- Case 2: Legacy dict schema --------
    elif isinstance(health_report, dict):
        # (a) Summarized notes
        nsum = str(health_report.get("notes_for_llm", "")).strip()
        if nsum:
            rows.append({
                "Source": "orchestrator_summary",
                "Ticker": None,
                "Data Incomplete": None,
                "Notes": nsum
            })

        # (b) Method status
        mstatus = health_report.get("method_status", {})
        if isinstance(mstatus, dict):
            for mname, mval in mstatus.items():
                st = None
                if isinstance(mval, dict):
                    st = mval.get("status")
                elif isinstance(mval, str):
                    st = mval
                if st:
                    rows.append({
                        "Source": f"method:{mname}",
                        "Ticker": None,
                        "Data Incomplete": None,
                        "Notes": f"status={st}"
                    })

        # (c) Peer set echo (optional display)
        pset = health_report.get("peer_set", {})
        if isinstance(pset, dict):
            used = pset.get("tickers_used")
            prov = pset.get("tickers_provided")
            if used:
                rows.append({
                    "Source": "peer_set",
                    "Ticker": None,
                    "Data Incomplete": None,
                    "Notes": f"tickers_used={used}"
                })
            if prov:
                rows.append({
                    "Source": "peer_set",
                    "Ticker": None,
                    "Data Incomplete": None,
                    "Notes": f"tickers_provided={prov}"
                })

    # -------- Fallback: nothing parsable --------
    if not rows:
        rows = [{
            "Source": "(unknown)",
            "Ticker": None,
            "Data Incomplete": None,
            "Notes": ""
        }]

    # Build DataFrame
    df = pd.DataFrame(rows, columns=["Source", "Ticker", "Data Incomplete", "Notes"])

    # Drop perfect duplicates
    df = df.drop_duplicates()

    # Optional sort: by Source, then Ticker (None last)
    if sort:
        df["Ticker_sort"] = df["Ticker"].fillna("~")  # tilde sorts after letters
        df = df.sort_values(["Source", "Ticker_sort"]).drop(columns=["Ticker_sort"]).reset_index(drop=True)

    return df if as_df else df.to_dict(orient="records")


def health_to_markdown(health_report) -> str:
    """
    Render a health report (new list-of-blocks OR legacy dict) as Markdown.
    Uses `health_to_tables` under the hood.
    """
    df = health_to_tables(health_report, as_df=True)

    if df.empty:
        return "_No health messages._"

    # Build a compact Markdown table
    lines = []
    lines.append("| Source | Ticker | Data Incomplete | Notes |")
    lines.append("|---|---|---:|---|")
    for _, r in df.iterrows():
        src   = str(r.get("Source", "") or "")
        tkr   = str(r.get("Ticker", "") or "")
        di    = r.get("Data Incomplete", None)
        di_s  = "" if pd.isna(di) else ("True" if bool(di) else "False")
        notes = str(r.get("Notes", "") or "")
        lines.append(f"| {src} | {tkr} | {di_s} | {notes} |")

    return "\n".join(lines)

def save_health_report_excel(
    health_report,
    path: str = "output/health_report.xlsx",
    *,
    include_raw: bool = True
) -> str:
    """
    Save a health report (new list-of-blocks OR legacy dict) to an Excel file.
    - Sheet 'Health Report' -> normalized table from `health_to_tables`
    - Optional sheet 'Raw'   -> raw JSON (for audit/debug), if include_raw=True
    Returns the path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Normalize to a tidy DataFrame
    df = health_to_tables(health_report, as_df=True)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Health Report", index=False)

        if include_raw:
            # Write raw data as single-cell JSON for auditability
            raw_json = json.dumps(health_report, default=str, ensure_ascii=False, indent=2)
            raw_df = pd.DataFrame({"raw_json": [raw_json]})
            raw_df.to_excel(writer, sheet_name="Raw", index=False)

    return path

# --- new internal, extended helper used by other functions ---
def _price_snapshots_ext(ticker: str) -> dict:
    t = _sanitize_ticker(ticker)
    notes = []
    try:
        yt = yf.Ticker(t)
        hist = yt.history(period="200d", auto_adjust=True, actions=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return {"ticker": t, "avg_price_1d": None, "avg_price_30d": None,
                    "avg_price_90d": None, "avg_price_180d": None,
                    "price_asof": None, "notes": "No price history or 'Close' missing in last ~200d."}
        px = hist["Close"].dropna()
        if px.empty:
            return {"ticker": t, "avg_price_1d": None, "avg_price_30d": None,
                    "avg_price_90d": None, "avg_price_180d": None,
                    "price_asof": None, "notes": "'Close' series empty after dropna."}

        def _avg(days:int):
            cutoff = px.index.max() - pd.Timedelta(days=days-1)
            w = px.loc[px.index >= cutoff]
            return float(w.mean()) if not w.empty else None

        a1, a30, a90, a180 = _avg(1), _avg(30), _avg(90), _avg(180)
        for d, v in [(1,a1),(30,a30),(90,a90),(180,a180)]:
            if v is None: notes.append(f"No data to compute {d}d average.")
        return {"ticker": t, "avg_price_1d": a1, "avg_price_30d": a30,
                "avg_price_90d": a90, "avg_price_180d": a180,
                "price_asof": str(px.index.max().date()),
                "notes": " ".join(notes).strip()}
    except Exception as e:
        return {"ticker": t, "avg_price_1d": None, "avg_price_30d": None,
                "avg_price_90d": None, "avg_price_180d": None,
                "price_asof": None, "notes": f"Error fetching prices: {str(e)[:120]}"}

#legacy code retained to avoid "breaking" other functions

def _price_snapshots(ticker: str):
    """
    Returns a 3-tuple: (avg_price_1d, price_asof, notes)

    NOTE: A new internal helper `_price_snapshots_ext` returns a dict with 1/30/90/180d.
    """
    d = _price_snapshots_ext(ticker)  # use the new extended helper under the hood
    return d["avg_price_1d"], d["price_asof"], d["notes"]

def _peer_usability_reasons(row: pd.Series) -> List[str]:
    r = []
    if not _is_num(row.get("earnings")) or (row.get("earnings") is not None and row.get("earnings") <= 0):
        r.append("earnings<=0 or missing")
    if not _is_pos(row.get("shares_outstanding")):
        r.append("shares_outstanding<=0 or missing")
    if not _is_num(row.get("pe_ratio")):
        r.append("pe_ratio missing")
    if not _is_num(row.get("revenue")) or (row.get("revenue") is not None and row.get("revenue") <= 0):
        r.append("revenue<=0 or missing")
    if not _is_num(row.get("ps_ratio")):
        r.append("ps_ratio missing")
    if not _is_num(row.get("ebitda")) or (row.get("ebitda") is not None and row.get("ebitda") <= 0):
        r.append("ebitda<=0 or missing")
    if not _is_num(row.get("ev_to_ebitda")):
        r.append("ev_to_ebitda missing")
    return r

def _pull_company_snapshot(ticker: str) -> Dict[str, Any]:
    s = yf.Ticker(ticker)
    info = s.info
    return {
        'ticker': ticker,
        'revenue_series': s.financials.loc['Total Revenue'].dropna() if 'Total Revenue' in s.financials.index else pd.Series(dtype=float),
        'beta': info.get('beta', None),
        'shares_outstanding': info.get('sharesOutstanding', None),
        'cashflow': s.cashflow,
        'market_cap': info.get('marketCap'),
        'ebitda': info.get('ebitda'),
        'revenue': info.get('totalRevenue'),
        'earnings': info.get('netIncomeToCommon'),
        'pe_ratio': info.get('trailingPE'),
        'ps_ratio': info.get('priceToSalesTrailing12Months'),
        'ev_to_ebitda': info.get('enterpriseToEbitda'),
        'current_price': info.get('currentPrice'),
        'free_cash_flow': info.get('freeCashflow')
    }

def peer_multiples(
    tickers: List[str],
    *,
    target_ticker: Optional[str] = None,
    include_target: bool = False,
    as_df: bool = True,
    analysis_report_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build peer multiples and valuation bands.

    Behavior (requested/spec’d):
    - target_ticker is REQUIRED. If missing, raise ValueError explaining to supply
      target_ticker and choose include_target True/False.
    - peer_comp_detail: per-ticker detail for ALL tickers you pass in (including
      target if it’s in the list).
    - peer_multiple_bands_wide & peer_comp_bands: computed off the 'peer-only' set.
      If include_target=False (default), the target is EXCLUDED from those stats.
      If include_target=True and target is not inside `tickers`, we append it to
      the peer set for stats.

    Returns
    -------
    {
      "peer_comp_detail": DataFrame([... all tickers incl. target if present ...]),
      "peer_multiple_bands_wide": DataFrame(index=[PE,PS,EV_EBITDA], columns=[Min,P25,Median,P75,Max,Average]),
      "peer_comp_bands": DataFrame([{"Scenario":"PE_P25","Valuation_per_Share":...}, ...]),
      # (If you previously returned other keys, keep them unchanged here)
    }
    """

    if not tickers or len(tickers) < 2:
        raise ValueError("peer_multiples: provide at least two tickers for a meaningful peer set.")

    # ---- enforce target_ticker presence ----
    if not target_ticker or not isinstance(target_ticker, str) or not target_ticker.strip():
        raise ValueError(
            "peer_multiples requires target_ticker. "
            "Call like: peer_multiples(tickers, target_ticker='MSFT', include_target=False). "
            "Choose whether to include the target in peer stats via include_target=True/False."
        )
    target_tkr = target_ticker.strip().upper()

    # Normalize list of tickers
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]

    # --- Build the 'detail' universe exactly as the user passed it (do not remove the target) ---
    # (This preserves the behavior: target appears in peer_comp_detail if present in the list.)
    detail_snaps = []
    for t in tickers:
        try:
            detail_snaps.append(_pull_company_snapshot(_sanitize_ticker(t)))
        except Exception:
            continue
    peer_comp_detail = pd.DataFrame(detail_snaps)

    # If detail is empty, return empty frames in the correct shape
    if peer_comp_detail.empty:
        empty_bands = pd.DataFrame(columns=["Min","P25","Median","P75","Max","Average"])
        empty_long = pd.DataFrame(columns=["Scenario", "Valuation_per_Share"])
        out = {
            "peer_comp_detail": peer_comp_detail,
            "peer_multiple_bands_wide": empty_bands,
            "peer_comp_bands": empty_long,
        }
        return out if as_df else out  # your to_records wrapper if you use one

    # --- Build the 'peer-only' Calc DF for stats/bands ---
    peer_only = peer_comp_detail.copy()

    if include_target:
        # Ensure target participates in stats — if it's not in tickers, try to append its snapshot
        if target_tkr not in set(peer_only["ticker"].astype(str).str.upper()):
            try:
                snap = _pull_company_snapshot(_sanitize_ticker(target_tkr))
                peer_only = pd.concat([peer_only, pd.DataFrame([snap])], ignore_index=True)
            except Exception:
                warnings.warn(f"include_target=True but could not fetch snapshot for {target_tkr}; proceeding without it.", UserWarning)
    else:
        # Exclude target from stats if present
        peer_only = peer_only[peer_only["ticker"].astype(str).str.upper() != target_tkr]

    # --- (1) Ratio bands (wide: PE/PS/EV_EBITDA x Min/P25/Median/P75/Max/Average) ---
    def _bands(series: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {}
        return {
            "Min": float(s.min()),
            "P25": float(np.percentile(s, 25)),
            "Median": float(np.percentile(s, 50)),
            "P75": float(np.percentile(s, 75)),
            "Max": float(s.max()),
            "Average": float(s.mean()),
        }

    vals_by_ticker = peer_only[["ticker","pe_ratio","ps_ratio","ev_to_ebitda"]].rename(
        columns={"pe_ratio":"PE","ps_ratio":"PS","ev_to_ebitda":"EV_EBITDA"}
    )

    rows = []
    for metric in ["PE","PS","EV_EBITDA"]:
        b = _bands(vals_by_ticker[metric])
        if b:
            rows.append(pd.Series(b, name=metric))
    peer_multiple_bands_wide = pd.DataFrame(rows, columns=["Min","P25","Median","P75","Max","Average"])

    # --- (2) Per-share valuation bands (long) built from peer-only as in your Rev35 logic ---
    # Helper converting multiples to per-share valuations; reuse your existing field mapping
    def _val_per_share_from_multiple(row, metric: str):
        if metric == "PE":
            eps = row.get("eps_ttm")
            mult = row.get("pe_ratio")
            return eps * mult if _is_num(eps) and _is_num(mult) else None
        if metric == "PS":
            rps = row.get("revenue_per_share_ttm")
            mult = row.get("ps_ratio")
            return rps * mult if _is_num(rps) and _is_num(mult) else None
        if metric == "EV_EBITDA":
            # If you have a direct per-share valuation field already computed in your code, use it here
            return row.get("EV_EBITDA_Valuation_per_share")
        return None

    bands_records = []
    for metric in ["PE","PS","EV_EBITDA"]:
        s = peer_only.apply(lambda r: _val_per_share_from_multiple(r, metric), axis=1)
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            continue
        bands_records.extend([
            {"Scenario": f"{metric}_Min",    "Valuation_per_Share": float(s.min())},
            {"Scenario": f"{metric}_P25",    "Valuation_per_Share": float(np.percentile(s, 25))},
            {"Scenario": f"{metric}_Median", "Valuation_per_Share": float(np.percentile(s, 50))},
            {"Scenario": f"{metric}_P75",    "Valuation_per_Share": float(np.percentile(s, 75))},
            {"Scenario": f"{metric}_Max",    "Valuation_per_Share": float(s.max())},
            {"Scenario": f"{metric}_Avg",    "Valuation_per_Share": float(s.mean())},
        ])
    peer_comp_bands = pd.DataFrame(bands_records)

    out = {
        "target_ticker": target_tkr,
        "peer_comp_detail": peer_comp_detail,           # ALL tickers as passed (incl. target if present)
        "peer_multiple_bands_wide": peer_multiple_bands_wide,  # peers only (excl target unless include_target=True)
        "peer_comp_bands": peer_comp_bands,             # peers only (per-share valuation bands)
    }
    return out if as_df else out  # (or your to_records wrapper)

def price_from_peer_multiples(
    peer_multiples_output: Dict[str, Any],
    *,
    ticker: Optional[str] = None,
    analysis_report_date: Optional[str] = None,
    save_csv: bool = False,
    as_df: bool = True,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Compute implied per-share price bands (P25/P50/P75) from peer_multiples(...) output.
    You can pass the dict directly. 'ticker' is optional; inferred when possible.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    notes = []

    # --- infer ticker --------------------------------------------------------
    t = _sanitize_ticker(ticker) if ticker else None
    if t is None:
        # preferred: explicit key carried by your peer_multiples()
        t = _sanitize_ticker(peer_multiples_output.get("target_ticker")) if peer_multiples_output.get("target_ticker") else None
    if t is None:
        # fallback: if peer_comp_detail includes the target row, pick the one present in bands or with missing peers flag
        pcd = peer_multiples_output.get("peer_comp_detail")
        if isinstance(pcd, dict): pcd = pd.DataFrame(pcd)
        if isinstance(pcd, pd.DataFrame) and not pcd.empty:
            # try to find the only/non-peer row; else pick first index
            idx = list(pcd.index)
            if len(idx) >= 1:
                t = _sanitize_ticker(str(idx[0]))
    if not t:
        raise ValueError("Could not infer target ticker; pass ticker=... or include 'target_ticker' in peer_multiples_output.")

    # --- bands table ---------------------------------------------------------
    bands_df = peer_multiples_output.get("peer_multiple_bands_wide")
    if isinstance(bands_df, dict):
        bands_df = pd.DataFrame(bands_df)
    if not isinstance(bands_df, pd.DataFrame) or bands_df.empty:
        raise ValueError("peer_multiples_output['peer_multiple_bands_wide'] is missing or empty.")

    bdf = bands_df.copy()
    bdf.index = [str(i).upper().replace("/", "_") for i in bdf.index]
    bdf.columns = [str(c).upper() for c in bdf.columns]

    def _pick(row, labels):
        # labels is a tuple of acceptable aliases
        for lab in labels:
            for c in bdf.columns:
                if lab in str(c).upper():
                    val = row.get(c)
                    if _is_num(val): return float(val)
        return None

    # --- fundamentals for per-share conversions ------------------------------
    def _pull_fundamentals_for_price_bands(t: str):
        """
        Returns dict with SharesOutstanding, Revenue, NetIncome, EBITDA,
        TotalDebt, CashAndCashEquivalents, MinorityInterest, NetDebt
        using yfinance first, then library fallbacks.
        """
        notes = []
        # 1) Try yfinance (as before) ...
        try:
            yt = yf.Ticker(t); info = yt.info or {}
            shares = _safe_float(info.get("sharesOutstanding"))
            revenue = net_income = ebitda = total_debt = cash_eq = mi = None

            fin = getattr(yt,"financials",None); qfin = getattr(yt,"quarterly_financials",None)
            def _latest_or_roll4(df, field, qdf=None):
                v=None
                if isinstance(df,pd.DataFrame) and field in df.index and not df.loc[field].dropna().empty:
                    v=float(df.loc[field].dropna().iloc[0])
                if not _is_num(v) and isinstance(qdf,pd.DataFrame) and field in qdf.index:
                    s=qdf.loc[field].dropna().iloc[:4]; v=float(s.sum()) if len(s)>0 else None
                return v
            revenue    = _latest_or_roll4(fin,"Total Revenue", qfin)
            net_income = _latest_or_roll4(fin,"Net Income", qfin)
            ebitda     = _safe_float(info.get("ebitda"))
            if not _is_num(ebitda) and isinstance(fin,pd.DataFrame) and "Ebitda" in fin.index and not fin.loc["Ebitda"].dropna().empty:
                ebitda = float(fin.loc["Ebitda"].dropna().iloc[0])

            total_debt = _safe_float(info.get("totalDebt"))
            cash_eq    = _safe_float(info.get("totalCash"))
            mi         = _safe_float(info.get("minorityInterest"))
            bs = getattr(yt,"balance_sheet",None)
            if isinstance(bs,pd.DataFrame) and not bs.empty:
                if not _is_num(total_debt):
                    for fld in ["Total Debt","Short Long Term Debt","Long Term Debt"]:
                        if fld in bs.index and not bs.loc[fld].dropna().empty:
                            total_debt=float(bs.loc[fld].dropna().iloc[0]); break
                if not _is_num(cash_eq):
                    for fld in ["Cash And Cash Equivalents","Cash"]:
                        if fld in bs.index and not bs.loc[fld].dropna().empty:
                            cash_eq=float(bs.loc[fld].dropna().iloc[0]); break
                if not _is_num(mi) and "Minority Interest" in bs.index and not bs.loc["Minority Interest"].dropna().empty:
                    mi=float(bs.loc["Minority Interest"].dropna().iloc[0])

            if any(_is_num(x) for x in [shares,revenue,net_income,ebitda,total_debt,cash_eq,mi]):
                return {"SharesOutstanding":shares,"Revenue":revenue,"NetIncome":net_income,"EBITDA":ebitda,
                        "TotalDebt":total_debt,"CashAndCashEquivalents":cash_eq,"MinorityInterest":mi,
                        "NetDebt": (total_debt or 0.0) - (cash_eq or 0.0)}, notes
        except Exception as e:
            notes.append(f"yfinance error: {str(e)[:80]}")

        # 2) Fallback to your library’s fundamentals (robust offline path)
        try:
            _df = compute_fundamentals_actuals([t], basis="annual", as_df=True)
            # adapt these field names to your actual columns if different:
            shares    = _safe_float(_df.loc[t, "SharesOutstanding"]) if t in _df.index and "SharesOutstanding" in _df.columns else None
            revenue   = _safe_float(_df.loc[t, "TotalRevenue"])      if t in _df.index and "TotalRevenue"      in _df.columns else None
            net_income= _safe_float(_df.loc[t, "NetIncome"])         if t in _df.index and "NetIncome"         in _df.columns else None
            ebitda    = _safe_float(_df.loc[t, "EBITDA"])            if t in _df.index and "EBITDA"            in _df.columns else None
            total_debt= _safe_float(_df.loc[t, "TotalDebt"])         if t in _df.index and "TotalDebt"         in _df.columns else None
            cash_eq   = _safe_float(_df.loc[t, "CashAndCashEquivalents"]) if t in _df.index and "CashAndCashEquivalents" in _df.columns else None
            mi        = _safe_float(_df.loc[t, "MinorityInterest"])  if t in _df.index and "MinorityInterest"  in _df.columns else None

            return {"SharesOutstanding":shares,"Revenue":revenue,"NetIncome":net_income,"EBITDA":ebitda,
                    "TotalDebt":total_debt,"CashAndCashEquivalents":cash_eq,"MinorityInterest":mi,
                    "NetDebt": (total_debt or 0.0) - (cash_eq or 0.0)}, notes
        except Exception as e:
            notes.append(f"fallback fundamentals error: {str(e)[:80]}")

        # 3) give up gracefully
        return {"SharesOutstanding":None,"Revenue":None,"NetIncome":None,"EBITDA":None,
                "TotalDebt":None,"CashAndCashEquivalents":None,"MinorityInterest":None,"NetDebt":None}, notes

    inputs, fetch_notes = _pull_fundamentals_for_price_bands(t)
    notes.extend(fetch_notes)
    shares = inputs["SharesOutstanding"]; revenue = inputs["Revenue"]; net_income = inputs["NetIncome"]
    ebitda = inputs["EBITDA"]; total_debt = inputs["TotalDebt"]; cash_eq = inputs["CashAndCashEquivalents"]; mi = inputs["MinorityInterest"]
    net_debt = inputs["NetDebt"]

    # --- pick bands -> to per-share prices -----------------------------------
    def _bands(metric_key):
        if metric_key not in bdf.index: return (None, None, None)
        row = bdf.loc[metric_key]
        p25 = _pick(row, ("P25","25TH"))
        p50 = _pick(row, ("MEDIAN","P50","50TH"))
        p75 = _pick(row, ("P75","75TH"))
        return (p25, p50, p75)

    pe_p25, pe_med, pe_p75 = _bands("PE")
    ps_p25, ps_med, ps_p75 = _bands("PS")
    ee_p25, ee_med, ee_p75 = _bands("EV_EBITDA")

    def _price_tuple_PE():
        if not (_is_pos(shares) and _is_num(net_income)): return (None,None,None)
        base = net_income / shares
        return tuple( (m * base) if _is_num(m) else None for m in (pe_p25, pe_med, pe_p75) )

    def _price_tuple_PS():
        if not (_is_pos(shares) and _is_num(revenue)): return (None,None,None)
        base = revenue / shares
        return tuple( (m * base) if _is_num(m) else None for m in (ps_p25, ps_med, ps_p75) )

    def _price_tuple_EE():
        if not (_is_pos(shares) and _is_num(ebitda)): return (None,None,None)
        def _one(mult):
            if not _is_num(mult): return None
            ev = mult * ebitda
            equity = ev - (net_debt or 0.0) - (mi or 0.0) + (cash_eq or 0.0)
            return equity / shares if _is_pos(shares) else None
        return (_one(ee_p25), _one(ee_med), _one(ee_p75))

    out = pd.DataFrame([{
        "Ticker": t,
        "PE_Based_Valuation_P25_P50_P75": _price_tuple_PE(),
        "PS_Based_Valuation_P25_P50_P75": _price_tuple_PS(),
        "EV_EBITDA_Based_Valuation_P25_P50_P75": _price_tuple_EE(),
        "Inputs_Used": {
            "SharesOutstanding": shares,
            "NetIncome": net_income,
            "Revenue": revenue,
            "EBITDA": ebitda,
            "TotalDebt": total_debt,
            "CashAndCashEquivalents": cash_eq,
            "MinorityInterest": mi,
            "NetDebt": net_debt,
        },
        "Notes": " ".join(n for n in notes if n).strip()
    }])

    if save_csv:
        _ensure_dir("output")
        out.to_csv(f"output/price_from_peer_multiples_{t}_{analysis_report_date.replace('-','')}.csv", index=False)
    return out if as_df else to_records(out, analysis_report_date=analysis_report_date)

def historical_growth_metrics(
    tickers: Union[str, List[str]],
    *,
    min_years: int = 3,
    analysis_report_date: Optional[str] = None,
    save_csv: bool = False,
    as_df: bool = True,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Compute historical CAGRs for Revenue, Net Income, and Free Cash Flow (FCF).
    Falls back to quarterly roll-ups (last 4 quarters) when annuals are sparse.
    FCF CAGR is only computed when first and last values > 0 (to avoid undefined growth).
    """
    analysis_report_date = analysis_report_date or _today_iso()
    if isinstance(tickers, str):
        tickers = [tickers]

    def _cagr(first, last, n_years):
        try:
            if _is_num(first) and _is_num(last) and n_years > 0 and first > 0 and last > 0:
                return (last / first) ** (1.0 / n_years) - 1.0
        except Exception:
            pass
        return None

    def _annual_series(df, field):
        if not isinstance(df, pd.DataFrame) or field not in df.index:
            return None
        s = df.loc[field].dropna()
        if s.empty: return None
        return s[::-1]  # oldest→newest

    rows = []
    for raw in tickers:
        t = _sanitize_ticker(raw)
        notes = []
        R_cagr = E_cagr = F_cagr = None
        n_years_obs = None
        y0 = y1 = None

        try:
            yt   = yf.Ticker(t)
            fin  = getattr(yt, "financials", None)
            qfin = getattr(yt, "quarterly_financials", None)
            cfa  = getattr(yt, "cashflow", None)
            qcf  = getattr(yt, "quarterly_cashflow", None)

            # Revenue & Net Income annual
            rev = _annual_series(fin, "Total Revenue")
            ni  = _annual_series(fin, "Net Income")

            # --- FCF series (annual) using the canonical helper ---
            fcf = _fcf_series_from_cashflow(cfa)  # returns OCF + CapEx (CapEx negative → added)
            if isinstance(fcf, pd.Series) and not fcf.empty:
                # enforce chronological order oldest→newest
                years = pd.to_datetime(fcf.index, errors="coerce").year
                order = np.argsort(years)
                fcf = pd.Series(fcf.values[order], index=fcf.index[order])
            else:
                fcf = None

            # helper to compute CAGR & period
            def _compute(s, label):
                if s is None or len(s.dropna()) < min_years:
                    return None, None, None, f"{label} series has <{min_years} clean annual points."
                s = s.dropna()
                first, last = float(s.iloc[0]), float(s.iloc[-1])
                years = len(s) - 1
                cagr = _cagr(first, last, years)
                reason = None
                if cagr is None and (first <= 0 or last <= 0):
                    reason = f"{label} CAGR undefined (first/last ≤ 0)."
                return cagr, years, (s.index[0].year, s.index[-1].year), reason

            R_cagr, R_years, R_period, R_reason = _compute(rev, "Revenue")
            E_cagr, E_years, E_period, E_reason = _compute(ni,  "Earnings")
            F_cagr, F_years, F_period, F_reason = _compute(fcf, "FCF")

            for r in [R_reason, E_reason, F_reason]:
                if r: notes.append(r)

            # observed period (max span among available series)
            years_list = [y for y in [R_years, E_years, F_years] if isinstance(y, int)]
            if years_list:
                n_years_obs = int(max(years_list))
            periods = [p for p in [R_period, E_period, F_period] if isinstance(p, tuple)]
            if periods:
                y0 = min(p[0] for p in periods)
                y1 = max(p[1] for p in periods)

        except Exception as e:
            notes.append(f"Error fetching statements: {str(e)[:120]}")

        rows.append({
            "Ticker": t,
            "Revenue_CAGR": R_cagr,
            "Earnings_CAGR": E_cagr,
            "FCF_CAGR": F_cagr,
            #"Observed_Time_Period_Years": n_years_obs,
            "Period_Start_Year": y0,
            "Period_End_Year": y1,
            "Notes": " ".join(n for n in notes if n).strip()
        })

    out = pd.DataFrame(rows)
    if save_csv:
        _ensure_dir("output")
        out.to_csv(f"output/historical_growth_{analysis_report_date.replace('-','')}.csv", index=False)
    return out if as_df else to_records(out, analysis_report_date=analysis_report_date)

def _fcf_series_from_cashflow(cf: pd.DataFrame) -> pd.Series:
    """CORRECTED: FCF = Operating Cash Flow - Capital Expenditure"""
    if not isinstance(cf, pd.DataFrame) or cf.empty:
        return pd.Series(dtype=float)
    
    ocf = cf.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cf.index else pd.Series(dtype=float)
    capex = cf.loc['Capital Expenditure'].dropna() if 'Capital Expenditure' in cf.index else pd.Series(dtype=float)
    
    # CORRECTED: Subtract CapEx (it's usually negative in yfinance, so we add it)
    if not ocf.empty and not capex.empty:
        # Ensure same index
        common_idx = ocf.index.intersection(capex.index)
        if not common_idx.empty:
            return ocf.loc[common_idx] + capex.loc[common_idx]  # CapEx is negative
    
    return pd.Series(dtype=float)

def _revenue_cagr_from_series(series: pd.Series) -> Optional[float]:
    if isinstance(series, pd.Series) and len(series) >= 2:
        start = _safe_float(series.iloc[-1])
        end   = _safe_float(series.iloc[0])
        years = len(series) - 1
        if _is_pos(start) and _is_pos(end) and years > 0:
            return (end / start) ** (1/years) - 1
    return None

def _dcf_enterprise_value(fcf: float, g: float, wacc: float, years: int = 5) -> float:
    if not _is_pos(fcf) or not _is_pos(wacc) or g is None or (g >= wacc):
        return 0.0
    # Year 1..years
    cfs = [(fcf * (1 + g)**i) / ((1 + wacc)**i) for i in range(1, years + 1)]
    tv  = (fcf * (1 + g)**years) * (1 + g) / (wacc - g)
    tvd = tv / ((1 + wacc)**years)
    return float(sum(cfs) + tvd)

def _calculate_wacc(snap: Dict[str, Any], risk_free_rate: float, equity_risk_premium: float, beta: float) -> Optional[float]:
    """Calculate proper WACC including debt costs"""
    try:
        # Get balance sheet data
        ticker_obj = yf.Ticker(snap['ticker'])
        bs = ticker_obj.balance_sheet
        income = ticker_obj.financials
        
        if bs.empty or income.empty:
            # Fallback to cost of equity if no debt data
            return risk_free_rate + beta * equity_risk_premium
        
        # Market value of equity
        market_cap = _safe_float(snap.get('market_cap'))
        if not _is_pos(market_cap):
            return None
            
        # Book value of debt (total debt)
        total_debt = 0
        debt_items = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
        for item in debt_items:
            if item in bs.index:
                debt_val = _safe_float(bs.loc[item].iloc[0])
                if _is_pos(debt_val):
                    total_debt += debt_val
                    break
        
        # If no debt, return cost of equity
        if total_debt <= 0:
            return risk_free_rate + beta * equity_risk_premium
        
        # Cost of debt approximation
        interest_expense = 0
        if 'Interest Expense' in income.index:
            interest_expense = abs(_safe_float(income.loc['Interest Expense'].iloc[0]) or 0)
        
        cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.04
        cost_of_debt = max(cost_of_debt, 0.02)  # Minimum 2%
        cost_of_debt = min(cost_of_debt, 0.15)  # Maximum 15%
        
        # Tax rate approximation
        pretax_income = _safe_float(income.loc['Pretax Income'].iloc[0]) if 'Pretax Income' in income.index else None
        net_income = _safe_float(income.loc['Net Income'].iloc[0]) if 'Net Income' in income.index else None
        
        if _is_pos(pretax_income) and _is_num(net_income):
            tax_rate = max(0, (pretax_income - net_income) / pretax_income)
            tax_rate = min(tax_rate, 0.35)  # Cap at 35%
        else:
            tax_rate = 0.25  # Default corporate tax rate
        
        # Calculate weights
        total_value = market_cap + total_debt
        equity_weight = market_cap / total_value
        debt_weight = total_debt / total_value
        
        # Cost of equity
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        
        # WACC calculation
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc
        
    except Exception:
        # Fallback to cost of equity
        return risk_free_rate + beta * equity_risk_premium

def _fcf_cagr_from_series(fcf_series: pd.Series) -> Optional[float]:
    """Calculate FCF CAGR from time series"""
    if isinstance(fcf_series, pd.Series) and len(fcf_series) >= 3:
        # Remove zeros and negatives for CAGR calculation
        positive_fcf = fcf_series[fcf_series > 0]
        if len(positive_fcf) >= 3:
            start = _safe_float(positive_fcf.iloc[-1])
            end = _safe_float(positive_fcf.iloc[0])
            years = len(positive_fcf) - 1
            if _is_pos(start) and _is_pos(end) and years > 0:
                cagr = (end / start) ** (1/years) - 1
                return cagr if -0.5 <= cagr <= 0.5 else None
    return None


def _normalized_fcf_baseline(fcf_series: pd.Series) -> Optional[float]:
    """Calculate normalized FCF baseline with outlier removal"""
    if fcf_series.empty:
        return None
    
    # Use last 4-6 years if available
    recent_fcf = fcf_series.tail(6)
    
    if len(recent_fcf) < 2:
        return None
    
    # Remove outliers (beyond 1.5 * IQR)
    q1, q3 = recent_fcf.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_fcf = recent_fcf[(recent_fcf >= lower_bound) & (recent_fcf <= upper_bound)]
    else:
        filtered_fcf = recent_fcf
    
    if filtered_fcf.empty:
        return _safe_float(recent_fcf.median())
    
    # Weight more recent years more heavily
    weights = np.linspace(0.5, 1.0, len(filtered_fcf))
    weighted_avg = np.average(filtered_fcf, weights=weights)
    
    return _safe_float(weighted_avg)

def dcf_three_scenarios(
    ticker: str,
    *,
    peer_tickers: Optional[List[str]] = None,   # optional seeding
    years: int = 5,
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    
    # NEW: Advanced controls for transformational companies
    fcf_window_years: int | None = VALUATION_DEFAULTS["fcf_window_years"],     # set default window to latest 3Y FCF
    manual_baseline_fcf: Optional[float] = None, # Override FCF baseline (None = calculated)
    manual_growth_rates: Optional[List[float]] = None, # [low, mid, high] growth override
    assumptions_as_of: Optional[str] = None,
    
    as_df: bool = True,
    analysis_report_date: Optional[str] = None
):
    """
    Compute Low/Mid/High per-share DCF using:
      - growth seeds from peers' FCF CAGRs if provided, else target FCF CAGR, else fallback (3%).
      - Proper WACC calculation including debt costs.
      - growth capped at (WACC - 0.5%).
    
    NEW ADVANCED CONTROLS:
      - fcf_window_years: Limit FCF averaging window (useful for transformational companies)
      - manual_baseline_fcf: Override calculated FCF baseline with current run-rate
      - manual_growth_rates: Override growth logic with custom [low, mid, high] rates
    
    Returns DF with columns: Scenario, Growth_Used, WACC_Used, Per_Share_Value.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(ticker)
    snap = _pull_company_snapshot(t)
    assumptions_used = valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=fcf_window_years,
    )
    
    # Initialize notes list FIRST
    notes = []

    beta = _safe_float(snap["beta"])
    if beta is None:
        raise ValueError(f"Missing beta from yfinance for {t}. Cannot compute WACC.")

    # Calculate proper WACC
    wacc_mid = _calculate_wacc(snap, risk_free_rate, equity_risk_premium, beta)
    if wacc_mid is None:
        raise ValueError(f"Cannot calculate WACC for {t}. Missing required financial data.")

    # CORRECTED: Higher growth should have higher WACC (more risk)
    wacc_low = wacc_mid - 0.005   # Low growth = lower risk = lower WACC
    wacc_high = wacc_mid + 0.01   # High growth = higher risk = higher WACC

    # UPDATED: Enhanced FCF baseline with window control
    fcf_series = _fcf_series_from_cashflow(snap['cashflow'])
    
    # NEW: Manual FCF override
    if manual_baseline_fcf is not None:
        avg_fcf = _safe_float(manual_baseline_fcf)
        notes.append(f"Using manual FCF baseline: ${avg_fcf/1e9:.1f}B")
    else:
        # NEW: FCF window control
        if fcf_window_years is not None and fcf_window_years > 0:
            fcf_series = fcf_series.tail(fcf_window_years)
            notes.append(f"FCF calculated using last {fcf_window_years} years only")
        
        avg_fcf = _normalized_fcf_baseline(fcf_series)

    if not _is_pos(avg_fcf):
        notes.append("FCF baseline not positive; per-share values will be None.")
        df = pd.DataFrame([
            {"Scenario": "DCF_Lo_Growth_Lo_WACC", "Growth_Used": None, "WACC_Used": wacc_low, "Per_Share_Value": None, "Assumptions_Used": assumptions_used},
            {"Scenario": "DCF_Mid_Growth_Mid_WACC","Growth_Used": None, "WACC_Used": wacc_mid, "Per_Share_Value": None, "Assumptions_Used": assumptions_used},
            {"Scenario": "DCF_Hi_Growth_Hi_WACC", "Growth_Used": None, "WACC_Used": wacc_high, "Per_Share_Value": None, "Assumptions_Used": assumptions_used},
        ])
        if as_df:
            return df
        return to_records(df, analysis_report_date=analysis_report_date, notes=notes)

    # NEW: Manual growth rates override
    if manual_growth_rates is not None:
        if len(manual_growth_rates) != 3:
            raise ValueError("manual_growth_rates must contain exactly 3 values: [low, mid, high]")
        g_low, g_mid, g_high = manual_growth_rates
        notes.append(f"Using manual growth rates: {g_low:.1%}, {g_mid:.1%}, {g_high:.1%}")
    else:
        # Original growth seeding logic
        peer_fcf_cagrs = []
        if peer_tickers:
            for p in peer_tickers:
                try:
                    peer_snap = _pull_company_snapshot(_sanitize_ticker(p))
                    peer_fcf_series = _fcf_series_from_cashflow(peer_snap['cashflow'])
                    c = _fcf_cagr_from_series(peer_fcf_series)
                    if _is_num(c) and -0.5 <= c <= 0.5:  # Reasonable bounds
                        peer_fcf_cagrs.append(float(c))
                except Exception:
                    continue

        # Try FCF CAGR first, then revenue CAGR as fallback
        target_fcf_series = fcf_series if fcf_window_years is None else _fcf_series_from_cashflow(snap['cashflow'])
        fcf_cagr_target = _fcf_cagr_from_series(target_fcf_series)
        rev_cagr_target = _revenue_cagr_from_series(snap["revenue_series"])
        
        if len(peer_fcf_cagrs) >= 3:
            g_low, g_mid, g_high = np.percentile(peer_fcf_cagrs, [25, 50, 75])
            notes.append("Growth rates seeded from peer FCF CAGRs")
        else:
            # Use FCF CAGR if available and reasonable, else revenue CAGR, else fallback
            if _is_num(fcf_cagr_target) and -0.3 <= fcf_cagr_target <= 0.3:
                base = fcf_cagr_target
                notes.append("Growth rates based on target FCF CAGR")
            elif _is_num(rev_cagr_target) and -0.3 <= rev_cagr_target <= 0.3:
                base = rev_cagr_target * 0.8  # Conservative adjustment
                notes.append("Growth rates based on target revenue CAGR (adjusted)")
            else:
                base = target_cagr_fallback
                notes.append("Growth rates using fallback assumption")
            
            g_low, g_mid, g_high = base * 0.6, base, base * 1.3

    # Cap growth if it exceeds WACC
    def _cap(g, w): 
        if not _is_num(g) or not _is_num(w):
            return None
        gap = assumptions_used["terminal_growth_gap"]
        return min(max(g, -0.05), w - gap)  # Also floor at -5%
    
    # CORRECTED: Match risk levels properly
    gL = _cap(g_low, wacc_low)      # Low growth, low WACC
    gM = _cap(g_mid, wacc_mid)      # Mid growth, mid WACC  
    gH = _cap(g_high, wacc_high)    # High growth, high WACC

    vL = _dcf_enterprise_value(avg_fcf, gL, wacc_low, years=years)
    vM = _dcf_enterprise_value(avg_fcf, gM, wacc_mid, years=years)
    vH = _dcf_enterprise_value(avg_fcf, gH, wacc_high, years=years)

    so = _safe_float(snap['shares_outstanding'])
    vLp = round(vL / so, 3) if _is_pos(so) and _is_pos(vL) else None
    vMp = round(vM / so, 3) if _is_pos(so) and _is_pos(vM) else None
    vHp = round(vH / so, 3) if _is_pos(so) and _is_pos(vH) else None

    # Add reasonableness checks
    if _is_pos(vM) and _is_pos(avg_fcf):
        ev_fcf_multiple = vM / avg_fcf
        if ev_fcf_multiple > 50:
            notes.append(f"Warning: EV/FCF multiple of {ev_fcf_multiple:.1f}x seems high; check assumptions.")
        elif ev_fcf_multiple < 5:
            notes.append(f"Note: EV/FCF multiple of {ev_fcf_multiple:.1f}x is relatively low.")

    df = pd.DataFrame([
        {"Scenario": "DCF_Lo_Growth_Lo_WACC", "Growth_Used": gL, "WACC_Used": wacc_low, "Per_Share_Value": vLp, "Assumptions_Used": assumptions_used},
        {"Scenario": "DCF_Mid_Growth_Mid_WACC","Growth_Used": gM, "WACC_Used": wacc_mid,  "Per_Share_Value": vMp, "Assumptions_Used": assumptions_used},
        {"Scenario": "DCF_Hi_Growth_Hi_WACC", "Growth_Used": gH, "WACC_Used": wacc_high,  "Per_Share_Value": vHp, "Assumptions_Used": assumptions_used},
    ])

    if as_df:
        return df
        
    return to_records(df, analysis_report_date=analysis_report_date, notes=notes)

# HELPER FUNCTION: More robust debt extraction
def _get_total_debt_from_info(info: dict) -> Optional[float]:
    """Extract total debt from yfinance info dict with multiple fallbacks"""
    debt_keys = [
        "totalDebt",
        "longTermDebt", 
        "shortTermDebt",
        "totalCurrentLiabilities"
    ]
    
    total_debt = 0
    found_debt = False
    
    for key in debt_keys:
        debt_val = _safe_float(info.get(key))
        if _is_pos(debt_val):
            if key in ["totalDebt"]:
                return debt_val  # Use total debt if available
            elif key in ["longTermDebt", "shortTermDebt"]:
                total_debt += debt_val
                found_debt = True
    
    return total_debt if found_debt else None

def dcf_implied_enterprise_value(
    ticker: str,
    *,
    years: Optional[int] = None,             # None => perpetuity-only (Gordon Growth)
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    growth: Optional[float] = None,          # if None -> use FCF CAGR first, then revenue CAGR; fallback 3%
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    use_average_fcf_years: Optional[int] = VALUATION_DEFAULTS["fcf_window_years"],   # None -> use ALL available FCF points
    volatility_threshold: float = 0.5,       # coefficient of variation threshold for a volatility note
    assumptions_as_of: Optional[str] = None,
    as_df: bool = True,
    analysis_report_date: Optional[str] = None
):
    """
    Forward valuation: take avg historical FCF, growth g, and WACC; return implied Enterprise Value (EV).

    IMPROVEMENTS:
    - Now uses proper WACC calculation including debt costs
    - Prioritizes FCF CAGR over revenue CAGR for growth estimation
    - Enhanced FCF normalization with outlier removal
    - Better error handling and validation

    Modes:
      - years is None  => Perpetuity-only (Gordon Growth), no explicit forecast horizon.
      - years > 0      => Multi-year DCF for 'years' then a terminal value at year 'years'.

    Notes:
      - use_average_fcf_years = None → use all available historical FCF points from Yahoo (typically 4–5).
      - Adds a volatility note if (std/|mean|) of the FCF sample > volatility_threshold.
      - See notes in return payload for data health flags and assumptions.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(ticker)
    snap = _pull_company_snapshot(t)
    assumptions_used = valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
    )

    # --- IMPROVED: Calculate proper WACC ---
    beta = _safe_float(snap.get("beta"))
    if beta is None:
        raise ValueError(f"Missing beta from yfinance for {t}. Cannot compute WACC.")
    
    # Use the improved WACC calculation
    wacc = _calculate_wacc(snap, risk_free_rate, equity_risk_premium, beta)
    if wacc is None:
        # Fallback to cost of equity if WACC calculation fails
        wacc = risk_free_rate + beta * equity_risk_premium

    # FCF series (Operating CF - CapEx) - using corrected calculation
    fcf_series_all = _fcf_series_from_cashflow(snap.get('cashflow'))
    notes: List[str] = []

    if not isinstance(fcf_series_all, pd.Series) or fcf_series_all.empty:
        notes.append("No historical FCF points available from Yahoo.")
        df = pd.DataFrame([{
            "Ticker": t, "Avg_FCF_Used": None, "Growth_Used": None, "WACC_Used": wacc,
            "Years": 0 if years is None else int(years), "EV_Implied": None, "Assumptions_Used": assumptions_used, "Notes": " ".join(notes)
        }])
        return df if as_df else to_records(df, analysis_report_date=analysis_report_date, notes=notes)

    # Determine the averaging window
    fcf_series_all = fcf_series_all.dropna()
    available = len(fcf_series_all)
    n = available if use_average_fcf_years is None else max(1, min(use_average_fcf_years, available))
    fcf_window = fcf_series_all.tail(n)

    # IMPROVED: Enhanced volatility analysis
    mean_f = fcf_window.mean()
    std_f = fcf_window.std(ddof=1) if len(fcf_window) >= 2 else 0.0
    cov = (abs(std_f / mean_f) if (mean_f not in (0, None) and pd.notna(mean_f) and mean_f != 0) else np.nan)
    
    # Add volatility context
    if _is_num(cov):
        if cov > volatility_threshold:
            notes.append(f"Historical FCF highly volatile (CoV={cov:.2f} > threshold {volatility_threshold:.2f}); consider shortening averaging window.")
        elif cov > 0.3:
            notes.append(f"Historical FCF moderately volatile (CoV={cov:.2f}).")

    # IMPROVED: Use normalized FCF baseline instead of simple mean
    avg_fcf = _normalized_fcf_baseline(fcf_window)
    
    if not _is_pos(avg_fcf):
        notes.append("Average FCF not positive; EV will be None.")
        df = pd.DataFrame([{
            "Ticker": t, "Avg_FCF_Used": avg_fcf, "Growth_Used": None, "WACC_Used": wacc,
            "Years": 0 if years is None else int(years), "EV_Implied": None, "Assumptions_Used": assumptions_used, "Notes": " ".join(notes)
        }])
        return df if as_df else to_records(df, analysis_report_date=analysis_report_date, notes=notes)

    # IMPROVED: Better growth estimation logic
    if growth is not None:
        g = growth
        notes.append("Using user-provided growth rate.")
    else:
        # Try FCF CAGR first, then revenue CAGR, then fallback
        fcf_cagr = _fcf_cagr_from_series(fcf_series_all)
        rev_cagr = _revenue_cagr_from_series(snap.get('revenue_series'))
        
        if _is_num(fcf_cagr) and -0.3 <= fcf_cagr <= 0.3:
            g = fcf_cagr
            notes.append("Using historical FCF CAGR for growth.")
        elif _is_num(rev_cagr) and -0.3 <= rev_cagr <= 0.3:
            g = rev_cagr * 0.8  # Conservative adjustment
            notes.append("Using historical revenue CAGR (adjusted) for growth.")
        else:
            g = target_cagr_fallback
            notes.append("Using fallback growth rate.")

    # Guardrail: cap g slightly below WACC to avoid division by zero in TV
    capped = False
    original_g = g
    if _is_num(wacc) and g is not None and g >= wacc:
        g = wacc - assumptions_used["terminal_growth_gap"]
        capped = True

    # Additional validation for negative growth
    if g is not None and g < -0.05:
        g = max(g, -0.05)  # Floor at -5%
        notes.append("Growth rate floored at -5% for stability.")

    # Compute EV depending on mode
    if years is None:
        # Perpetuity-only (Gordon Growth)
        ev = (avg_fcf * (1 + g)) / (wacc - g) if (_is_num(g) and _is_pos(wacc) and (wacc - g) > 0) else 0.0
        notes.append("Perpetuity-only valuation (no explicit forecast horizon).")
        years_used = 0
    else:
        ev = _dcf_enterprise_value(avg_fcf, g, wacc, years=years) if (_is_num(g) and _is_pos(wacc)) else 0.0
        notes.append(f"DCF with {years} years explicit forecast before terminal value.")
        years_used = int(years)

    if capped:
        notes.append(f"Growth capped from {original_g:.1%} to {g:.1%} (WACC - 0.5%) to stabilize terminal value.")

    # Add reasonableness checks
    if _is_pos(ev) and _is_pos(avg_fcf):
        ev_fcf_multiple = ev / avg_fcf
        if ev_fcf_multiple > 50:
            notes.append(f"Warning: EV/FCF multiple of {ev_fcf_multiple:.1f}x seems high; check assumptions.")
        elif ev_fcf_multiple < 5:
            notes.append(f"Note: EV/FCF multiple of {ev_fcf_multiple:.1f}x is relatively low.")

    out = pd.DataFrame([{
        "Ticker": t,
        "Avg_FCF_Used": float(avg_fcf) if _is_num(avg_fcf) else None,
        "Growth_Used": float(g) if _is_num(g) else None,
        "WACC_Used": float(wacc) if _is_num(wacc) else None,
        "Years": years_used,
        "Assumptions_Used": assumptions_used,
        "EV_Implied": float(ev) if _is_num(ev) else None,
        "Notes": " ".join(notes)
    }])
    return out if as_df else to_records(out, analysis_report_date=analysis_report_date, notes=notes)


def compare_to_market_ev(
    ticker: str,
    *,
    years: Optional[int] = None,             # None => perpetuity-only mode in implied EV step
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    growth: Optional[float] = None,
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    use_average_fcf_years: int | None = VALUATION_DEFAULTS["fcf_window_years"],
    volatility_threshold: float = 0.5,
    assumptions_as_of: Optional[str] = None,
    as_df: bool = True,
    analysis_report_date: Optional[str] = None
):
    """
    Compute implied EV (perpetuity-only if years=None; multi-year DCF if years>0) and compare to Yahoo enterpriseValue.
    Uses explicit horizon + terminal value (or horizon=0 → terminal-only) to estimate EV.

    IMPROVEMENTS:
    - Enhanced market data validation
    - Better premium interpretation with context
    - More robust error handling

    Returns DataFrame (or JSON envelope) with:
      ['Ticker','Observed_EV','EV_Implied','Premium_%','Avg_FCF_Used','Growth_Used','WACC_Used','Years','Notes']

    Interpretation:
      Premium_% > 0 → observed EV exceeds DCF-implied EV (potentially over-valued or pricing in higher growth).
      Premium_% < 0 → observed EV below DCF-implied EV (potentially undervalued or market expects lower growth).
    """
    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(ticker)

    # Get implied EV using improved function
    implied_df = dcf_implied_enterprise_value(
        t,
        years=years,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        growth=growth,
        target_cagr_fallback=target_cagr_fallback,
        use_average_fcf_years=use_average_fcf_years,
        volatility_threshold=volatility_threshold,
        assumptions_as_of=assumptions_as_of,
        as_df=True,
        analysis_report_date=analysis_report_date
    )

    implied_ev = implied_df.loc[0, "EV_Implied"]
    avg_fcf = implied_df.loc[0, "Avg_FCF_Used"]
    g_used = implied_df.loc[0, "Growth_Used"]
    wacc_used = implied_df.loc[0, "WACC_Used"]
    years_used = int(implied_df.loc[0, "Years"] )
    assumptions_used = implied_df.loc[0, "Assumptions_Used"] if "Assumptions_Used" in implied_df.columns else valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
    )
    base_notes = implied_df.loc[0, "Notes"] or ""

    # IMPROVED: More robust market data retrieval
    try:
        info = yf.Ticker(t).info
        observed_ev = _safe_float(info.get("enterpriseValue"))
        market_cap = _safe_float(info.get("marketCap"))
        
        # Additional validation
        if observed_ev is None and market_cap is not None:
            # Try to estimate EV from market cap if direct EV not available
            total_debt = _get_total_debt_from_info(info)
            cash = _safe_float(info.get("totalCash", 0))
            estimated_ev = market_cap + (total_debt or 0) - cash
            if estimated_ev > 0:
                observed_ev = estimated_ev
                base_notes += " EV estimated from market cap + debt - cash."
                
    except Exception as e:
        observed_ev = None
        base_notes += f" Error retrieving market data: {str(e)[:50]}."

    notes = [base_notes] if base_notes else []
    
    if not _is_pos(observed_ev):
        notes.append("Yahoo enterpriseValue not available; comparison limited.")
        premium_pct = None
    else:
        if implied_ev is None or not _is_num(implied_ev) or implied_ev <= 0:
            premium_pct = None
            notes.append("Cannot calculate premium: implied EV not valid.")
        else:
            premium_pct = (observed_ev / implied_ev - 1.0) * 100
            
            # IMPROVED: More nuanced interpretation
            if abs(premium_pct) < 5:
                notes.append("Observed EV roughly equals DCF-implied EV (within 5%).")
            elif premium_pct > 20:
                notes.append("Observed EV significantly above DCF-implied EV (>20% premium); market expects higher growth or lower risk.")
            elif premium_pct > 0:
                notes.append("Observed EV modestly above DCF-implied EV (positive premium).")
            elif premium_pct < -20:
                notes.append("Observed EV significantly below DCF-implied EV (>20% discount); potential undervaluation or market expects lower growth.")
            else:
                notes.append("Observed EV modestly below DCF-implied EV (negative premium).")

    out = pd.DataFrame([{
        "Ticker": t,
        "Observed_EV": float(observed_ev) if _is_num(observed_ev) else None,
        "EV_Implied": float(implied_ev) if _is_num(implied_ev) else None,
        "Premium_%": float(premium_pct) if _is_num(premium_pct) else None,
        "Avg_FCF_Used": float(avg_fcf) if _is_num(avg_fcf) else None,
        "Growth_Used": float(g_used) if _is_num(g_used) else None,
        "WACC_Used": float(wacc_used) if _is_num(wacc_used) else None,
        "Years": years_used,
        "Assumptions_Used": assumptions_used,
        "Notes": " ".join([n for n in notes if n])
    }])

    return out if as_df else to_records(out, analysis_report_date=analysis_report_date, notes=[n for n in notes if n])


def implied_equity_value_from_ev(
    ticker: str,
    ev_implied: float,
    *,
    notes_from_ev: Optional[str] = None,   # carry-forward notes like "Perpetuity-only valuation"
    as_df: bool = True,
    analysis_report_date: Optional[str] = None
):
    """
    Convert an implied Enterprise Value (EV) into implied Equity Value using the MOST RECENT balance sheet:
      Equity ≈ EV − TotalDebt + CashAndCashEquivalents − MinorityInterest

    IMPROVEMENTS:
    - More robust balance sheet data extraction
    - Better handling of missing data with multiple fallback sources
    - Enhanced validation and error reporting
    - Per-share calculation added

    - If any of the balance-sheet fields are missing, they are treated as 0 and a note is added.
    - `notes_from_ev` (if provided) is prepended to the Notes for traceability.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(ticker)
    
    notes: List[str] = []
    if notes_from_ev:
        notes.append(str(notes_from_ev).strip())

    # IMPROVED: More robust balance sheet data retrieval
    try:
        ticker_obj = yf.Ticker(t)
        bs = ticker_obj.balance_sheet
        info = ticker_obj.info
        
        # Try balance sheet first, then info as fallback
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            col = bs.columns[0]  # Most recent column
            
            def _get_row(name: str, alternatives: List[str] = None) -> Optional[float]:
                # Try primary name first
                if name in bs.index:
                    return _safe_float(bs.loc[name, col])
                # Try alternatives
                if alternatives:
                    for alt in alternatives:
                        if alt in bs.index:
                            return _safe_float(bs.loc[alt, col])
                return None

            total_debt = _get_row("Total Debt", ["Long Term Debt", "Short Long Term Debt"])
            cash_eq = _get_row("Cash And Cash Equivalents", ["Cash", "Cash And Short Term Investments"])
            minority = _get_row("Minority Interest", ["Noncontrolling Interest"])
            
        else:
            # Fallback to info dict
            notes.append("Balance sheet not available; using summary data from info.")
            total_debt = _get_total_debt_from_info(info)
            cash_eq = _safe_float(info.get("totalCash"))
            minority = None  # Usually not in info
            
    except Exception as e:
        notes.append(f"Error accessing financial data: {str(e)[:50]}.")
        total_debt = cash_eq = minority = None

    # Handle missing values
    if total_debt is None:
        total_debt = 0.0
        notes.append("Total Debt missing; treated as 0.")
    if cash_eq is None:
        cash_eq = 0.0
        notes.append("Cash And Cash Equivalents missing; treated as 0.")
    if minority is None:
        minority = 0.0
        notes.append("Minority Interest missing; treated as 0.")

    # Calculate equity value
    if _is_num(ev_implied):
        equity_implied = ev_implied - total_debt + cash_eq - minority
    else:
        equity_implied = None
        notes.append("Cannot calculate equity value: EV_Implied is not valid.")

    # IMPROVED: Add per-share calculation
    shares_outstanding = None
    per_share_value = None
    try:
        shares_outstanding = _safe_float(yf.Ticker(t).info.get("sharesOutstanding"))
        if _is_pos(shares_outstanding) and _is_num(equity_implied):
            per_share_value = equity_implied / shares_outstanding
    except Exception:
        notes.append("Could not retrieve shares outstanding for per-share calculation.")

    # Validation checks
    if _is_num(equity_implied) and equity_implied < 0:
        notes.append("Warning: Implied equity value is negative (debt exceeds EV + cash).")
    
    if _is_num(total_debt) and _is_num(cash_eq) and total_debt > cash_eq * 3:
        notes.append("Note: Company appears highly leveraged (debt > 3x cash).")

    out = pd.DataFrame([{
        "Ticker": t,
        "EV_Implied": float(ev_implied) if _is_num(ev_implied) else None,
        "TotalDebt": float(total_debt),
        "CashAndCashEquivalents": float(cash_eq),
        "MinorityInterest": float(minority),
        "Equity_Implied": float(equity_implied) if _is_num(equity_implied) else None,
        "SharesOutstanding": float(shares_outstanding) if _is_num(shares_outstanding) else None,
        "Per_Share_Value": float(per_share_value) if _is_num(per_share_value) else None,
        "Notes": " ".join([n for n in notes if n])
    }])
    return out if as_df else to_records(out, analysis_report_date=analysis_report_date, notes=[n for n in notes if n])

def compare_to_market_cap(
    ticker_or_evdf,
    *,
    years: Optional[int] = None,                  # mirrors compare_to_market_ev
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    growth: Optional[float] = None,
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    use_average_fcf_years: int | None = VALUATION_DEFAULTS["fcf_window_years"],
    volatility_threshold: float = 0.5,
    assumptions_as_of: Optional[str] = None,
    as_df: bool = True,
    analysis_report_date: Optional[str] = None
):
    """
    Compare IMPLIED EQUITY VALUE (derived from implied EV) vs OBSERVED MARKET CAP (Yahoo Finance).

    INPUTS:
      - Either pass a ticker string (same options as compare_to_market_ev),
        OR pass a single-row DataFrame returned by compare_to_market_ev().

    RETURNS:
      Single-row DataFrame with:
        Ticker, Observed_MarketCap, Equity_Implied, Premium_%,
        EV_Implied, NetDebt, CashAndCashEquivalents, MinorityInterest,
        Avg_FCF_Used, Growth_Used, WACC_Used, Years, Notes
    """
    analysis_report_date = analysis_report_date or _today_iso()

    # 1) Normalize inputs -> get compare_to_market_ev row
    if isinstance(ticker_or_evdf, pd.DataFrame):
        ev_df = ticker_or_evdf.copy()
        if ev_df.shape[0] != 1:
            raise ValueError("compare_to_market_cap expects a single-row DataFrame from compare_to_market_ev.")
        t = _sanitize_ticker(str(ev_df.iloc[0]["Ticker"]))
    else:
        # Treat as ticker and compute the EV comparison first
        t = _sanitize_ticker(str(ticker_or_evdf))
        ev_df = compare_to_market_ev(
            t,
            years=years,
            risk_free_rate=risk_free_rate,
            equity_risk_premium=equity_risk_premium,
            growth=growth,
            target_cagr_fallback=target_cagr_fallback,
            use_average_fcf_years=use_average_fcf_years,
            volatility_threshold=volatility_threshold,
            assumptions_as_of=assumptions_as_of,
            as_df=True,
            analysis_report_date=analysis_report_date
        )

    # Pull core EV outputs
    ev_implied = ev_df.iloc[0].get("EV_Implied")
    avg_fcf = ev_df.iloc[0].get("Avg_FCF_Used")
    g_used = ev_df.iloc[0].get("Growth_Used")
    wacc_used = ev_df.iloc[0].get("WACC_Used")
    years_used = int(ev_df.iloc[0].get("Years"))
    assumptions_used = ev_df.iloc[0].get("Assumptions_Used") if "Assumptions_Used" in ev_df.columns else valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
    )
    base_notes = str(ev_df.iloc[0].get("Notes") or "")

    # 2) Convert implied EV -> implied Equity via your existing helper
    eq_df = implied_equity_value_from_ev(
        t,
        ev_implied=ev_implied,
        notes_from_ev=base_notes,
        as_df=True,
        analysis_report_date=analysis_report_date
    )

    equity_implied = eq_df.iloc[0].get("Equity_Implied")
    total_debt = eq_df.iloc[0].get("TotalDebt") or 0.0
    cash_eq = eq_df.iloc[0].get("CashAndCashEquivalents") or 0.0
    minority = eq_df.iloc[0].get("MinorityInterest") or 0.0
    net_debt = float(total_debt) - float(cash_eq)

    # 3) Fetch observed Market Cap (with a basic fallback)
    notes = [base_notes, str(eq_df.iloc[0].get("Notes") or "")]
    try:
        info = yf.Ticker(t).info
        observed_mktcap = _safe_float(info.get("marketCap"))
        if observed_mktcap is None:
            # fallback: shares_outstanding * currentPrice (if both exist)
            shares = _safe_float(info.get("sharesOutstanding"))
            price = _safe_float(info.get("currentPrice"))
            if _is_pos(shares) and _is_pos(price):
                observed_mktcap = shares * price
                notes.append("MarketCap estimated from sharesOutstanding × currentPrice.")
    except Exception as e:
        observed_mktcap = None
        notes.append(f"Error retrieving market cap: {str(e)[:50]}.")

    # 4) Premium calculation (MarketCap vs Implied Equity)
    if not (_is_num(equity_implied) and equity_implied > 0 and _is_num(observed_mktcap)):
        premium_pct = None
        notes.append("Cannot compute premium: missing or invalid Equity_Implied / MarketCap.")
    else:
        premium_pct = (observed_mktcap / equity_implied - 1.0) * 100.0
        # brief interpretation
        if abs(premium_pct) < 5:
            notes.append("Observed Market Cap ≈ Implied Equity (within 5%).")
        elif premium_pct > 20:
            notes.append("Market Cap materially above Implied Equity (>20% premium).")
        elif premium_pct > 0:
            notes.append("Market Cap modestly above Implied Equity (positive premium).")
        elif premium_pct < -20:
            notes.append("Market Cap materially below Implied Equity (<-20% premium).")
        else:
            notes.append("Market Cap modestly below Implied Equity (negative premium).")

    out = pd.DataFrame([{
        "Ticker": t,
        "Observed_MarketCap": float(observed_mktcap) if _is_num(observed_mktcap) else None,
        "Equity_Implied": float(equity_implied) if _is_num(equity_implied) else None,
        "Premium_%": float(premium_pct) if _is_num(premium_pct) else None,  # moved earlier
        "EV_Implied": float(ev_implied) if _is_num(ev_implied) else None,
        "NetDebt": float(net_debt),
        "CashAndCashEquivalents": float(cash_eq),
        "MinorityInterest": float(minority),
        "Avg_FCF_Used": float(avg_fcf) if _is_num(avg_fcf) else None,
        "Growth_Used": float(g_used) if _is_num(g_used) else None,
        "WACC_Used": float(wacc_used) if _is_num(wacc_used) else None,
        "Years": int(years_used),
        "Assumptions_Used": assumptions_used,
        "Notes": " ".join([n for n in notes if n]).strip()
    }])

    return out if as_df else to_records(
        out, analysis_report_date=analysis_report_date, notes=[n for n in notes if n]
    )

def make_peer_metric_frame(res_payload: Dict[str, Any], metric: str) -> pd.DataFrame:
    """
    Build a simple ['ticker', metric] DataFrame from estimate_company_value(...) output.
    metric ∈ {'PE_Valuation_per_share','PS_Valuation_per_share','EV_EBITDA_Valuation_per_share'}
    """
    df = res_payload.get("peer_comp_detail")
    if not isinstance(df, pd.DataFrame) or df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["ticker", metric])
    out = df[["ticker", metric]].dropna().copy()
    out.rename(columns={metric: metric}, inplace=True)
    return out

# ---- Backward-compatible all-in-one valuation (keeps server.py working) ----
def estimate_company_value(*args, **kwargs):
    """Deprecated alias. Use orchestrator_function(...)."""
    return orchestrator_function(*args, **kwargs)

def _call_with_supported_kwargs(func, *pargs, **pkwargs):
    """Call `func` with only kwargs it supports (future-proof across revs)."""
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in pkwargs.items() if k in allowed}
        return func(*pargs, **filtered)
    except Exception:
        return func(*pargs)

def _collect_health_block(source: str, x) -> List[Dict[str, Any]]:
    """
    Collect health info from a component result.
    Returns a list of dict blocks:
      {"source": str, "ticker": str|None, "data_incomplete": bool|None, "notes": [..]}
    """
    blocks = []

    if isinstance(x, pd.DataFrame) and not x.empty:
        # Multi-ticker case: emit one block per ticker if Ticker column present
        if "Ticker" in x.columns:
            for _, row in x.iterrows():
                blocks.append({
                    "source": source,
                    "ticker": row.get("Ticker"),
                    "data_incomplete": bool(row["data_incomplete"]) if "data_incomplete" in x.columns else None,
                    "notes": [str(row["Notes"]).strip()] if "Notes" in x.columns and pd.notna(row["Notes"]) else []
                })
        else:
            # Single DF, no Ticker column: one block
            notes = []
            if "Notes" in x.columns:
                vals = [str(v).strip() for v in x["Notes"].dropna().tolist() if str(v).strip()]
                notes = sorted(set(vals))
            data_incomplete = None
            if "data_incomplete" in x.columns:
                try:
                    data_incomplete = bool(x["data_incomplete"].iloc[0])
                except Exception:
                    pass
            blocks.append({
                "source": source,
                "ticker": None,
                "data_incomplete": data_incomplete,
                "notes": notes
            })

    elif isinstance(x, dict):
        data_incomplete = x.get("data_incomplete")
        notes = []
        if "Notes" in x and x["Notes"]:
            notes = [str(x["Notes"]).strip()]
        for v in x.values():
            if isinstance(v, pd.DataFrame) and "Notes" in v.columns:
                vals = [str(vv).strip() for vv in v["Notes"].dropna().tolist() if str(vv).strip()]
                notes.extend(vals)
        blocks.append({
            "source": source,
            "ticker": None,
            "data_incomplete": data_incomplete,
            "notes": sorted(set(notes))
        })

    return blocks

def orchestrator_function(
    target_ticker: str,
    peer_tickers: List[str],
    *,
    include_target_in_peers: bool = False,
    years: int = 5,
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    growth: Optional[float] = None,
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    use_average_fcf_years: Optional[int] = VALUATION_DEFAULTS["fcf_window_years"],
    volatility_threshold: float = 0.5,
    assumptions_as_of: Optional[str] = None,
    basis: Literal["annual","ttm"] = "annual",
    save_csv: bool = False,
    output_dir: str = "output",
    analysis_report_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chart-free orchestrator running:
      - historical_average_share_prices
      - historical_growth_metrics
      - compute_fundamentals_actuals + compute_fundamentals_scores
      - peer_multiples (+/- target depending on include_target_in_peers)
      - price_from_peer_multiples
      - compare_to_market_ev
      - compare_to_market_cap
      - dcf_three_scenarios

    Produces:
      - convenience price anchors
      - component outputs
      - data_health_report: list of per-function (and per-ticker) health blocks
    """

    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(target_ticker)

    # Ensure target always in peer set
    peers_all = sorted({_sanitize_ticker(x) for x in peer_tickers} | {t})
    peers_for_multiples = peers_all if include_target_in_peers else [p for p in peers_all if p != t]

    def _ensure_outdir():
        if save_csv:
            os.makedirs(output_dir, exist_ok=True)
    _ensure_outdir()

    def _save(name: str, df_or_dict):
        if not save_csv:
            return
        try:
            if isinstance(df_or_dict, pd.DataFrame):
                df_or_dict.to_csv(os.path.join(output_dir, f"{name}_{t}_{analysis_report_date.replace('-','')}.csv"), index=False)
            else:
                with open(os.path.join(output_dir, f"{name}_{t}_{analysis_report_date.replace('-','')}.json"), "w", encoding="utf-8") as fh:
                    json.dump(df_or_dict, fh, default=str, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---- 1) Price snapshots ----
    prices_df = historical_average_share_prices(t, analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    _save("avg_prices", prices_df)
    avg_price_1d   = float(prices_df.loc[0, "avg_price_1d"])   if "avg_price_1d"   in prices_df.columns else None
    avg_price_30d  = float(prices_df.loc[0, "avg_price_30d"])  if "avg_price_30d"  in prices_df.columns else None
    avg_price_90d  = float(prices_df.loc[0, "avg_price_90d"])  if "avg_price_90d"  in prices_df.columns else None
    avg_price_180d = float(prices_df.loc[0, "avg_price_180d"]) if "avg_price_180d" in prices_df.columns else None
    price_asof     = str(prices_df.loc[0, "price_asof"])       if "price_asof"     in prices_df.columns else None

    # ---- 2) Growth metrics ----
    growth_df = historical_growth_metrics([t], analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    _save("growth_metrics", growth_df)

    # ---- 3) Fundamentals (actuals + scores) ----
    actuals_df = _call_with_supported_kwargs(
        compute_fundamentals_actuals,
        peers_all,
        as_df=True,
        analysis_report_date=analysis_report_date,
    )
    _save("fundamentals_actuals", actuals_df)

    scores_df = _call_with_supported_kwargs(
        compute_fundamentals_scores,
        peers_all,
        as_df=True,
        merge_with_actuals=True,
        basis=basis,
    )
    _save("fundamentals_scores", scores_df)

    # ---- 4) Peer multiples ----
    peers_out = _call_with_supported_kwargs(
        peer_multiples,
        peers_for_multiples,
        target_ticker=t,
        as_df=False,
    )
    _save("peer_multiples", peers_out)

    # ---- 5) Price from peer multiples ----
    price_bands_df = price_from_peer_multiples(peers_out, ticker=t, analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    _save("price_from_peer_multiples", price_bands_df)

    # ---- 6) EV vs Market ----
    ev_vs_market_df = _call_with_supported_kwargs(
        compare_to_market_ev,
        t,
        years=years,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        growth=growth,
        target_cagr_fallback=target_cagr_fallback,
        use_average_fcf_years=use_average_fcf_years,
        volatility_threshold=volatility_threshold,
        assumptions_as_of=assumptions_as_of,
        as_df=True,
        analysis_report_date=analysis_report_date
    )
    _save("ev_vs_market", ev_vs_market_df)

    # ---- 7) Market Cap vs Equity ----
    market_cap_vs_equity_df = _call_with_supported_kwargs(
        compare_to_market_cap,
        ev_vs_market_df,
        as_df=True,
        analysis_report_date=analysis_report_date
    )
    _save("market_cap_vs_equity", market_cap_vs_equity_df)

    # ---- 8) DCF scenarios ----
    dcf_df = _call_with_supported_kwargs(
        dcf_three_scenarios,
        t,
        peer_tickers=peers_for_multiples,
        years=years,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        volatility_threshold=volatility_threshold,
        analysis_report_date=analysis_report_date
    )
    _save("dcf_three_scenarios", dcf_df)

    # ---- Collect health blocks (flattened) ----
    health_blocks: List[Dict[str, Any]] = []
    health_blocks.extend(_collect_health_block("historical_average_share_prices", prices_df))
    health_blocks.extend(_collect_health_block("historical_growth_metrics", growth_df))
    health_blocks.extend(_collect_health_block("compute_fundamentals_actuals", actuals_df))
    health_blocks.extend(_collect_health_block("compute_fundamentals_scores", scores_df))
    health_blocks.extend(_collect_health_block("peer_multiples", peers_out))
    health_blocks.extend(_collect_health_block("price_from_peer_multiples", price_bands_df))
    health_blocks.extend(_collect_health_block("compare_to_market_ev", ev_vs_market_df))
    health_blocks.extend(_collect_health_block("compare_to_market_cap", market_cap_vs_equity_df))
    health_blocks.extend(_collect_health_block("dcf_three_scenarios", dcf_df))

    # ---- Final payload ----
    return {
        "analysis_report_date": analysis_report_date,
        "ticker": t,
        "peer_tickers": peers_all,
        "include_target_in_peers": include_target_in_peers,
        "avg_price_1d": avg_price_1d,
        "avg_price_30d": avg_price_30d,
        "avg_price_90d": avg_price_90d,
        "avg_price_180d": avg_price_180d,
        "price_asof": price_asof,
        "growth_metrics": growth_df,
        "fundamentals_actuals": actuals_df,
        "fundamentals_scores": scores_df,
        "peer_multiples": peers_out,
        "price_from_peer_multiples": price_bands_df,
        "ev_vs_market": ev_vs_market_df,
        "market_cap_vs_equity": market_cap_vs_equity_df,
        "dcf_scenarios": dcf_df,
        "data_health_report": health_blocks,
    }

# VISUALISATION FUNCTIONS

def _fmt_billions(x):
    try:
        return f"{x/1e9:.1f}B"
    except Exception:
        return str(x)

def plot_peer_metric_boxplot(
    peer_comp_detail: pd.DataFrame,
    peer_multiple_bands_wide: pd.DataFrame,
    *,
    metric: str,                # "EV_EBITDA" | "PE" | "PS"
    target_ticker: str,
    include_target_in_stats: bool = False,
    save_path: str | None = None,
):
    """
    Boxplot of peer distribution (ratios) with target overlay using `peer_comp_detail`
    and `peer_multiple_bands_wide`. Returns (fig, ax). If `save_path` is provided,
    also writes a PNG to that path.

    peer_comp_detail: DataFrame containing 'ticker' and raw ratio columns:
        pe_ratio, ps_ratio, ev_to_ebitda
    peer_multiple_bands_wide: DataFrame with index {PE,PS,EV_EBITDA} and columns
        [Min,P25,Median,P75,Max,Average] (computed per your peer_multiples).

    The plotted distribution will EXCLUDE the target if include_target_in_stats=False
    (to match how bands_wide values was computed).
    
    However note that the target will still overlaid as a scatter point data marker along side the peer min-percentile-max ranges even if excluded from the peer stats 

    """

    ratio_col_map = {
        "PE": "pe_ratio",
        "PS": "ps_ratio",
        "EV_EBITDA": "ev_to_ebitda",
    }
    if metric not in ratio_col_map:
        raise ValueError(f"metric must be one of {list(ratio_col_map)}")

    if "ticker" not in peer_comp_detail.columns:
        raise KeyError("peer_comp_detail must have a 'ticker' column.")

    ratio_col = ratio_col_map[metric]

    # --- Build the peer-only series for plotting ---
    df = peer_comp_detail.copy()
    df["ticker_norm"] = df["ticker"].astype(str).str.upper()
    tgt = str(target_ticker).upper()

    if not include_target_in_stats:
        plot_df = df[df["ticker_norm"] != tgt]
    else:
        plot_df = df  # optionally include target

    series = pd.to_numeric(plot_df[ratio_col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No numeric data for peers (metric={metric}).")

    # --- Get target point (even if excluded from plot series) ---
    trow = df.loc[df["ticker_norm"] == tgt, ratio_col]
    target_val = float(pd.to_numeric(trow, errors="coerce").dropna().iloc[0]) if not trow.empty else None

    # --- Pull stats from bands_wide for title (if present), else compute from series ---
    stats = None
    if isinstance(peer_multiple_bands_wide.index, pd.Index) and metric in peer_multiple_bands_wide.index:
        row = peer_multiple_bands_wide.loc[metric]
        if set(["Min","P25","Median","P75","Max"]).issubset(row.index):
            stats = dict(Min=row["Min"], P25=row["P25"], Median=row["Median"], P75=row["P75"], Max=row["Max"])

    if stats is None:
        stats = {
            "Min": float(np.nanmin(series.values)),
            "P25": float(np.nanpercentile(series.values, 25)),
            "Median": float(np.nanpercentile(series.values, 50)),
            "P75": float(np.nanpercentile(series.values, 75)),
            "Max": float(np.nanmax(series.values)),
        }

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.boxplot(series.values, vert=True, widths=0.5, patch_artist=False)
    ax.axhline(stats["Median"], linestyle="--", linewidth=1, alpha=0.6)

    # overlay target point even if excluded
    if target_val is not None:
        ax.scatter([1.05], [target_val], s=60, zorder=3)
        ax.annotate(
            f"{target_ticker} {target_val:.2f}",
            xy=(1.05, target_val),
            xytext=(1.10, target_val),
            textcoords="data",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.8),
        )

    ax.set_title(
        f"{metric} — Peer Distribution (min/P25/Med/P75/max = "
        f"{stats['Min']:.2f}/{stats['P25']:.2f}/{stats['Median']:.2f}/{stats['P75']:.2f}/{stats['Max']:.2f})",
        fontsize=11
    )
    ax.set_ylabel(metric)
    ax.set_xticks([1])
    ax.set_xticklabels([metric])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

def plot_ev_observed_vs_implied(ev_vs_market_df, *, save_path: str = None):
    """
    ev_vs_market_df: output from compare_to_market_ev(...), single-row DF expected.
    Produces a 2-bar chart: Observed EV vs Implied EV, with Δ and premium% annotated.
    """
    row = ev_vs_market_df.iloc[0]
    obs = row.get("Observed_EV"); imp = row.get("EV_Implied")
    if not (np.isfinite(obs) and np.isfinite(imp)):
        raise ValueError("Observed_EV or EV_Implied missing.")

    delta = obs - imp
    premium_pct = (obs / imp - 1) * 100 if imp else np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Observed EV", "Implied EV"], [obs, imp])
    ax.set_ylabel("Enterprise Value")
    ax.set_title("Observed vs Implied Enterprise Value")

    # Human-readable labels
    for x, y in zip([0, 1], [obs, imp]):
        ax.text(x, y, _fmt_billions(y), ha="center", va="bottom")

    ax.text(0.5, max(obs, imp)*1.03, f"Δ = {_fmt_billions(delta)}  ({premium_pct:.1f}%)",
            ha="center", va="bottom")

    fig.tight_layout()
    if save_path: fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

def plot_market_cap_observed_vs_implied_equity_val(mktcap_vs_equity_df: pd.DataFrame, *, save_path: str = None):
    """
    Input: output from compare_to_market_cap(...), single-row DF expected.
    Produces a 2-bar chart: Observed Market Cap vs Implied Equity Value, with Δ and premium% annotated.
    """
    if not isinstance(mktcap_vs_equity_df, pd.DataFrame) or mktcap_vs_equity_df.empty:
        raise ValueError("Expected a single-row DataFrame from compare_to_market_cap().")
    row = mktcap_vs_equity_df.iloc[0]

    obs = row.get("Observed_MarketCap")
    imp = row.get("Equity_Implied")
    if not (np.isfinite(obs) and np.isfinite(imp)):
        raise ValueError("Observed_MarketCap or Equity_Implied missing.")

    delta = obs - imp
    premium_pct = (obs / imp - 1) * 100 if imp else np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Observed Market Cap", "Implied Equity"], [obs, imp])
    ax.set_ylabel("Equity Value")
    ax.set_title("Observed Market Capitalization vs Implied Equity Value")

    for x, y in zip([0, 1], [obs, imp]):
        ax.text(x, y, _fmt_billions(y), ha="center", va="bottom")

    ax.text(0.5, max(obs, imp) * 1.03, f"Δ = {_fmt_billions(delta)}  ({premium_pct:.1f}%)",
            ha="center", va="bottom")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

def plot_dcf_scenarios_vs_price(
    audit_df,
    current_price: float,
    *,
    save_path: str = None,
    horizontal: bool = False,       # set True if you prefer a horizontal bar chart
    max_label_len: int = 18,        # truncate labels longer than this with …
    rotation: int = 30,             # x-tick rotation (ignored when horizontal=True)
    bar_annot_fmt: str = "{:.2f}"   # annotation format for bar values
):
    """
    Plot DCF scenarios vs current price with smart labels.

    Upstream: Typically use dcf_three_scenarios(...) for audit_df and _price_snapshots(...) for current_price

    audit_df: 3-scenario table with columns ['Scenario', <per-share column>].
              Accepts 'Per_Share_Value' (your code) or 'Intrinsic_per_share'.
    current_price: latest price to draw as a dashed reference line.

    Options:
      - horizontal=True  → use a horizontal bar chart (no rotation needed).
      - max_label_len    → truncate long labels with an ellipsis.
      - rotation         → angle for x-labels (if not horizontal)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    import textwrap

    if "Scenario" not in audit_df.columns:
        raise KeyError("'Scenario' column missing in audit_df.")

    # Accept multiple column spellings and fall back to a heuristic
    candidates = ["Per_Share_Value", "Intrinsic_per_share"]
    per_col = next((c for c in candidates if c in audit_df.columns), None)
    if per_col is None:
        per_cols = [c for c in audit_df.columns if "per" in c.lower() and "share" in c.lower()]
        if not per_cols:
            raise KeyError("No per-share column found. Expected one of: 'Per_Share_Value', 'Intrinsic_per_share'.")
        per_col = per_cols[0]

    dfp = audit_df[["Scenario", per_col]].copy()
    dfp.rename(columns={per_col: "Per_Share_Value"}, inplace=True)

    # ---------- label cleanup ----------
    def _clean_label(s: str) -> str:
        s = str(s)
        # compress common prefixes/suffixes from your codebase
        s = s.replace("DCF_", "")
        s = s.replace("Ave_Actual_", "").replace("Average_", "")
        s = s.replace("_Growth", "").replace("_WACC", "")
        s = s.replace("Price_1d", "Price_1d").replace("Price_30d", "Price_30d").replace("Price_90d", "Price_90d")
        # optional: Lo_Growth_Hi -> Lo_Hi ; Mid_Growth_Mid -> Mid_Mid (already removed _Growth above)
        s = re.sub(r"(^|_)Lo(_|$)", "Lo", s)
        s = re.sub(r"(^|_)Mid(_|$)", "Mid", s)
        s = re.sub(r"(^|_)Hi(_|$)", "Hi", s)
        # collapse double underscores that might result from replacements
        s = re.sub(r"__+", "_", s).strip("_")
        # truncate if too long
        if max_label_len and len(s) > max_label_len:
            s = s[:max_label_len - 1] + "…"
        return s

    labels = dfp["Scenario"].astype(str).map(_clean_label).tolist()
    y = dfp["Per_Share_Value"].astype(float).values
    n = len(labels)

    # wide enough figure for number of bars
    fig_w = max(6.0, 0.9 * n) if not horizontal else max(6.5, 0.7 * n)
    fig_h = 4.0 if not horizontal else max(3.5, 0.5 * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if horizontal:
        # horizontal bars
        y_pos = np.arange(n)
        ax.barh(y_pos, y)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.axvline(float(current_price), linestyle="--")
        ax.set_xlabel("Intrinsic Value per Share")
        ax.set_title("DCF Scenarios vs Current Price")
        # annotations at bar ends
        for yi, xi in zip(y_pos, y):
            ax.text(xi, yi, " " + bar_annot_fmt.format(xi), va="center", ha="left")
        # price label
        ax.text(0.02, 0.95, f"Price ≈ {float(current_price):.2f}", transform=ax.transAxes)
    else:
        # vertical bars
        x = np.arange(n)
        bars = ax.bar(x, y)  # Use x positions instead of labels
        ax.axhline(float(current_price), linestyle="--")
        ax.set_ylabel("Intrinsic Value per Share")
        ax.set_title("DCF Scenarios vs Current Price")

        # FIXED: Set ticks first, then labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")

        # annotate bar tops
        for xi, yi in zip(x, y):
            ax.text(xi, yi, bar_annot_fmt.format(yi), ha="center", va="bottom")

        # price label
        ax.text(0.02, 0.95, f"Price ≈ {float(current_price):.2f}", transform=ax.transAxes)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

def plot_metrics_multiples(df, ticker_col: str = "ticker", highlight: str = "total_score", max_cols: int = 3):
    """
    Plot small multiples of bar charts for each numeric column in df (excluding ticker_col).
    - Works for both scores_df and actuals_df.
    - If 'highlight' column is present (e.g., total_score), it's drawn first and highlighted.
    """
    # Select numeric columns only
    numeric_cols = [c for c in df.columns if c != ticker_col and pd.api.types.is_numeric_dtype(df[c])]
    
    # Move highlight column (if present) to front
    if highlight in numeric_cols:
        numeric_cols.remove(highlight)
        numeric_cols = [highlight] + numeric_cols

    n = len(numeric_cols)
    ncols = max_cols
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        vals = df[col].astype(float)
        ax.bar(df[ticker_col], vals, color="steelblue")

        # Highlight special column if requested
        if col == highlight:
            for bar in ax.patches:
                bar.set_color("darkorange")

        ax.set_title(col, fontsize=10 if col != highlight else 12, fontweight="bold")
        ax.set_ylim(0, max(vals.dropna()) * 1.2 if len(vals.dropna()) > 0 else 1)

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metrics Multiples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes

def plot_scores_clustered(
    scores_or_tickers: Union[pd.DataFrame, List[str]],
    *,
    basis: Literal["annual","ttm"] = "annual",
    metrics: Optional[List[str]] = None,     # can include per-metric score_* and/or rollups like 'profitability_score'
    include_total: bool = True,              # include total_score as one more bar group when not explicitly filtered out
    sort_by: Literal["family","avg","name","none"] = "family",
    figsize: tuple = (12, 6),
    bar_width: Optional[float] = None,       # None -> auto (0.8 / #tickers)
    title: Optional[str] = None,
):
    """
    Clustered bar chart of fundamentals scores for multiple tickers.

    Accepts either:
      - a scores DataFrame from compute_fundamentals_scores(..., merge_with_actuals=False), or
      - a list of tickers (then computes actuals->scores for you).

    'metrics' can be a mix of:
      - per-metric columns: e.g. 'score_Profitability-ROE-TTM'
      - factor rollups: 'profitability_score', 'growth_score', 'reinvestment_score', 'risk_score'
      - optionally 'total_score'
    """
    # 1) Build scores_df
    if isinstance(scores_or_tickers, pd.DataFrame):
        scores_df = scores_or_tickers.copy()
    else:
        tickers = list(scores_or_tickers)
        actuals = compute_fundamentals_actuals(tickers, basis=basis, save_csv=False, as_df=True)
        scores_df = compute_fundamentals_scores(actuals, basis=basis, merge_with_actuals=False, as_df=True)

    if "ticker" not in scores_df.columns:
        raise ValueError("scores_df must contain a 'ticker' column")

    # Available columns
    per_metric_cols = [c for c in scores_df.columns if isinstance(c, str) and c.startswith("score_")]
    rollup_all = ["profitability_score","growth_score","reinvestment_score","risk_score","total_score"]
    rollup_cols = [c for c in rollup_all if c in scores_df.columns]

    # 2) Choose which columns to plot
    if metrics:
        # Normalize metrics list:
        requested = []
        for m in metrics:
            # allow passing metric names without 'score_' for per-metric columns? keep as-is;
            # users should specify exact per-metric names for those.
            requested.append(m)
        # If include_total=True but not requested explicitly and no other selection excludes it,
        # we’ll add it at the end (only if present).
        selected_cols = []
        # First, add any requested per-metric score_* columns that exist
        selected_cols += [c for c in requested if c in per_metric_cols]
        # Then, add any requested rollups that exist
        selected_cols += [c for c in requested if c in rollup_cols]
        # If nothing matched, fail early with a friendly error
        if not selected_cols:
            raise ValueError(
                "None of the requested 'metrics' were found. "
                "You can pass per-metric columns like 'score_Profitability-ROE-TTM' "
                "and/or rollups like 'profitability_score', 'growth_score', 'reinvestment_score', 'risk_score', 'total_score'."
            )
        # Optionally append total_score if present and requested via include_total and not already included
        if include_total and "total_score" in rollup_cols and "total_score" not in selected_cols:
            selected_cols.append("total_score")
    else:
        # No explicit metrics: plot ALL per-metric columns + (optional) total_score
        selected_cols = per_metric_cols.copy()
        if include_total and "total_score" in rollup_cols:
            selected_cols.append("total_score")

    # 3) Build tidy long frame
    use_cols = ["ticker"] + selected_cols
    long_df = scores_df[use_cols].melt(id_vars="ticker", var_name="Metric", value_name="Score")

    # Pretty labels: strip 'score_' prefix only for per-metric columns
    def _pretty(lbl: str) -> str:
        if lbl.startswith("score_"):
            return lbl.replace("score_","",1)
        # Make rollups nicer case
        if lbl == "profitability_score": return "Profitability"
        if lbl == "growth_score": return "Growth"
        if lbl == "reinvestment_score": return "Reinvestment"
        if lbl == "risk_score": return "Risk"
        if lbl == "total_score": return "Total"
        return lbl
    long_df["MetricLabel"] = long_df["Metric"].map(_pretty)

    # If all scores are NaN, nothing to plot
    if long_df["Score"].dropna().empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No score data to plot", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    # 4) Order metrics

    # --- NEW: group metrics by factor family and create an ordered category for the x-axis ---

    # Map each metric to a family
    FAMILY_ORDER = ["Profitability", "Growth", "Reinvestment", "Risk", "Total"]

    METRIC_FAMILY = {
        # per-metric
        "score_roe": "Profitability",
        "score_profit_margin": "Profitability",
        "score_op_margin": "Profitability",
        "score_roa": "Profitability",

        "score_revenue_cagr": "Growth",
        "score_earnings_cagr": "Growth",
        "score_peg": "Growth",

        "score_reinvestment_rate": "Reinvestment",
        "score_capex_ratio": "Reinvestment",

        "score_de_ratio": "Risk",
        "score_beta": "Risk",
        "score_current_ratio": "Risk",

        # rollups
        "profitability_score": "Profitability",
        "growth_score": "Growth",
        "reinvestment_score": "Reinvestment",
        "risk_score": "Risk",
        "total_score": "Total",
    }

    # A stable within-family order (tweak as you like)
    WITHIN_FAMILY_ORDER = {
        "Profitability": ["score_roe", "score_op_margin", "score_profit_margin", "score_roa", "profitability_score"],
        "Growth":        ["score_revenue_cagr", "score_earnings_cagr", "score_peg", "growth_score"],
        "Reinvestment":  ["score_reinvestment_rate", "score_capex_ratio", "reinvestment_score"],
        "Risk":          ["score_de_ratio", "score_beta", "score_current_ratio", "risk_score"],
        "Total":         ["total_score"],
    }

    # Attach family + within-family rank to each melted row
    long_df["Family"] = long_df["Metric"].map(METRIC_FAMILY).fillna("Other")

    def metric_rank_key(metric: str) -> Tuple[int, int]:
        fam = METRIC_FAMILY.get(metric, "Other")
        fam_idx = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        within = WITHIN_FAMILY_ORDER.get(fam, [])
        try:
            within_idx = within.index(metric)
        except ValueError:
            within_idx = 999
        return fam_idx, within_idx

    long_df["__fam_idx__"], long_df["__within_idx__"] = zip(*long_df["Metric"].map(metric_rank_key))

    # Decide final order
    if sort_by == "family":
        # Primary: family order; Secondary: within-family order; Tertiary: metric label as fallback
        long_df = long_df.sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])
    elif sort_by == "metric":
        long_df = long_df.sort_values(["MetricLabel"])
    elif sort_by == "ticker":
        long_df = long_df.sort_values(["ticker", "MetricLabel"])
    elif sort_by == "avg":
        # Compute metric means and order by descending average
        means = long_df.groupby("Metric", observed=False)["Score"].mean().sort_values(ascending=False)
        long_df["__avg_order__"] = long_df["Metric"].map(means.to_dict())
        long_df = long_df.sort_values(["__avg_order__", "MetricLabel"])
    else:
        # default to family if unknown
        long_df = long_df.sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])

    # Lock x-axis order via an ordered Categorical on MetricLabel
    ordered_labels = (
        long_df
        .drop_duplicates(subset=["Metric","MetricLabel"])
        .sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])["MetricLabel"]
        .tolist()
    )
    long_df["MetricLabel"] = pd.Categorical(long_df["MetricLabel"], categories=ordered_labels, ordered=True)

    # 5) Pivot to rows=MetricLabel, cols=ticker
    pivot = (
        long_df.pivot_table(index="MetricLabel", columns="ticker", values="Score", aggfunc="first", observed=False)
        .reindex(index=ordered_labels)  # ensure the axis respects our order
    )

    # If pivot is empty, bail gracefully
    if pivot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No score data to plot", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    # 6) Plot
    tickers = pivot.columns.tolist()
    n_groups = pivot.shape[0]
    n_series = len(tickers)
    x = np.arange(n_groups)

    if bar_width is None:
        bar_width = 0.8 / max(1, n_series)

    fig, ax = plt.subplots(figsize=figsize)

    drew_any = False
    for i, tkr in enumerate(tickers):
        y = pivot[tkr].astype(float).values
        ax.bar(x + i*bar_width - (n_series-1)*bar_width/2, y, width=bar_width, label=tkr)
        if np.isfinite(y).any():
            drew_any = True

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("Score (0–5)")
    ttl = title or f"Fundamentals Scores — basis={basis}"
    ax.set_title(ttl)
    ax.grid(axis="y", alpha=0.2)

    # Only add legend if we drew something
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Ticker", ncol=min(len(tickers), 4), fontsize=8)

    plt.tight_layout()
    return fig, ax

# Internal adapter: fetch a tidy time series for a ticker/family/metrics
def _fetch_timeseries_for_plot(
    ticker: str,
    *,
    family: str,                     # "Profitability" | "Liquidity" | "Growth-basic" | "Reinvestment-basic"
    metrics: list[str] | None = None,
    basis: str = "annual",           # "annual" or "ttm"
    include_ttm: bool | None = None, # auto from basis if None
    return_long: bool = True,
) -> pd.DataFrame:
    """
    Returns tidy long df: ['Period','Metric','Value'] for the requested family/metrics.
    Uses your existing compute_*_timeseries() for Profitability/Liquidity.
    For 'Growth-basic': computes YoY growth for Revenue/Net Income from statements.
    For 'Reinvestment-basic': builds per-period CapexRatio and ReinvestmentRate from statements.
    """

    include_ttm = (basis == "ttm") if include_ttm is None else include_ttm

    if family.lower().startswith("profit"):
        df = compute_profitability_timeseries(
            ticker, include_ttm=include_ttm, as_df=True, return_format="long"
        )
        # Metric names in this long frame are: ROE, NetMargin, OpMargin, ROA
        keep = metrics or ["ROE","NetMargin","OpMargin","ROA"]
        df = df[df["Metric"].isin(keep)][["Period","Metric","Value"]].copy()
        return df.sort_values(["Metric","Period"])

    if family.lower().startswith("liquid"):
        df = compute_liquidity_timeseries(
            ticker, include_ttm=include_ttm, as_df=True, return_format="long"
        )
        # Metric name: CurrentRatio
        keep = metrics or ["CurrentRatio"]
        df = df[df["Metric"].isin(keep)][["Period","Metric","Value"]].copy()
        return df.sort_values(["Metric","Period"])

    # --- Growth-basic: YoY (% as decimal) for Revenue and Net Income ---
    if family.lower().startswith("growth"):
        tkr = yf.Ticker(ticker)
        IS_a = _annual_sorted(tkr.financials)
        IS_q = tkr.quarterly_financials

        def _yoy_from_series(s: pd.Series) -> pd.DataFrame:
            if s is None or s.empty: 
                return pd.DataFrame(columns=["Period","Metric","Value"])
            ser = s.dropna().astype(float)
            # chronological order oldest->newest
            years = [int(pd.to_datetime(c).year) for c in ser.index]
            order = np.argsort(years)
            ser = ser.iloc[order]
            yoy = ser.pct_change() # decimal
            out = pd.DataFrame({
                "Period": [str(int(pd.to_datetime(i).year)) for i in yoy.index],
                "Value": yoy.values
            })
            return out

        frames = []

        # Annual YoY
        rev_a = _alias_first(IS_a, ["Total Revenue","Revenue"])
        ni_a  = _alias_first(IS_a, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])
        if isinstance(rev_a, pd.Series):
            f = _yoy_from_series(rev_a); f["Metric"]="RevenueYoY"; frames.append(f)
        if isinstance(ni_a, pd.Series):
            f = _yoy_from_series(ni_a);  f["Metric"]="EarningsYoY"; frames.append(f)

        # TTM YoY (optional)
        if include_ttm:
            def _ttm_total(s: pd.Series) -> float | None:
                if s is None or s.empty: return None
                ser = s.dropna().astype(float)
                if len(ser) < 4: return None
                return float(ser.iloc[:4].sum())
            # last 4 quarters vs the 4 before that
            rev_q = _alias_first(IS_q, ["Total Revenue","Revenue"])
            ni_q  = _alias_first(IS_q, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])
            for s, metric in [(rev_q,"RevenueYoY_TTM"), (ni_q,"EarningsYoY_TTM")]:
                if s is not None and len(s.dropna()) >= 8:
                    ser = s.dropna().astype(float)
                    ttm_now = float(ser.iloc[:4].sum())
                    ttm_prev= float(ser.iloc[4:8].sum())
                    val = (ttm_now/ttm_prev - 1.0) if (ttm_prev not in (None,0)) else None
                    frames.append(pd.DataFrame({"Period":["TTM"],"Metric":[metric],"Value":[val]}))

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Period","Metric","Value"])
        if metrics:
            out = out[out["Metric"].isin(metrics)]
        return out

    # --- Reinvestment-basic: per-period CapexRatio & ReinvestmentRate (annual + optional TTM) ---
    if family.lower().startswith("reinvest"):
        tkr = yf.Ticker(ticker)
        IS_a = _annual_sorted(tkr.financials)
        CF_a = _annual_sorted(tkr.cashflow)

        rows = []
        # Annuals
        if isinstance(IS_a, pd.DataFrame) and isinstance(CF_a, pd.DataFrame):
            s_rev   = _alias_first(IS_a, ["Total Revenue","Revenue"])
            s_ni    = _alias_first(IS_a, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])
            s_div   = _alias_first(CF_a, ["Common Stock Dividend Paid"])
            s_capex = _alias_first(CF_a, ["Capital Expenditure","Capital Expenditures"])

            # align by columns index intersection
            if s_rev is not None and s_capex is not None:
                cols = [c for c in s_rev.index if c in s_capex.index]
                for c in cols:
                    year = str(int(pd.to_datetime(c).year))
                    rev   = _safe_float(s_rev.loc[c])
                    capex = _safe_float(s_capex.loc[c])
                    if (rev is not None and math.isfinite(rev) and float(rev) > 0) and (capex is not None and math.isfinite(capex)):
                        rows.append({"Period":year, "Metric":"CapexRatio", "Value": abs(float(capex))/float(rev)})
            if s_ni is not None:
                cols = list(s_ni.index)
                for c in cols:
                    year = str(int(pd.to_datetime(c).year))
                    ni   = _safe_float(s_ni.loc[c])
                    # coalesce dividends to 0.0; _safe_float may return None
                    div  = 0.0
                    if s_div is not None and c in s_div.index:
                        _d = _safe_float(s_div.loc[c])
                        div = 0.0 if (_d is None or not math.isfinite(_d)) else float(_d)
                    if ni is not None and math.isfinite(ni) and float(ni) != 0.0:
                        rows.append({"Period":year, "Metric":"ReinvestmentRate", "Value": (float(ni) + float(div))/float(ni)})


        # TTM
        if include_ttm:
            IS_q = tkr.quarterly_financials
            CF_q = tkr.quarterly_cashflow
            s_rev_q   = _alias_first(IS_q, ["Total Revenue","Revenue"])
            s_ni_q    = _alias_first(IS_q, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])
            s_div_q   = _alias_first(CF_q, ["Common Stock Dividend Paid"])
            s_capex_q = _alias_first(CF_q, ["Capital Expenditure","Capital Expenditures"])
            def _sum4(s): 
                if s is None: return None
            if (rev_ttm is not None and math.isfinite(rev_ttm) and float(rev_ttm) > 0) and (capex_ttm is not None and math.isfinite(capex_ttm)):
                rows.append({"Period":"TTM", "Metric":"CapexRatio", "Value": abs(float(capex_ttm))/float(rev_ttm)})
            # coalesce div_ttm
            if div_ttm is None or not math.isfinite(div_ttm):
                div_ttm = 0.0
            if ni_ttm is not None and math.isfinite(ni_ttm) and float(ni_ttm) != 0.0:
                rows.append({"Period":"TTM", "Metric":"ReinvestmentRate", "Value": (float(ni_ttm) + float(div_ttm))/float(ni_ttm)})
            capex_ttm = _sum4(s_capex_q)
            if rev_ttm not in (None,0) and capex_ttm is not None:
                rows.append({"Period":"TTM", "Metric":"CapexRatio", "Value": abs(capex_ttm)/rev_ttm})
            if ni_ttm not in (None,0) and div_ttm is not None:
                rows.append({"Period":"TTM", "Metric":"ReinvestmentRate", "Value": (ni_ttm + div_ttm)/ni_ttm})

        out = pd.DataFrame(rows)
        if metrics:
            out = out[out["Metric"].isin(metrics)]
        return out[["Period","Metric","Value"]].sort_values(["Metric","Period"])

    raise ValueError(f"Unsupported family: {family}")


def plot_single_metric_ts(ticker: str, metric: str, *, family: str, basis: str = "annual"):
    """
    (a) Single ticker · single metric over time.
    """
    df = _fetch_timeseries_for_plot(ticker, family=family, metrics=[metric], basis=basis)
    if df.empty:
        print(f"No data for {ticker} / {family} / {metric}")
        return None, None
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(df["Period"], df["Value"], marker="o")
    ax.set_title(f"{ticker} — {metric} ({basis})")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax


def plot_metrics_family_ts(
    ticker: str,
    family: str,
    *,
    metrics: list[str] | None = None,
    basis: str = "annual",
    save_path: str | None = None,
):
    """
    Plots for a single ticker multiple metrics (on a family basis) over time where 
    
    family can be Profitability, Growth, Risk or Reinvestment
    
    metrics reflect the individual elements (ie column names) within the function compute_fundamentals_actuals such as 
    - Profitability-ROE,NetMargin,OpMargin,ROA       
    - Growth-RevenueCAGR,EarningsCAGR
    - Reinvestment-ReinvestmentRate,CapexRatio
    - Risk-DebtEquityRatio,CurrentRatio

    Returns (fig, ax). If `save_path` is provided, also writes a PNG to that path.
    """
    df = _fetch_timeseries_for_plot(ticker, family=family, metrics=metrics, basis=basis)
    if df.empty:
        print(f"No data for {ticker} / {family}")
        return None, None
    fig, ax = plt.subplots(figsize=(7.5,4))
    for m, g in df.groupby("Metric"):
        ax.plot(g["Period"], g["Value"], marker="o", label=m)
    ax.set_title(f"{ticker} — {family} ({basis})")
    ax.set_xlabel("Period")
    ax.set_ylabel("Value")
    ax.legend(title="Metric", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_multi_tickers_multi_metrics_ts(
    tickers: list[str],
    family: str,
    *,
    metrics: list[str] | None = None,
    basis: str = "annual",
    save_path: str | None = None,
):
    """
    Plots multiple tickers for multiple metrics over time for comparison.

    family can be Profitability, Growth, Risk or Reinvestment
    
    metrics reflect the individual elements (ie column names) within the function compute_fundamentals_actuals such as 
    - Profitability-ROE,NetMargin,OpMargin,ROA       
    - Growth-RevenueCAGR,EarningsCAGR
    - Reinvestment-ReinvestmentRate,CapexRatio
    - Risk-DebtEquityRatio,CurrentRatio   

    Returns (fig, axes). If `save_path` is provided, also writes a PNG to that path.
    """
    # Build concatenated tidy frame per ticker
    frames = []
    for t in tickers:
        df = _fetch_timeseries_for_plot(t, family=family, metrics=metrics, basis=basis)
        if not df.empty:
            df = df.copy()
            df["Ticker"] = t
            frames.append(df)
    if not frames:
        print("No data for requested tickers/family/metrics")
        return None, None

    big = pd.concat(frames, ignore_index=True)
    metrics_in_data = list(big["Metric"].unique())
    n = len(metrics_in_data)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.5, max(3, 2.4*n)))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_in_data):
        g = big[big["Metric"] == metric]
        for t, sub in g.groupby("Ticker"):
            ax.plot(sub["Period"], sub["Value"], marker="o", label=t)
        ax.set_title(metric)
        ax.set_xlabel("Period")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes
