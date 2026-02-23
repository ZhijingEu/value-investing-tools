from __future__ import annotations

import os
import json
import math
import warnings
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Literal, Union

import numpy as np
import pandas as pd
import re
import inspect

from vitlib.utils import (
    _today_iso,
    _sanitize_ticker,
    _is_num,
    _is_pos,
    _safe_float,
    _pct_from_info,
    _ensure_dir,
    _provider_ticker,
    _provider_info,
    _provider_financials,
    _provider_cashflow,
    _provider_balance_sheet,
    to_records,
    _fcf_series_from_cashflow,
)
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

DEFAULT_SCORING: Dict[str, Any] = {
    "thresholds": _THRESHOLDS,
    "weights": _WEIGHTS,
    "data_incomplete_threshold": 0.25,
    "recommendation_cutoffs": {
        "total_high": 17,
        "elite_total": 18,
        "elite_min_factor": 4,
        "risk_low": 2.5,
        "reinvest_low": 2.0,
        "avg_pg_low": 2.5,
        "total_mid_low": 14,
        "total_mid_high": 17,
        "risk_mid_low": 2.0,
        "total_low": 11,
    },
}

def _merge_scoring_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return DEFAULT_SCORING
    merged = {
        "thresholds": DEFAULT_SCORING["thresholds"],
        "weights": DEFAULT_SCORING["weights"],
        "data_incomplete_threshold": DEFAULT_SCORING["data_incomplete_threshold"],
        "recommendation_cutoffs": dict(DEFAULT_SCORING["recommendation_cutoffs"]),
    }
    if overrides.get("thresholds") is not None:
        merged["thresholds"] = overrides["thresholds"]
    if overrides.get("weights") is not None:
        merged["weights"] = overrides["weights"]
    if overrides.get("data_incomplete_threshold") is not None:
        merged["data_incomplete_threshold"] = overrides["data_incomplete_threshold"]
    if overrides.get("recommendation_cutoffs") is not None:
        merged["recommendation_cutoffs"].update(overrides["recommendation_cutoffs"])
    return merged

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
    - PEG, Beta: always '-TTM' (point-in-time from provider info)
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
    tkr = _provider_ticker(ticker)
    info = _provider_info(ticker)
    financials = _provider_financials(ticker)
    cashflow = _provider_cashflow(ticker)
    balance = _provider_balance_sheet(ticker)

    years = []
    if isinstance(financials, pd.DataFrame): years = [c for c in financials.columns]
    if isinstance(cashflow, pd.DataFrame):   years = list(set(years) & set(cashflow.columns)) if years else list(cashflow.columns)
    if isinstance(balance, pd.DataFrame):    years = list(set(years) & set(balance.columns)) if years else list(balance.columns)
    period_range = f"{min(years).year}-{max(years).year}" if len(years) else "N/A"
    min_periods = len(years)

    return info, financials, cashflow, balance, period_range, min_periods

def _score_metric(value: Optional[float], metric: str, thresholds: Optional[Dict[str, List[float]]] = None) -> Optional[int]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    use_thresholds = thresholds or _THRESHOLDS
    cuts = use_thresholds[metric]
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

def _strategic_recommendation(row: pd.Series, cutoffs: Dict[str, Any], data_incomplete_threshold: float) -> Tuple[str, str]:
    factors = ['profitability_score', 'growth_score', 'reinvestment_score', 'risk_score']
    factor_scores = {f: row.get(f) for f in factors}
    if any(pd.isna(v) for v in factor_scores.values()):
        return "Inconclusive", "Missing factor scores."

    min_score = min(factor_scores.values())
    avg_pg = (row['profitability_score'] + row['growth_score']) / 2

    missing_count = sum(pd.isna(row[col]) for col in _RAW_METRICS)
    data_incomplete = (missing_count / len(_RAW_METRICS)) > float(data_incomplete_threshold)

    total = row['total_score']
    if data_incomplete:
        return "Inconclusive", "More than 25% of input metrics missing."

    if total >= cutoffs["total_high"]:
        if min_score == 1:
            return "Uneven Fundamentals", "At least one factor score is critically low."
        if factor_scores['risk_score'] < cutoffs["risk_low"]:
            return "Uneven Fundamentals", "Risk score below 2.5."
        if factor_scores['reinvestment_score'] < cutoffs["reinvest_low"]:
            return "Uneven Fundamentals", "Reinvestment score below 2.0."
        if avg_pg < cutoffs["avg_pg_low"]:
            return "Uneven Fundamentals", "Weak profitability + growth despite high total."
        if total >= cutoffs["elite_total"] and all(v >= cutoffs["elite_min_factor"] for v in factor_scores.values()):
            return "Elite Performer", "Exceptional balance across all four factors."
        return "Resilient Core", "Meets threshold with no red flags."

    if cutoffs["total_mid_low"] <= total < cutoffs["total_mid_high"]:
        if factor_scores['risk_score'] < cutoffs["risk_mid_low"] or factor_scores['reinvestment_score'] < cutoffs["reinvest_low"]:
            return "Uneven Fundamentals", "Moderate total but risk/reinvestment weak."
        return "Resilient Core", "Moderate fundamentals with manageable risk."

    if cutoffs["total_low"] <= total < cutoffs["total_mid_low"]:
        return "Uneven Fundamentals", "Mixed fundamentals."

    return "Weak Fundamentals", "Low total score."

def compute_fundamentals_actuals(
    tickers: List[str],
    *,
    output_dir: str = "output",
    save_csv: bool = False,           
    as_df: bool = True,
    basis: Literal["annual","ttm"] = "annual",
    data_incomplete_threshold: Optional[float] = None,
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
                tkr = _provider_ticker(ticker)
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
    threshold = DEFAULT_SCORING["data_incomplete_threshold"] if data_incomplete_threshold is None else float(data_incomplete_threshold)
    df["data_incomplete"] = df[_RAW_METRICS].isna().sum(axis=1) / len(_RAW_METRICS) > threshold
    
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
    basis: Literal["annual","ttm"] = "annual",
    scoring_overrides: Optional[Dict[str, Any]] = None,
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

    scoring = _merge_scoring_overrides(scoring_overrides)
    thresholds = scoring["thresholds"]
    weights = scoring["weights"]

    # Per-metric scores
    for factor, mweights in weights.items():
        for metric in mweights:
            df[f"score_{metric}"] = df[metric].apply(lambda x: _score_metric(x, metric, thresholds))

    # Factor rollups
    for factor, mweights in weights.items():
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
    basis: Literal["annual","ttm"] = "annual",
    scoring_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Convenience orchestrator: actuals → scores → recommendation.
    Returns a single table (actuals + scores + recommendation) by default.
    """
    analysis_report_date = analysis_report_date or _today_iso()
    scoring = _merge_scoring_overrides(scoring_overrides)
    actuals = compute_fundamentals_actuals(
        tickers,
        output_dir=output_dir,
        save_csv=save_csv,
        as_df=True,
        analysis_report_date=analysis_report_date,
        basis=basis,
        data_incomplete_threshold=scoring["data_incomplete_threshold"],
    )
    scored = compute_fundamentals_scores(
        actuals,
        as_df=True,
        merge_with_actuals=True,
        analysis_report_date=analysis_report_date,
        basis=basis,
        scoring_overrides=scoring_overrides,
    ) if include_scores_in_actuals else compute_fundamentals_scores(
        actuals,
        as_df=True,
        merge_with_actuals=False,
        analysis_report_date=analysis_report_date,
        basis=basis,
        scoring_overrides=scoring_overrides,
    )

    df = scored.copy() if include_scores_in_actuals else actuals.merge(scored, on="ticker", how="left")

    # Recommendation, dispersion, data flag
    df["factor_dispersion"] = df[["profitability_score","growth_score","reinvestment_score","risk_score"]].std(axis=1)
    recs = df.apply(
        lambda r: _strategic_recommendation(
            r,
            cutoffs=scoring["recommendation_cutoffs"],
            data_incomplete_threshold=scoring["data_incomplete_threshold"],
        ),
        axis=1,
    )
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
    Index match is exact, case-sensitive to align with provider keys.
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
    # Provider data usually returns most-recent on the left; enforce ascending by column name if they are dates/years
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
    return float(s.iloc[:4].sum())  # quarterly statements usually have most-recent first

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
    tkr = _provider_ticker(ticker)

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
    tkr = _provider_ticker(ticker)
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
    info = _provider_info(ticker) or {}
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
    tkr = _provider_ticker(ticker)
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
    is_annual = _annual_sorted(_provider_financials(ticker))
    s_rev = _alias_first(is_annual, ["Total Revenue","Revenue"])
    s_ni  = _alias_first(is_annual, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"])

    rev_window = _coverage_from_series(s_rev)
    ni_window  = _coverage_from_series(s_ni)

    # Balance sheet for leverage coverage
    bs_annual = _annual_sorted(_provider_balance_sheet(ticker))
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
            yt   = _provider_ticker(t)
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


__all__ = [
    'compute_fundamentals_actuals',
    'compute_fundamentals_scores',
    'full_fundamentals_table',
    'historical_average_share_prices',
    'compute_profitability_timeseries',
    'compute_liquidity_timeseries',
    'fundamentals_ttm_vs_average',
    'evaluate_fundamentals',
    'historical_growth_metrics',
]
