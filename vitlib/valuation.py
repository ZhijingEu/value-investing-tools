from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from vitlib.utils import (
    _today_iso,
    _sanitize_ticker,
    _is_num,
    _is_pos,
    _safe_float,
    _equity_value_from_ev,
    _valuation_confidence_from_flags,
    _fcf_series_from_cashflow,
    to_records,
    valuation_defaults,
    VALUATION_DEFAULTS,
)
from vitlib.peers import _pull_company_snapshot

DEFAULT_GUARDRAILS: Dict[str, Any] = {
    "cost_of_debt_fallback": 0.04,
    "cost_of_debt_min": 0.02,
    "cost_of_debt_max": 0.15,
    "tax_rate_default": 0.21,
    "tax_rate_cap": 0.35,
    "fcf_cagr_bounds": (-0.3, 0.3),
    "rev_cagr_bounds": (-0.3, 0.3),
    "revenue_cagr_haircut": 0.8,
    "growth_floor": -0.05,
    "wacc_spread_low": 0.005,
    "wacc_spread_high": 0.01,
    "scenario_growth_multipliers": (0.6, 1.0, 1.3),
    "ev_fcf_multiple_warn_high": 50.0,
    "ev_fcf_multiple_warn_low": 5.0,
    "premium_band_small": 5.0,
    "premium_band_large": 20.0,
    "cov_moderate": 0.3,
}

def _merge_guardrails(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return dict(DEFAULT_GUARDRAILS)
    merged = dict(DEFAULT_GUARDRAILS)
    for k, v in overrides.items():
        if k in merged and v is not None:
            merged[k] = v
    return merged
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

def _calculate_wacc(
    snap: Dict[str, Any],
    risk_free_rate: float,
    equity_risk_premium: float,
    beta: float,
    guardrails: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
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
        
        guardrails = guardrails or DEFAULT_GUARDRAILS
        cost_of_debt = interest_expense / total_debt if total_debt > 0 else guardrails["cost_of_debt_fallback"]
        cost_of_debt = max(cost_of_debt, guardrails["cost_of_debt_min"])
        cost_of_debt = min(cost_of_debt, guardrails["cost_of_debt_max"])
        
        # Tax rate approximation
        pretax_income = _safe_float(income.loc['Pretax Income'].iloc[0]) if 'Pretax Income' in income.index else None
        net_income = _safe_float(income.loc['Net Income'].iloc[0]) if 'Net Income' in income.index else None
        
        if _is_pos(pretax_income) and _is_num(net_income):
            tax_rate = max(0, (pretax_income - net_income) / pretax_income)
            tax_rate = min(tax_rate, guardrails["tax_rate_cap"])
        else:
            tax_rate = guardrails["tax_rate_default"]
        
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
    assumptions_overrides: Optional[Dict[str, Any]] = None,
    
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
    guardrails = _merge_guardrails(assumptions_overrides)
    assumptions_used = valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=fcf_window_years,
        terminal_growth_gap=assumptions_overrides.get("terminal_growth_gap") if assumptions_overrides else None,
    )
    
    # Initialize notes list FIRST
    notes = []

    beta = _safe_float(snap["beta"])
    if beta is None:
        raise ValueError(f"Missing beta from yfinance for {t}. Cannot compute WACC.")

    # Calculate proper WACC
    wacc_mid = _calculate_wacc(snap, risk_free_rate, equity_risk_premium, beta, guardrails=guardrails)
    if wacc_mid is None:
        raise ValueError(f"Cannot calculate WACC for {t}. Missing required financial data.")

    # CORRECTED: Higher growth should have higher WACC (more risk)
    wacc_low = wacc_mid - float(guardrails["wacc_spread_low"])   # Low growth = lower risk = lower WACC
    wacc_high = wacc_mid + float(guardrails["wacc_spread_high"]) # High growth = higher risk = higher WACC

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
                    lo_b, hi_b = guardrails["fcf_cagr_bounds"]
                    if _is_num(c) and lo_b <= c <= hi_b:
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
            lo_b, hi_b = guardrails["fcf_cagr_bounds"]
            if _is_num(fcf_cagr_target) and lo_b <= fcf_cagr_target <= hi_b:
                base = fcf_cagr_target
                notes.append("Growth rates based on target FCF CAGR")
            else:
                r_lo_b, r_hi_b = guardrails["rev_cagr_bounds"]
                if _is_num(rev_cagr_target) and r_lo_b <= rev_cagr_target <= r_hi_b:
                    base = rev_cagr_target * float(guardrails["revenue_cagr_haircut"])
                    notes.append("Growth rates based on target revenue CAGR (adjusted)")
                else:
                    base = target_cagr_fallback
                    notes.append("Growth rates using fallback assumption")
            m_low, m_mid, m_high = guardrails["scenario_growth_multipliers"]
            g_low, g_mid, g_high = base * m_low, base * m_mid, base * m_high

    # Cap growth if it exceeds WACC
    def _cap(g, w): 
        if not _is_num(g) or not _is_num(w):
            return None
        gap = assumptions_used["terminal_growth_gap"]
        floor = float(guardrails["growth_floor"])
        return min(max(g, floor), w - gap)
    
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
        if ev_fcf_multiple > float(guardrails["ev_fcf_multiple_warn_high"]):
            notes.append(f"Warning: EV/FCF multiple of {ev_fcf_multiple:.1f}x seems high; check assumptions.")
        elif ev_fcf_multiple < float(guardrails["ev_fcf_multiple_warn_low"]):
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


def _dcf_sensitivity_grid_from_inputs(
    *,
    avg_fcf: float,
    shares_outstanding: float,
    years: int,
    wacc_values: List[float],
    growth_values: List[float],
    terminal_growth_gap: float = VALUATION_DEFAULTS["terminal_growth_gap"],
    growth_floor: float = DEFAULT_GUARDRAILS["growth_floor"],
) -> pd.DataFrame:
    """
    Pure helper for DCF sensitivity tables.

    Returns a long-form DataFrame with one row per (growth, wacc) cell.
    Growth is floored at -5% and capped at (wacc - terminal_growth_gap) for stability.
    """
    rows: List[Dict[str, Any]] = []

    for g_in in growth_values:
        for w_in in wacc_values:
            g_used = None
            g_floored = False
            g_capped = False
            ev = None
            per_share = None

            if _is_num(g_in) and _is_num(w_in) and _is_pos(avg_fcf) and _is_pos(shares_outstanding) and _is_pos(w_in):
                g_tmp = float(g_in)
                if g_tmp < growth_floor:
                    g_tmp = growth_floor
                    g_floored = True

                cap_level = float(w_in) - float(terminal_growth_gap)
                if g_tmp >= cap_level:
                    g_tmp = cap_level
                    g_capped = True

                g_used = g_tmp
                ev = _dcf_enterprise_value(float(avg_fcf), g_used, float(w_in), years=int(years))
                per_share = (ev / float(shares_outstanding)) if _is_pos(ev) else None

            rows.append({
                "Growth_Input": float(g_in) if _is_num(g_in) else None,
                "WACC_Input": float(w_in) if _is_num(w_in) else None,
                "Growth_Used": float(g_used) if _is_num(g_used) else None,
                "WACC_Used": float(w_in) if _is_num(w_in) else None,
                "Growth_Floored": bool(g_floored),
                "Growth_Capped": bool(g_capped),
                "EV_Implied": float(ev) if _is_num(ev) else None,
                "Per_Share_Value": float(per_share) if _is_num(per_share) else None,
            })

    return pd.DataFrame(rows)


def dcf_sensitivity_grid(
    ticker: str,
    *,
    years: int = 5,
    risk_free_rate: float = VALUATION_DEFAULTS["risk_free_rate"],
    equity_risk_premium: float = VALUATION_DEFAULTS["equity_risk_premium"],
    growth: Optional[float] = None,
    target_cagr_fallback: float = VALUATION_DEFAULTS["target_cagr_fallback"],
    use_average_fcf_years: Optional[int] = VALUATION_DEFAULTS["fcf_window_years"],
    assumptions_as_of: Optional[str] = None,
    wacc_values: Optional[List[float]] = None,
    growth_values: Optional[List[float]] = None,
    assumptions_overrides: Optional[Dict[str, Any]] = None,
    as_df: bool = True,
    analysis_report_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a DCF sensitivity table (WACC x terminal growth) using the same baseline inputs
    and guardrails as the implied-EV DCF path.

    Returns a dict with:
      - grid_long: one row per (growth, wacc) cell
      - grid_wide: matrix of per-share values (rows=growth, cols=wacc)
      - inputs_used: baseline assumptions and generated grids
      - notes: explanatory warnings / selection notes
    """
    analysis_report_date = analysis_report_date or _today_iso()
    t = _sanitize_ticker(ticker)
    snap = _pull_company_snapshot(t)
    guardrails = _merge_guardrails(assumptions_overrides)
    assumptions_used = valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
        terminal_growth_gap=assumptions_overrides.get("terminal_growth_gap") if assumptions_overrides else None,
    )

    notes: List[str] = []

    beta = _safe_float(snap.get("beta"))
    if beta is None:
        raise ValueError(f"Missing beta from yfinance for {t}. Cannot compute sensitivity WACC baseline.")

    base_wacc = _calculate_wacc(snap, risk_free_rate, equity_risk_premium, beta, guardrails=guardrails)
    if base_wacc is None:
        base_wacc = risk_free_rate + beta * equity_risk_premium
        notes.append("WACC baseline fell back to cost of equity due to missing debt inputs.")

    fcf_series_all = _fcf_series_from_cashflow(snap.get("cashflow"))
    if not isinstance(fcf_series_all, pd.Series) or fcf_series_all.dropna().empty:
        raise ValueError(f"No historical FCF points available from Yahoo for {t}.")

    fcf_series_all = fcf_series_all.dropna()
    available = len(fcf_series_all)
    n = available if use_average_fcf_years is None else max(1, min(int(use_average_fcf_years), available))
    fcf_window = fcf_series_all.tail(n)
    avg_fcf = _normalized_fcf_baseline(fcf_window)
    if not _is_pos(avg_fcf):
        raise ValueError(f"Average FCF not positive for {t}; sensitivity grid would be misleading.")

    if growth is not None:
        base_g = float(growth)
        notes.append("Using user-provided terminal growth baseline.")
    else:
        fcf_cagr = _fcf_cagr_from_series(fcf_series_all)
        rev_cagr = _revenue_cagr_from_series(snap.get("revenue_series"))
        lo_b, hi_b = guardrails["fcf_cagr_bounds"]
        if _is_num(fcf_cagr) and lo_b <= fcf_cagr <= hi_b:
            base_g = float(fcf_cagr)
            notes.append("Using historical FCF CAGR as terminal growth baseline.")
        else:
            r_lo_b, r_hi_b = guardrails["rev_cagr_bounds"]
            if _is_num(rev_cagr) and r_lo_b <= rev_cagr <= r_hi_b:
                base_g = float(rev_cagr) * float(guardrails["revenue_cagr_haircut"])
                notes.append("Using adjusted historical revenue CAGR as terminal growth baseline.")
            else:
                base_g = float(target_cagr_fallback)
                notes.append("Using fallback terminal growth baseline.")

    if growth_values is None:
        growth_values = [base_g - 0.02, base_g - 0.01, base_g, base_g + 0.01, base_g + 0.02]
    if wacc_values is None:
        growth_risk_spread = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]
        wacc_values = [float(w) for w in growth_risk_spread if _is_pos(w)]

    # Normalize and sort for stable output shape
    wacc_values = sorted({round(float(w), 6) for w in wacc_values if _is_pos(w)})
    growth_values = sorted({round(float(g), 6) for g in growth_values if _is_num(g)})
    if not wacc_values or not growth_values:
        raise ValueError("Sensitivity grid requires at least one valid WACC and growth value.")

    so = _safe_float(snap.get("shares_outstanding"))
    if not _is_pos(so):
        raise ValueError(f"Missing shares outstanding from yfinance for {t}. Cannot compute per-share sensitivity grid.")

    grid_long = _dcf_sensitivity_grid_from_inputs(
        avg_fcf=float(avg_fcf),
        shares_outstanding=float(so),
        years=int(years),
        wacc_values=wacc_values,
        growth_values=growth_values,
        terminal_growth_gap=assumptions_used["terminal_growth_gap"],
        growth_floor=float(guardrails["growth_floor"]),
    )

    grid_wide = (
        grid_long.pivot(index="Growth_Input", columns="WACC_Input", values="Per_Share_Value")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    grid_wide.index.name = "Growth_Input"
    grid_wide.columns.name = "WACC_Input"

    # Notes on grid adjustments
    capped_cells = int(grid_long["Growth_Capped"].sum()) if "Growth_Capped" in grid_long.columns else 0
    floored_cells = int(grid_long["Growth_Floored"].sum()) if "Growth_Floored" in grid_long.columns else 0
    if capped_cells:
        notes.append(f"{capped_cells} grid cells had growth capped at WACC - {assumptions_used['terminal_growth_gap']:.1%}.")
    if floored_cells:
        notes.append(f"{floored_cells} grid cells had growth floored at {float(guardrails['growth_floor']):.1%}.")

    out = {
        "ticker": t,
        "analysis_report_date": analysis_report_date,
        "grid_long": grid_long,
        "grid_wide": grid_wide,
        "inputs_used": {
            "years": int(years),
            "avg_fcf_used": float(avg_fcf),
            "shares_outstanding": float(so),
            "base_wacc": float(base_wacc),
            "base_growth": float(base_g),
            "wacc_values": wacc_values,
            "growth_values": growth_values,
            "assumptions_used": assumptions_used,
        },
        "notes": notes,
    }

    if as_df:
        return out

    return {
        "ticker": t,
        "analysis_report_date": analysis_report_date,
        "grid_long": to_records(grid_long, analysis_report_date=analysis_report_date),
        "grid_wide": json.loads(grid_wide.to_json()),
        "inputs_used": out["inputs_used"],
        "notes": notes,
    }

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
    assumptions_overrides: Optional[Dict[str, Any]] = None,
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
    guardrails = _merge_guardrails(assumptions_overrides)
    assumptions_used = valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
        terminal_growth_gap=assumptions_overrides.get("terminal_growth_gap") if assumptions_overrides else None,
    )

    # --- IMPROVED: Calculate proper WACC ---
    beta = _safe_float(snap.get("beta"))
    if beta is None:
        raise ValueError(f"Missing beta from yfinance for {t}. Cannot compute WACC.")
    
    # Use the improved WACC calculation
    wacc = _calculate_wacc(snap, risk_free_rate, equity_risk_premium, beta, guardrails=guardrails)
    used_cost_of_equity_fallback = False
    if wacc is None:
        # Fallback to cost of equity if WACC calculation fails
        wacc = risk_free_rate + beta * equity_risk_premium
        used_cost_of_equity_fallback = True

    # FCF series (Operating CF - CapEx) - using corrected calculation
    fcf_series_all = _fcf_series_from_cashflow(snap.get('cashflow'))
    notes: List[str] = []
    confidence_flags: Dict[str, bool] = {
        "used_cost_of_equity_fallback": used_cost_of_equity_fallback,
        "missing_fcf_history": False,
        "non_positive_fcf": False,
        "used_fallback_growth": False,
        "used_revenue_growth_proxy": False,
        "high_fcf_volatility": False,
        "moderate_fcf_volatility": False,
        "short_fcf_history": False,
        "growth_capped": False,
        "growth_floored": False,
        "invalid_implied_ev": False,
        "extreme_ev_fcf_multiple": False,
    }

    if not isinstance(fcf_series_all, pd.Series) or fcf_series_all.empty:
        notes.append("No historical FCF points available from Yahoo.")
        confidence_flags["missing_fcf_history"] = True
        confidence_flags["invalid_implied_ev"] = True
        valuation_confidence = _valuation_confidence_from_flags(confidence_flags, context="dcf_implied_enterprise_value")
        df = pd.DataFrame([{
            "Ticker": t, "Avg_FCF_Used": None, "Growth_Used": None, "WACC_Used": wacc,
            "Years": 0 if years is None else int(years), "EV_Implied": None, "Assumptions_Used": assumptions_used, "Valuation_Confidence": valuation_confidence, "Notes": " ".join(notes)
        }])
        return df if as_df else to_records(df, analysis_report_date=analysis_report_date, notes=notes)

    # Determine the averaging window
    fcf_series_all = fcf_series_all.dropna()
    available = len(fcf_series_all)
    n = available if use_average_fcf_years is None else max(1, min(use_average_fcf_years, available))
    fcf_window = fcf_series_all.tail(n)
    confidence_flags["short_fcf_history"] = bool(len(fcf_window) < 3)

    # IMPROVED: Enhanced volatility analysis
    mean_f = fcf_window.mean()
    std_f = fcf_window.std(ddof=1) if len(fcf_window) >= 2 else 0.0
    cov = (abs(std_f / mean_f) if (mean_f not in (0, None) and pd.notna(mean_f) and mean_f != 0) else np.nan)
    
    # Add volatility context
    if _is_num(cov):
        if cov > volatility_threshold:
            notes.append(f"Historical FCF highly volatile (CoV={cov:.2f} > threshold {volatility_threshold:.2f}); consider shortening averaging window.")
            confidence_flags["high_fcf_volatility"] = True
        elif cov > float(guardrails["cov_moderate"]):
            notes.append(f"Historical FCF moderately volatile (CoV={cov:.2f}).")
            confidence_flags["moderate_fcf_volatility"] = True

    # IMPROVED: Use normalized FCF baseline instead of simple mean
    avg_fcf = _normalized_fcf_baseline(fcf_window)
    
    if not _is_pos(avg_fcf):
        notes.append("Average FCF not positive; EV will be None.")
        confidence_flags["non_positive_fcf"] = True
        confidence_flags["invalid_implied_ev"] = True
        valuation_confidence = _valuation_confidence_from_flags(confidence_flags, context="dcf_implied_enterprise_value")
        df = pd.DataFrame([{
            "Ticker": t, "Avg_FCF_Used": avg_fcf, "Growth_Used": None, "WACC_Used": wacc,
            "Years": 0 if years is None else int(years), "EV_Implied": None, "Assumptions_Used": assumptions_used, "Valuation_Confidence": valuation_confidence, "Notes": " ".join(notes)
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
        
        lo_b, hi_b = guardrails["fcf_cagr_bounds"]
        if _is_num(fcf_cagr) and lo_b <= fcf_cagr <= hi_b:
            g = fcf_cagr
            notes.append("Using historical FCF CAGR for growth.")
        else:
            r_lo_b, r_hi_b = guardrails["rev_cagr_bounds"]
            if _is_num(rev_cagr) and r_lo_b <= rev_cagr <= r_hi_b:
                g = rev_cagr * float(guardrails["revenue_cagr_haircut"])
                notes.append("Using historical revenue CAGR (adjusted) for growth.")
                confidence_flags["used_revenue_growth_proxy"] = True
            else:
                g = target_cagr_fallback
                notes.append("Using fallback growth rate.")
                confidence_flags["used_fallback_growth"] = True

    # Guardrail: cap g slightly below WACC to avoid division by zero in TV
    capped = False
    original_g = g
    if _is_num(wacc) and g is not None and g >= wacc:
        g = wacc - assumptions_used["terminal_growth_gap"]
        capped = True

    # Additional validation for negative growth
    floor = float(guardrails["growth_floor"])
    if g is not None and g < floor:
        g = max(g, floor)
        notes.append(f"Growth rate floored at {floor:.1%} for stability.")
        confidence_flags["growth_floored"] = True

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
        notes.append(f"Growth capped from {original_g:.1%} to {g:.1%} (WACC - {assumptions_used['terminal_growth_gap']:.1%}) to stabilize terminal value.")
        confidence_flags["growth_capped"] = True

    # Add reasonableness checks
    if _is_pos(ev) and _is_pos(avg_fcf):
        ev_fcf_multiple = ev / avg_fcf
        if ev_fcf_multiple > float(guardrails["ev_fcf_multiple_warn_high"]):
            notes.append(f"Warning: EV/FCF multiple of {ev_fcf_multiple:.1f}x seems high; check assumptions.")
            confidence_flags["extreme_ev_fcf_multiple"] = True
        elif ev_fcf_multiple < float(guardrails["ev_fcf_multiple_warn_low"]):
            notes.append(f"Note: EV/FCF multiple of {ev_fcf_multiple:.1f}x is relatively low.")
            confidence_flags["extreme_ev_fcf_multiple"] = True

    if not _is_pos(ev):
        confidence_flags["invalid_implied_ev"] = True

    valuation_confidence = _valuation_confidence_from_flags(confidence_flags, context="dcf_implied_enterprise_value")

    out = pd.DataFrame([{
        "Ticker": t,
        "Avg_FCF_Used": float(avg_fcf) if _is_num(avg_fcf) else None,
        "Growth_Used": float(g) if _is_num(g) else None,
        "WACC_Used": float(wacc) if _is_num(wacc) else None,
        "Years": years_used,
        "Assumptions_Used": assumptions_used,
        "Valuation_Confidence": valuation_confidence,
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
    assumptions_overrides: Optional[Dict[str, Any]] = None,
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
    guardrails = _merge_guardrails(assumptions_overrides)
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
        assumptions_overrides=assumptions_overrides,
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
        terminal_growth_gap=assumptions_overrides.get("terminal_growth_gap") if assumptions_overrides else None,
    )
    base_notes = implied_df.loc[0, "Notes"] or ""
    upstream_conf = implied_df.loc[0, "Valuation_Confidence"] if "Valuation_Confidence" in implied_df.columns else None
    upstream_flags = dict(upstream_conf.get("flags", {})) if isinstance(upstream_conf, dict) else {}
    confidence_flags = dict(upstream_flags)
    confidence_flags.setdefault("missing_observed_ev", False)
    confidence_flags.setdefault("observed_ev_estimated_from_market_cap", False)

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
                confidence_flags["observed_ev_estimated_from_market_cap"] = True
                
    except Exception as e:
        observed_ev = None
        base_notes += f" Error retrieving market data: {str(e)[:50]}."

    notes = [base_notes] if base_notes else []
    
    if not _is_pos(observed_ev):
        notes.append("Yahoo enterpriseValue not available; comparison limited.")
        premium_pct = None
        confidence_flags["missing_observed_ev"] = True
    else:
        if implied_ev is None or not _is_num(implied_ev) or implied_ev <= 0:
            premium_pct = None
            notes.append("Cannot calculate premium: implied EV not valid.")
        else:
            premium_pct = (observed_ev / implied_ev - 1.0) * 100
            
            # IMPROVED: More nuanced interpretation
            if abs(premium_pct) < float(guardrails["premium_band_small"]):
                notes.append("Observed EV roughly equals DCF-implied EV (within 5%).")
            elif premium_pct > float(guardrails["premium_band_large"]):
                notes.append("Observed EV significantly above DCF-implied EV (>20% premium); market expects higher growth or lower risk.")
            elif premium_pct > 0:
                notes.append("Observed EV modestly above DCF-implied EV (positive premium).")
            elif premium_pct < -float(guardrails["premium_band_large"]):
                notes.append("Observed EV significantly below DCF-implied EV (>20% discount); potential undervaluation or market expects lower growth.")
            else:
                notes.append("Observed EV modestly below DCF-implied EV (negative premium).")

    valuation_confidence = _valuation_confidence_from_flags(
        confidence_flags,
        context="compare_to_market_ev",
        extra_reasons=[f"Upstream DCF confidence: {upstream_conf.get('level')}" for _ in [0] if isinstance(upstream_conf, dict) and upstream_conf.get("level")],
    )

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
        "Valuation_Confidence": valuation_confidence,
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
    equity_implied = _equity_value_from_ev(
        ev_implied,
        total_debt=total_debt,
        cash_eq=cash_eq,
        minority_interest=minority,
    )
    if not _is_num(equity_implied):
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
    assumptions_overrides: Optional[Dict[str, Any]] = None,
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
            assumptions_overrides=assumptions_overrides,
            as_df=True,
            analysis_report_date=analysis_report_date
        )

    # Pull core EV outputs
    ev_implied = ev_df.iloc[0].get("EV_Implied")
    avg_fcf = ev_df.iloc[0].get("Avg_FCF_Used")
    g_used = ev_df.iloc[0].get("Growth_Used")
    wacc_used = ev_df.iloc[0].get("WACC_Used")
    years_used = int(ev_df.iloc[0].get("Years"))
    guardrails = _merge_guardrails(assumptions_overrides)
    assumptions_used = ev_df.iloc[0].get("Assumptions_Used") if "Assumptions_Used" in ev_df.columns else valuation_defaults(
        as_of_date=assumptions_as_of or analysis_report_date,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_cagr_fallback=target_cagr_fallback,
        fcf_window_years=use_average_fcf_years if use_average_fcf_years is not None else VALUATION_DEFAULTS["fcf_window_years"],
        terminal_growth_gap=assumptions_overrides.get("terminal_growth_gap") if assumptions_overrides else None,
    )
    base_notes = str(ev_df.iloc[0].get("Notes") or "")
    upstream_conf = ev_df.iloc[0].get("Valuation_Confidence") if "Valuation_Confidence" in ev_df.columns else None
    upstream_flags = dict(upstream_conf.get("flags", {})) if isinstance(upstream_conf, dict) else {}
    confidence_flags = dict(upstream_flags)
    confidence_flags.setdefault("missing_observed_market_cap", False)
    confidence_flags.setdefault("market_cap_estimated_from_price", False)
    confidence_flags.setdefault("negative_equity_implied", False)
    confidence_flags.setdefault("missing_shares_for_per_share", False)

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

    confidence_flags["market_cap_estimated_from_price"] = any("MarketCap estimated" in str(n) for n in notes)

    # 4) Premium calculation (MarketCap vs Implied Equity)
    if not (_is_num(equity_implied) and equity_implied > 0 and _is_num(observed_mktcap)):
        premium_pct = None
        notes.append("Cannot compute premium: missing or invalid Equity_Implied / MarketCap.")
        if not _is_num(observed_mktcap):
            confidence_flags["missing_observed_market_cap"] = True
    else:
        premium_pct = (observed_mktcap / equity_implied - 1.0) * 100.0
        # brief interpretation
        if abs(premium_pct) < float(guardrails["premium_band_small"]):
            notes.append("Observed Market Cap ≈ Implied Equity (within 5%).")
        elif premium_pct > float(guardrails["premium_band_large"]):
            notes.append("Market Cap materially above Implied Equity (>20% premium).")
        elif premium_pct > 0:
            notes.append("Market Cap modestly above Implied Equity (positive premium).")
        elif premium_pct < -float(guardrails["premium_band_large"]):
            notes.append("Market Cap materially below Implied Equity (<-20% premium).")
        else:
            notes.append("Market Cap modestly below Implied Equity (negative premium).")

    confidence_flags["negative_equity_implied"] = bool(_is_num(equity_implied) and equity_implied < 0)
    confidence_flags["missing_shares_for_per_share"] = not _is_pos(eq_df.iloc[0].get("SharesOutstanding"))
    valuation_confidence = _valuation_confidence_from_flags(
        confidence_flags,
        context="compare_to_market_cap",
        extra_reasons=[f"Upstream EV confidence: {upstream_conf.get('level')}" for _ in [0] if isinstance(upstream_conf, dict) and upstream_conf.get("level")],
    )

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
        "Valuation_Confidence": valuation_confidence,
        "Notes": " ".join([n for n in notes if n]).strip()
    }])

    return out if as_df else to_records(
        out, analysis_report_date=analysis_report_date, notes=[n for n in notes if n]
    )
__all__ = [
    'dcf_three_scenarios',
    'dcf_sensitivity_grid',
    'dcf_implied_enterprise_value',
    'compare_to_market_ev',
    'implied_equity_value_from_ev',
    'compare_to_market_cap',
]
