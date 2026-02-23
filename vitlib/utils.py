from __future__ import annotations

import os
import json
import math
import warnings
import hashlib
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import inspect

from providers.base import MarketDataProvider
from providers.yahoo import YahooFinanceProvider

# -----------------------------
# General helpers
# -----------------------------
_PROVIDER: MarketDataProvider | None = None

def _get_provider() -> MarketDataProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = YahooFinanceProvider()
    return _PROVIDER

def _provider_ticker(symbol: str):
    return _get_provider().ticker(_sanitize_ticker(symbol))

def _provider_info(symbol: str) -> Dict[str, Any]:
    return dict(_get_provider().info(_sanitize_ticker(symbol)))

def _provider_financials(symbol: str):
    return _get_provider().financials(_sanitize_ticker(symbol))

def _provider_cashflow(symbol: str):
    return _get_provider().cashflow(_sanitize_ticker(symbol))

def _provider_balance_sheet(symbol: str):
    return _get_provider().balance_sheet(_sanitize_ticker(symbol))

def _provider_history(symbol: str, **kwargs: Any):
    return _get_provider().history(_sanitize_ticker(symbol), **kwargs)
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

def _equity_value_from_ev(
    ev: Optional[float],
    *,
    total_debt: Optional[float] = None,
    cash_eq: Optional[float] = None,
    minority_interest: Optional[float] = None,
    net_debt: Optional[float] = None,
) -> Optional[float]:
    """
    Convert enterprise value to equity value.

    If net_debt is provided, use: Equity = EV - NetDebt - MinorityInterest.
    Otherwise use: Equity = EV - TotalDebt + Cash - MinorityInterest.
    """
    ev_f = _safe_float(ev)
    if ev_f is None:
        return None

    mi = _safe_float(minority_interest) or 0.0
    nd = _safe_float(net_debt)
    if nd is not None:
        return ev_f - nd - mi

    td = _safe_float(total_debt) or 0.0
    c = _safe_float(cash_eq) or 0.0
    return ev_f - td + c - mi


def _valuation_confidence_from_flags(
    flags: Dict[str, Any],
    *,
    context: str,
    extra_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable confidence assessment for valuation outputs.

    This is a confidence score for valuation *reliability/interpretability*,
    not a confidence score for price direction or investment outcome.
    """
    rules = {
        "missing_fcf_history": (0.70, "No historical FCF history available."),
        "non_positive_fcf": (0.65, "Normalized historical FCF is non-positive."),
        "invalid_implied_ev": (0.60, "Implied EV could not be computed from available inputs."),
        "used_cost_of_equity_fallback": (0.18, "WACC fell back to cost of equity due to missing debt inputs."),
        "used_fallback_growth": (0.18, "Growth relied on fallback assumption rather than company history."),
        "used_revenue_growth_proxy": (0.08, "Growth used revenue CAGR proxy instead of FCF CAGR."),
        "high_fcf_volatility": (0.18, "Historical FCF volatility is high."),
        "moderate_fcf_volatility": (0.08, "Historical FCF volatility is moderate."),
        "short_fcf_history": (0.10, "FCF history window is short."),
        "growth_capped": (0.07, "Growth was capped by the WACC guardrail."),
        "growth_floored": (0.05, "Growth was floored for stability."),
        "extreme_ev_fcf_multiple": (0.08, "Implied EV/FCF multiple appears extreme."),
        "missing_observed_ev": (0.12, "Observed enterprise value was unavailable."),
        "observed_ev_estimated_from_market_cap": (0.06, "Observed EV was estimated from market cap + debt - cash."),
        "missing_observed_market_cap": (0.12, "Observed market cap was unavailable."),
        "market_cap_estimated_from_price": (0.06, "Observed market cap was estimated from shares x price."),
        "negative_equity_implied": (0.20, "Implied equity value is negative."),
        "missing_shares_for_per_share": (0.08, "Shares outstanding unavailable for per-share conversion."),
    }

    score = 1.0
    reasons: List[str] = []
    normalized_flags: Dict[str, bool] = {}

    for key, value in (flags or {}).items():
        active = bool(value)
        normalized_flags[key] = active
        if not active:
            continue
        penalty, reason = rules.get(key, (0.03, f"Flag raised: {key}"))
        score -= penalty
        reasons.append(reason)

    if extra_reasons:
        reasons.extend([str(r) for r in extra_reasons if str(r).strip()])

    score = max(0.0, min(1.0, round(float(score), 3)))
    if score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"

    # Deduplicate reasons while preserving order
    seen = set()
    reasons_deduped = []
    for r in reasons:
        if r not in seen:
            reasons_deduped.append(r)
            seen.add(r)

    return {
        "schema_version": "1.0",
        "context": context,
        "score": score,
        "level": level,
        "reasons": reasons_deduped,
        "flags": normalized_flags,
    }

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
    "risk_free_rate": 0.0418,
    "equity_risk_premium": 0.0423,
    "target_cagr_fallback": 0.020,
    "fcf_window_years": 3,
    "terminal_growth_gap": 0.005,  # g <= WACC - gap
}
VALUATION_ASSUMPTIONS_SCHEMA_VERSION = "1.0"


def _valuation_assumptions_snapshot_id(payload: Dict[str, Any]) -> str:
    """Deterministic fingerprint for the effective valuation assumptions payload."""
    material = {
        "as_of_date": str(payload.get("as_of_date")) if payload.get("as_of_date") is not None else None,
        "risk_free_rate": _safe_float(payload.get("risk_free_rate")),
        "equity_risk_premium": _safe_float(payload.get("equity_risk_premium")),
        "target_cagr_fallback": _safe_float(payload.get("target_cagr_fallback")),
        "fcf_window_years": int(payload.get("fcf_window_years")) if payload.get("fcf_window_years") is not None else None,
        "terminal_growth_gap": _safe_float(payload.get("terminal_growth_gap")),
        "assumptions_schema_version": payload.get("assumptions_schema_version", VALUATION_ASSUMPTIONS_SCHEMA_VERSION),
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"vit-val-{digest}"


def valuation_defaults(
    *,
    as_of_date: Optional[str] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: Optional[float] = None,
    target_cagr_fallback: Optional[float] = None,
    fcf_window_years: Optional[int] = None,
    terminal_growth_gap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a normalized assumptions payload for valuation outputs and audits.
    """
    payload = {
        "as_of_date": as_of_date or _today_iso(),
        "risk_free_rate": VALUATION_DEFAULTS["risk_free_rate"] if risk_free_rate is None else float(risk_free_rate),
        "equity_risk_premium": VALUATION_DEFAULTS["equity_risk_premium"] if equity_risk_premium is None else float(equity_risk_premium),
        "target_cagr_fallback": VALUATION_DEFAULTS["target_cagr_fallback"] if target_cagr_fallback is None else float(target_cagr_fallback),
        "fcf_window_years": VALUATION_DEFAULTS["fcf_window_years"] if fcf_window_years is None else int(fcf_window_years),
        "terminal_growth_gap": VALUATION_DEFAULTS["terminal_growth_gap"] if terminal_growth_gap is None else float(terminal_growth_gap),
        "assumptions_schema_version": VALUATION_ASSUMPTIONS_SCHEMA_VERSION,
        "assumptions_source": "ValueInvestingTools.valuation_defaults",
    }
    payload["assumptions_snapshot_id"] = _valuation_assumptions_snapshot_id(payload)
    return payload


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fcf_series_from_cashflow(cf: pd.DataFrame) -> pd.Series:
    """CORRECTED: FCF = Operating Cash Flow - Capital Expenditure"""
    if not isinstance(cf, pd.DataFrame) or cf.empty:
        return pd.Series(dtype=float)

    ocf = cf.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cf.index else pd.Series(dtype=float)
    capex = cf.loc['Capital Expenditure'].dropna() if 'Capital Expenditure' in cf.index else pd.Series(dtype=float)

    # CORRECTED: Subtract CapEx (it's usually negative in provider data, so we add it)
    if not ocf.empty and not capex.empty:
        # Ensure same index
        common_idx = ocf.index.intersection(capex.index)
        if not common_idx.empty:
            return ocf.loc[common_idx] + capex.loc[common_idx]  # CapEx is negative

    return pd.Series(dtype=float)

__all__ = [
    '_today_iso',
    '_sanitize_ticker',
    '_is_num',
    '_is_pos',
    '_safe_float',
    '_equity_value_from_ev',
    '_valuation_confidence_from_flags',
    '_ensure_dir',
    '_fcf_series_from_cashflow',
    '_pct_from_info',
    'to_records',
    'VALUATION_DEFAULTS',
    'VALUATION_ASSUMPTIONS_SCHEMA_VERSION',
    '_valuation_assumptions_snapshot_id',
    'valuation_defaults',
]
