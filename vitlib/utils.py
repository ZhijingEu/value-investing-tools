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
    info = _get_provider().info(_sanitize_ticker(symbol))
    return dict(info) if info else {}

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


def _safe_ratio(
    numerator: Any,
    denominator: Any,
    *,
    denominator_label: str = "denominator",
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute a ratio while preserving meaningful negative values.

    A ratio is N/M only for missing inputs or a true zero denominator. Negative
    denominators are computed and noted because they can carry analytical signal
    (for example negative equity).
    """
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None:
        return None, "numerator missing"
    if den is None:
        return None, f"{denominator_label} missing"
    if den == 0:
        return None, f"{denominator_label} = 0; ratio N/M"
    value = num / den
    if den < 0:
        return value, f"{denominator_label} negative; ratio computed with sign"
    if value < 0:
        return value, "ratio negative; check sign combination"
    return value, None


def _safe_cagr(
    first: Any,
    last: Any,
    n_years: int,
    *,
    label: str = "Series",
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """
    Compute CAGR only when endpoints are positive.

    If CAGR is undefined, return the absolute change as context instead of
    manufacturing a sign-flipped growth rate.
    """
    first_f = _safe_float(first)
    last_f = _safe_float(last)
    if first_f is None or last_f is None:
        return None, f"{label} CAGR undefined (missing endpoint).", None
    absolute_change = last_f - first_f
    if n_years <= 0:
        return None, f"{label} CAGR undefined (non-positive year span).", absolute_change
    if first_f <= 0 or last_f <= 0:
        return None, f"{label} CAGR undefined (first/last <= 0); absolute change reported instead.", absolute_change
    return (last_f / first_f) ** (1.0 / n_years) - 1.0, None, absolute_change

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
DEFAULT_MACRO_INPUTS: Dict[str, Any] = {
    "as_of": "2026-01",
    "risk_free_rate": 0.0418,
    "equity_risk_premium": 0.0423,
    "source": "Damodaran public implied ERP update, Jan 2026",
    "config_path": "data/damodaran_macro.json",
    "load_status": "fallback",
}


def _load_macro_inputs() -> Dict[str, Any]:
    """
    Load date-stamped macro assumptions once at import time.

    A running MCP process therefore uses one stable assumption set. To refresh
    after editing the JSON, restart the MCP server or pass explicit overrides.
    """
    cfg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "damodaran_macro.json")
    )
    payload = dict(DEFAULT_MACRO_INPUTS)
    payload["config_path"] = os.path.relpath(cfg_path, os.path.dirname(os.path.dirname(__file__)))
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        rf = _safe_float(raw.get("risk_free_rate"))
        erp = _safe_float(raw.get("equity_risk_premium"))
        if rf is None or erp is None:
            raise ValueError("risk_free_rate and equity_risk_premium must be numeric")
        payload.update({
            "as_of": str(raw.get("as_of") or payload["as_of"]),
            "risk_free_rate": rf,
            "equity_risk_premium": erp,
            "source": str(raw.get("source") or payload["source"]),
            "load_status": "loaded",
        })
    except Exception as exc:
        payload["load_status"] = "fallback"
        payload["load_error"] = str(exc)
    return payload


MACRO_INPUTS = _load_macro_inputs()

VALUATION_DEFAULTS: Dict[str, float] = {
    "risk_free_rate": float(MACRO_INPUTS["risk_free_rate"]),
    "equity_risk_premium": float(MACRO_INPUTS["equity_risk_premium"]),
    "target_cagr_fallback": 0.020,
    "fcf_window_years": 3,
    "terminal_growth_gap": 0.005,  # g <= WACC - gap
}
VALUATION_ASSUMPTIONS_SCHEMA_VERSION = "1.1"
RUN_MANIFEST_SCHEMA_VERSION = "1.0"


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
        "macro_as_of": payload.get("macro_as_of"),
        "macro_source": payload.get("macro_source"),
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
    rf_value = VALUATION_DEFAULTS["risk_free_rate"] if risk_free_rate is None else float(risk_free_rate)
    erp_value = VALUATION_DEFAULTS["equity_risk_premium"] if equity_risk_premium is None else float(equity_risk_premium)
    rf_source = "macro_config" if risk_free_rate is None or rf_value == VALUATION_DEFAULTS["risk_free_rate"] else "user_override"
    erp_source = "macro_config" if equity_risk_premium is None or erp_value == VALUATION_DEFAULTS["equity_risk_premium"] else "user_override"
    rf_as_of = MACRO_INPUTS.get("as_of") if rf_source == "macro_config" else as_of_date or _today_iso()
    erp_as_of = MACRO_INPUTS.get("as_of") if erp_source == "macro_config" else as_of_date or _today_iso()
    payload = {
        "as_of_date": as_of_date or _today_iso(),
        "risk_free_rate": rf_value,
        "equity_risk_premium": erp_value,
        "target_cagr_fallback": VALUATION_DEFAULTS["target_cagr_fallback"] if target_cagr_fallback is None else float(target_cagr_fallback),
        "fcf_window_years": VALUATION_DEFAULTS["fcf_window_years"] if fcf_window_years is None else int(fcf_window_years),
        "terminal_growth_gap": VALUATION_DEFAULTS["terminal_growth_gap"] if terminal_growth_gap is None else float(terminal_growth_gap),
        "assumptions_schema_version": VALUATION_ASSUMPTIONS_SCHEMA_VERSION,
        "assumptions_source": "ValueInvestingTools.valuation_defaults",
        "macro_as_of": MACRO_INPUTS.get("as_of"),
        "macro_source": MACRO_INPUTS.get("source"),
        "macro_config_path": MACRO_INPUTS.get("config_path"),
        "macro_load_status": MACRO_INPUTS.get("load_status"),
        "macro_inputs": {
            "risk_free_rate": {
                "value": rf_value,
                "source": rf_source,
                "source_detail": MACRO_INPUTS.get("source") if rf_source == "macro_config" else "user supplied function argument",
                "as_of": rf_as_of,
            },
            "equity_risk_premium": {
                "value": erp_value,
                "source": erp_source,
                "source_detail": MACRO_INPUTS.get("source") if erp_source == "macro_config" else "user supplied function argument",
                "as_of": erp_as_of,
            },
        },
    }
    if MACRO_INPUTS.get("load_error"):
        payload["macro_load_error"] = MACRO_INPUTS.get("load_error")
    payload["assumptions_snapshot_id"] = _valuation_assumptions_snapshot_id(payload)
    return payload


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-friendly scalar/container for manifests and MCP payloads."""
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _manifest_entries_from_assumptions(
    assumptions_used: Optional[Dict[str, Any]],
    *,
    user_override_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Normalize valuation assumptions into value/source/as_of manifest entries."""
    assumptions_used = assumptions_used or valuation_defaults()
    user_override_keys = set(user_override_keys or [])
    entries: Dict[str, Dict[str, Any]] = {}

    for key, entry in (assumptions_used.get("macro_inputs") or {}).items():
        if isinstance(entry, dict):
            entries[key] = {
                "value": _json_safe_value(entry.get("value")),
                "source": entry.get("source"),
                "source_detail": entry.get("source_detail"),
                "as_of": entry.get("as_of"),
            }

    non_macro_defaults = {
        "target_cagr_fallback": VALUATION_DEFAULTS["target_cagr_fallback"],
        "fcf_window_years": VALUATION_DEFAULTS["fcf_window_years"],
        "terminal_growth_gap": VALUATION_DEFAULTS["terminal_growth_gap"],
    }
    for key, default in non_macro_defaults.items():
        value = assumptions_used.get(key)
        source = "user_override" if key in user_override_keys or value != default else "default"
        entries[key] = {
            "value": _json_safe_value(value),
            "source": source,
            "as_of": assumptions_used.get("as_of_date"),
        }
    return entries


def build_run_manifest(
    *,
    ticker: str,
    analysis_report_date: Optional[str] = None,
    assumptions_used: Optional[Dict[str, Any]] = None,
    user_override_keys: Optional[List[str]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    data_as_of: Optional[Dict[str, Any]] = None,
    health_notes: Optional[List[str]] = None,
    source: str = "ValueInvestingTools",
) -> Dict[str, Any]:
    """
    Build a compact audit manifest for valuation/orchestrator outputs.

    The manifest is intentionally additive: callers keep their existing output
    columns and include this payload where reproducibility matters.
    """
    assumptions_used = assumptions_used or valuation_defaults(as_of_date=analysis_report_date)
    material = {
        "ticker": _sanitize_ticker(ticker),
        "analysis_report_date": analysis_report_date or _today_iso(),
        "assumptions_snapshot_id": assumptions_used.get("assumptions_snapshot_id"),
        "inputs": _json_safe_value(inputs or {}),
        "outputs": _json_safe_value(outputs or {}),
        "data_as_of": _json_safe_value(data_as_of or {}),
    }
    digest = hashlib.sha1(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    return {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"vit-run-{digest}",
        "source": source,
        "ticker": _sanitize_ticker(ticker),
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "analysis_report_date": analysis_report_date or _today_iso(),
        "data_as_of": _json_safe_value(data_as_of or {}),
        "assumptions_snapshot_id": assumptions_used.get("assumptions_snapshot_id"),
        "assumptions": _manifest_entries_from_assumptions(
            assumptions_used,
            user_override_keys=user_override_keys,
        ),
        "inputs": _json_safe_value(inputs or {}),
        "outputs": _json_safe_value(outputs or {}),
        "health": [_json_safe_value(n) for n in (health_notes or []) if str(n).strip()],
    }


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
    '_safe_ratio',
    '_safe_cagr',
    '_equity_value_from_ev',
    '_valuation_confidence_from_flags',
    '_ensure_dir',
    '_fcf_series_from_cashflow',
    '_pct_from_info',
    'to_records',
    'DEFAULT_MACRO_INPUTS',
    'MACRO_INPUTS',
    'VALUATION_DEFAULTS',
    'VALUATION_ASSUMPTIONS_SCHEMA_VERSION',
    'RUN_MANIFEST_SCHEMA_VERSION',
    '_valuation_assumptions_snapshot_id',
    'valuation_defaults',
    'build_run_manifest',
]
