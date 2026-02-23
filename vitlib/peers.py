from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal, Union

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
    _ensure_dir,
    to_records,
)
from vitlib.fundamentals import compute_fundamentals_actuals
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
    trailing_pe = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
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
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
        'operating_margin': info.get('operatingMargins'),
        'net_margin': info.get('profitMargins'),
        'return_on_equity_info': info.get('returnOnEquity'),
        'return_on_assets_info': info.get('returnOnAssets'),
        'debt_to_equity_info': info.get('debtToEquity'),
        'pe_ratio': trailing_pe,  # selected/default PE (TTM unless peer_multiples overrides)
        'trailing_pe_ratio': trailing_pe,
        'forward_pe_ratio': forward_pe,
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
    multiple_basis: Literal["ttm", "forward_pe"] = "ttm",
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

    multiple_basis:
      - "ttm" (default): PE/PS/EV_EBITDA use trailing Yahoo ratios.
      - "forward_pe": PE uses Yahoo `forwardPE` where available and falls back to trailing PE.
        PS and EV/EBITDA remain trailing because forward values are not consistently available.

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
            "target_ticker": target_tkr,
            "multiple_basis": multiple_basis,
            "metric_basis_map": {"PE": "trailingPE", "PS": "priceToSalesTrailing12Months", "EV_EBITDA": "enterpriseToEbitda"},
            "notes": ["No peer snapshots could be fetched."],
            "peer_quality_diagnostics": {
                "peer_count_for_stats": 0,
                "coverage_warn_threshold": 0.60,
                "metrics": [],
                "warnings": ["No peer snapshots could be fetched."],
            },
            "peer_comp_detail": peer_comp_detail,
            "peer_multiple_bands_wide": empty_bands,
            "peer_comp_bands": empty_long,
        }
        return out if as_df else out  # your to_records wrapper if you use one

    # --- Apply optional basis selection (Sprint 2 foundation) ---
    notes: List[str] = []
    metric_basis_map = {
        "PE": "trailingPE",
        "PS": "priceToSalesTrailing12Months",
        "EV_EBITDA": "enterpriseToEbitda",
    }

    if "trailing_pe_ratio" not in peer_comp_detail.columns:
        peer_comp_detail["trailing_pe_ratio"] = peer_comp_detail.get("pe_ratio")
    if "forward_pe_ratio" not in peer_comp_detail.columns:
        peer_comp_detail["forward_pe_ratio"] = np.nan

    if multiple_basis == "forward_pe":
        selected_pe = []
        selected_basis = []
        forward_used = 0
        for _, row in peer_comp_detail.iterrows():
            fpe = row.get("forward_pe_ratio")
            tpe = row.get("trailing_pe_ratio")
            if _is_num(fpe):
                selected_pe.append(float(fpe))
                selected_basis.append("forwardPE")
                forward_used += 1
            elif _is_num(tpe):
                selected_pe.append(float(tpe))
                selected_basis.append("trailingPE_fallback")
            else:
                selected_pe.append(np.nan)
                selected_basis.append(None)
        peer_comp_detail["pe_ratio"] = selected_pe
        peer_comp_detail["pe_ratio_basis"] = selected_basis
        metric_basis_map["PE"] = "forwardPE (fallback: trailingPE)"

        total_rows = len(peer_comp_detail)
        notes.append(f"PE basis mode 'forward_pe': used forwardPE for {forward_used}/{total_rows} rows; fallback to trailingPE when missing.")
        notes.append("PS and EV/EBITDA remain trailing metrics in this mode (Yahoo forward coverage is inconsistent).")
    else:
        # Preserve historical behavior and annotate basis for transparency
        peer_comp_detail["pe_ratio"] = peer_comp_detail["trailing_pe_ratio"]
        peer_comp_detail["pe_ratio_basis"] = "trailingPE"

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

    # --- (0) Peer comparability diagnostics (coverage + dispersion checks) ---
    def _peer_quality_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
        coverage_warn_threshold = 0.60
        warnings: List[str] = []
        metrics_out: List[Dict[str, Any]] = []

        n_peers = int(len(df))
        if n_peers == 0:
            return {
                "peer_count_for_stats": 0,
                "coverage_warn_threshold": coverage_warn_threshold,
                "metrics": [],
                "warnings": ["No peers available for peer-stat diagnostics (target may have been excluded from a 1-name set)."],
            }

        work = df.copy()
        if "ebitda" in work.columns and "revenue" in work.columns:
            e = pd.to_numeric(work["ebitda"], errors="coerce")
            r = pd.to_numeric(work["revenue"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                work["ebitda_margin_calc"] = np.where((r > 0) & np.isfinite(e), e / r, np.nan)

        metric_specs = [
            ("market_cap", "Market Cap", "size", "ratio_p75_p25", 5.0),
            ("revenue", "Revenue", "size", "ratio_p75_p25", 5.0),
            ("revenue_growth", "Revenue Growth (YoY)", "growth", "iqr", 0.15),
            ("earnings_growth", "Earnings Growth (YoY)", "growth", "iqr", 0.20),
            ("operating_margin", "Operating Margin", "margin", "iqr", 0.10),
            ("ebitda_margin_calc", "EBITDA Margin (calc)", "margin", "iqr", 0.12),
            ("debt_to_equity_info", "Debt/Equity (provider)", "leverage", "iqr", 100.0),
            ("beta", "Beta", "risk", "iqr", 0.6),
        ]

        for col, label, category, rule_type, threshold in metric_specs:
            if col not in work.columns:
                metrics_out.append({
                    "metric": col,
                    "label": label,
                    "category": category,
                    "n": 0,
                    "coverage": 0.0,
                    "status": "missing_column",
                    "rule_type": rule_type,
                    "threshold": threshold,
                })
                continue

            s = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(s))
            coverage = (n / n_peers) if n_peers > 0 else 0.0
            row: Dict[str, Any] = {
                "metric": col,
                "label": label,
                "category": category,
                "n": n,
                "coverage": round(float(coverage), 3),
                "rule_type": rule_type,
                "threshold": threshold,
            }

            if n == 0:
                row["status"] = "no_data"
                if coverage < coverage_warn_threshold:
                    warnings.append(f"{label}: low coverage ({n}/{n_peers}) may weaken comparability checks.")
                metrics_out.append(row)
                continue
            if n == 1:
                row["status"] = "warn_low_coverage" if coverage < coverage_warn_threshold else "insufficient_sample"
                row["median"] = float(s.iloc[0])
                if coverage < coverage_warn_threshold:
                    warnings.append(f"{label}: low coverage ({n}/{n_peers}) may weaken comparability checks.")
                metrics_out.append(row)
                continue

            p25 = float(np.percentile(s, 25))
            p50 = float(np.percentile(s, 50))
            p75 = float(np.percentile(s, 75))
            iqr = float(p75 - p25)
            row.update({"p25": p25, "median": p50, "p75": p75, "iqr": iqr})

            if rule_type == "ratio_p75_p25":
                denom = p25 if abs(p25) > 1e-9 else np.nan
                ratio = float(p75 / denom) if np.isfinite(denom) else None
                row["dispersion_ratio_p75_p25"] = ratio
                high_dispersion = bool(ratio is not None and ratio > threshold)
            else:
                row["dispersion_ratio_p75_p25"] = None
                high_dispersion = bool(iqr > threshold)

            low_coverage = coverage < coverage_warn_threshold
            if low_coverage and high_dispersion:
                row["status"] = "warn_low_coverage_and_high_dispersion"
            elif low_coverage:
                row["status"] = "warn_low_coverage"
            elif high_dispersion:
                row["status"] = "warn_high_dispersion"
            else:
                row["status"] = "ok"
            metrics_out.append(row)

            if low_coverage:
                warnings.append(f"{label}: low coverage ({n}/{n_peers}) may weaken comparability checks.")
            if high_dispersion:
                if rule_type == "ratio_p75_p25":
                    warnings.append(f"{label}: wide dispersion (P75/P25 > {threshold:.1f}x) suggests heterogeneous peers.")
                else:
                    warnings.append(f"{label}: wide dispersion (IQR > {threshold:.2f}) suggests heterogeneous peers.")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for w in warnings:
            if w not in seen:
                deduped.append(w)
                seen.add(w)

        return {
            "peer_count_for_stats": n_peers,
            "coverage_warn_threshold": coverage_warn_threshold,
            "metrics": metrics_out,
            "warnings": deduped,
        }

    peer_quality_diagnostics = _peer_quality_diagnostics(peer_only)

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
        "multiple_basis": multiple_basis,
        "metric_basis_map": metric_basis_map,
        "notes": notes,
        "peer_quality_diagnostics": peer_quality_diagnostics,
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
            equity = _equity_value_from_ev(
                ev,
                total_debt=total_debt,
                cash_eq=cash_eq,
                minority_interest=mi,
                net_debt=net_debt,
            )
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

__all__ = [
    'peer_multiples',
    'price_from_peer_multiples',
    '_price_snapshots',
    '_price_snapshots_ext',
]
