from __future__ import annotations

import inspect
import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd

from vitlib.utils import _today_iso, to_records, VALUATION_DEFAULTS
from vitlib.fundamentals import (
    historical_average_share_prices,
    historical_growth_metrics,
    compute_fundamentals_actuals,
    compute_fundamentals_scores,
)
from vitlib.peers import peer_multiples, price_from_peer_multiples
from vitlib.valuation import compare_to_market_ev, compare_to_market_cap, dcf_three_scenarios
from vitlib.health import health_to_tables
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


__all__ = [
    'make_peer_metric_frame',
    'estimate_company_value',
    'orchestrator_function',
]
