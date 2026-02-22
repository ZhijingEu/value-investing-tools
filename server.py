#!/usr/bin/env python3
"""
ValueInvestingTools — MCP server (STDIO) for Claude Desktop
Exposes one tool per library function (granular, verbose) — NO orchestrator.

Transport: STDIO (default). Claude Desktop launches this process and speaks MCP over stdio.
Artifacts: PNG/CSV/XLSX written under ./output/<TICKER>/ (created on demand).

Install (example):
  pip install "mcp>=1.2.0" pandas matplotlib yfinance

If you prefer the third-party 'fastmcp' package instead of the official 'mcp' package,
this file will also work — it falls back to 'fastmcp' if 'mcp' is unavailable.
"""

from __future__ import annotations

import os
import io
import json
import base64
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Union

# Headless plotting for server environments
import matplotlib
matplotlib.use("Agg")

# IMPORTANT: define HERE before using it anywhere
HERE = Path(__file__).resolve().parent

# Prefer official MCP server; fall back to third-party fastmcp if needed
try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
    MCP_IMPL = "mcp.server.fastmcp"
except Exception:
    from fastmcp import FastMCP  # type: ignore
    MCP_IMPL = "fastmcp"

# ---------- Load the library module ----------

import vit  # wrapper 

# ---------- Config ----------
OUTPUT_ROOT = Path(os.environ.get("VIT_OUTPUT_DIR", HERE / "output")).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

INLINE_IMAGES = os.environ.get("INLINE_IMAGES", "0") not in ("", "0", "false", "False")
MAX_INLINE_IMAGE_BYTES = int(os.environ.get("INLINE_IMAGE_MAX_BYTES", "2000000"))  # ~2 MB
# (INLINE_MARKDOWN / INLINE_DATA_URI are not needed for Claude Desktop)

# ---------- Helpers ----------
def ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def as_uri(p: Path) -> str:
    return p.resolve().as_uri()

def write_df_csv(df, path: Path) -> str:
    ensure_dir(path)
    df.to_csv(path, index=False)
    return as_uri(path)

def write_text(text: str, path: Path, *, encoding="utf-8") -> str:
    ensure_dir(path)
    path.write_text(text, encoding=encoding)
    return as_uri(path)

def file_resource(p: Any, mime: str) -> Dict[str, Any]:
    """
    Accepts either a Path or a string. If it's already a file:// URI, pass it through.
    Otherwise resolve the filesystem path and convert to a proper file URI.
    """
    if isinstance(p, Path):
        uri = p.resolve().as_uri()
    else:
        s = str(p)
        if s.startswith("file://"):
            uri = s
        else:
            uri = Path(s).resolve().as_uri()
    return {"type": "resource", "uri": uri, "mimeType": mime}

def text_item(s: str) -> Dict[str, Any]:
    return {"type": "text", "text": s}

def image_items_for_png(path: Path):
    """
    Return exactly one content item so Claude Desktop can inline it.
    If INLINE_IMAGES=1 and the PNG is <= MAX_INLINE_IMAGE_BYTES, return an `image` item (base64).
    Otherwise return a single `resource` item (file:// URI).
    """
    INLINE_IMAGES = False  # force file:// resource returns for now
    if INLINE_IMAGES:
        raw = path.read_bytes()
        if MAX_INLINE_IMAGE_BYTES and len(raw) > MAX_INLINE_IMAGE_BYTES:
            return [{"type": "resource", "mimeType": "image/png", "uri": as_uri(path)}]
        b64 = base64.b64encode(raw).decode("ascii")
        return [{"type": "image", "mimeType": "image/png", "data": b64}]
    return [{"type": "resource", "mimeType": "image/png", "uri": as_uri(path)}]

def ticker_dir(ticker: str) -> Path:
    t = (ticker or "GEN").replace(".", "-").upper().strip()
    d = OUTPUT_ROOT / t
    d.mkdir(parents=True, exist_ok=True)
    return d

def to_records_df(df) -> List[Dict[str, Any]]:
    """JSON-serializable records from a DataFrame."""
    return json.loads(df.to_json(orient="records"))

# ---------- App ----------
app = FastMCP(f"ValueInvestingTools (STDIO) — using {MCP_IMPL}")

# ============================
# General
# ============================
@app.tool(name="_price_snapshots", description="Return short price snapshot for a ticker (1d avg, as-of date, notes).")
def tool_price_snapshots(ticker: str):
    """Uses the legacy triple (avg_1d, price_asof, notes) for compatibility, and also returns the extended dict."""
    a1d, asof, notes = vit._price_snapshots(ticker)
    ext = vit._price_snapshots_ext(ticker)
    return [
        text_item(f"{ticker}: avg_price_1d={a1d}, asof={asof}"),
        text_item(json.dumps(ext, indent=2))
    ]

@app.tool(name="historical_average_share_prices", description="Average share prices over 1/30/90/180 days for tickers.")
def tool_hist_avg_px(tickers: Union[str, List[str]], analysis_report_date: Optional[str]=None, save_csv: bool=False):
    if isinstance(tickers, str):
        tickers = [tickers]
    df = vit.historical_average_share_prices(tickers, analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    content = [text_item(f"Historical average prices for {', '.join(tickers)}")]
    content.append(text_item(json.dumps(to_records_df(df), indent=2)))
    if save_csv:
        out = OUTPUT_ROOT / "AVG_PRICES" / f"avg_prices_{analysis_report_date or 'today'}.csv"
        uri = write_df_csv(df, out)
        content.append({"type": "resource", "uri": uri, "mimeType": "text/csv"})
    return content

@app.tool(name="historical_growth_metrics", description="Compute historical CAGRs (Revenue, Net Income, FCF) per ticker.")
def tool_hist_growth(tickers: Union[str, List[str]], min_years: int=3, analysis_report_date: Optional[str]=None, save_csv: bool=False):
    if isinstance(tickers, str):
        tickers = [tickers]
    df = vit.historical_growth_metrics(tickers, min_years=min_years, analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    content = [text_item(f"Growth CAGRs for {', '.join(tickers)} (min_years={min_years})")]
    content.append(text_item(json.dumps(to_records_df(df), indent=2)))
    if save_csv:
        out = OUTPUT_ROOT / "GROWTH" / f"growth_{analysis_report_date or 'today'}.csv"
        uri = write_df_csv(df, out)
        content.append({"type": "resource", "uri": uri, "mimeType": "text/csv"})
    return content

# ============================
# Fundamentals
# ============================
@app.tool(name="compute_fundamentals_actuals", description="Compute fundamentals (absolute) for tickers.")
def tool_compute_actuals(tickers: List[str], basis: Literal["annual","ttm"]="annual", save_csv: bool=False, analysis_report_date: Optional[str]=None):
    df = vit.compute_fundamentals_actuals(tickers, basis=basis, save_csv=False, as_df=True, analysis_report_date=analysis_report_date)
    content = [text_item(f"Fundamentals actuals for {', '.join(tickers)} (basis={basis})")]
    content.append(text_item(json.dumps(to_records_df(df), indent=2)))
    if save_csv:
        for t in tickers:
            out = ticker_dir(t) / f"fundamentals_actuals_{basis}.csv"
            write_df_csv(df[df['ticker']==t] if 'ticker' in df.columns else df, out)
            content.append(file_resource(out, "text/csv"))
    return content

@app.tool(name="compute_fundamentals_scores", description="Score fundamentals for tickers (or merge with given actuals).")
def tool_compute_scores(tickers: Optional[List[str]]=None, basis: Literal["annual","ttm"]="annual", merge_with_actuals: bool=False, analysis_report_date: Optional[str]=None):
    # For MCP ergonomics we take 'tickers'; library supports DF/list/strings
    if not tickers:
        raise ValueError("Please provide tickers (list[str]).")
    scores = vit.compute_fundamentals_scores(tickers, basis=basis, merge_with_actuals=merge_with_actuals, as_df=True, analysis_report_date=analysis_report_date)
    content = [text_item(f"Fundamentals scores for {', '.join(tickers)} (basis={basis}, merge={merge_with_actuals})")]
    content.append(text_item(json.dumps(to_records_df(scores), indent=2)))
    return content

@app.tool(name="plot_single_metric_ts", description="Plot a single metric over time for one ticker; returns PNG resource.")
def tool_plot_single_metric_ts(ticker: str, metric: str, family: Literal["Profitability","Growth","Reinvestment","Risk"]="Profitability", basis: Literal["annual","ttm"]="annual", save_png: bool=True):
    out = ticker_dir(ticker) / f"{ticker}_ts_{metric}_{basis}.png"
    fig, ax = vit.plot_single_metric_ts(ticker, metric, family=family, basis=basis)
    if fig is None:
        return [text_item(f"No data for {ticker} / {family} / {metric}")]
    fig.savefig(ensure_dir(out), bbox_inches="tight")
    #return [text_item(f"{ticker} — {metric} ({basis})"), *image_items_for_png(out)] #commented out as Claude DT only tends to only render the first thing (your caption), and then shows the rest as separate “Response” blocks instead of actually inlining the chart.
    return image_items_for_png(out)

@app.tool(name="plot_metrics_family_ts", description="Plot multiple metrics (family) over time for one ticker; returns PNG.")
def tool_plot_metrics_family_ts(ticker: str, family: Literal["Profitability","Growth","Reinvestment","Risk"]="Profitability", metrics: Optional[List[str]]=None, basis: Literal["annual","ttm"]="annual"):
    out = ticker_dir(ticker) / f"{ticker}_ts_{family}_{basis}.png"
    fig, ax = vit.plot_metrics_family_ts(ticker, family=family, metrics=metrics, basis=basis, save_path=str(out))
    if fig is None:
        return [text_item(f"No data for {ticker} / {family}")]
    #return [text_item(f"{ticker} — {family} ({basis})"), *image_items_for_png(out)] #commented out as Claude DT only tends to only render the first thing (your caption), and then shows the rest as separate “Response” blocks instead of actually inlining the chart.
    return image_items_for_png(out)

@app.tool(name="plot_multi_tickers_multi_metrics_ts", description="Plot multiple tickers * multiple metrics (family) over time; returns PNG.")
def tool_plot_multi_tickers_multi_metrics_ts(tickers: List[str], family: Literal["Profitability","Growth","Reinvestment","Risk"]="Profitability", metrics: Optional[List[str]]=None, basis: Literal["annual","ttm"]="annual"):
    tag = f"{family}_{basis}"
    out = OUTPUT_ROOT / "MULTI" / f"{tag}_{'-'.join([t.upper() for t in tickers])}.png"
    fig, axes = vit.plot_multi_tickers_multi_metrics_ts(tickers, family=family, metrics=metrics, basis=basis, save_path=str(out))
    if fig is None:
        return [text_item("No data for requested tickers/family/metrics")]
    #return [text_item(f"Multi-ticker {family} ({basis})"), *image_items_for_png(out)] #commented out as Claude DT only tends to only render the first thing (your caption), and then shows the rest as separate “Response” blocks instead of actually inlining the chart.
    return image_items_for_png(out)

@app.tool(name="plot_scores_clustered", description="Clustered bar chart of fundamentals scores for tickers; returns PNG.")
def tool_plot_scores_clustered(tickers: List[str], basis: Literal["annual","ttm"]="annual", metrics: Optional[List[str]]=None, include_total: bool=True, sort_by: Literal["family","avg","name","none"]="family", title: Optional[str]=None):
    out = OUTPUT_ROOT / "SCORES" / f"scores_{'-'.join([t.upper() for t in tickers])}_{basis}.png"
    fig, axes = vit.plot_scores_clustered(tickers, basis=basis, metrics=metrics, include_total=include_total, sort_by=sort_by, title=title)
    fig.savefig(ensure_dir(out), bbox_inches="tight")
    #return [text_item(f"Scores clustered chart for {', '.join(tickers)}"), *image_items_for_png(out)] #commented out as Claude DT only tends to only render the first thing (your caption), and then shows the rest as separate “Response” blocks instead of actually inlining the chart.
    return image_items_for_png(out)

# ============================
# Market Comparison (Price-to-X ratios)
# ============================
@app.tool(name="peer_multiples", description="Build peer multiples & valuation bands for a peer set (requires target_ticker).")
def tool_peer_multiples(tickers: List[str], target_ticker: str, include_target: bool=False, multiple_basis: Literal["ttm","forward_pe"]="ttm", analysis_report_date: Optional[str]=None):
    res = vit.peer_multiples(tickers, target_ticker=target_ticker, include_target=include_target, multiple_basis=multiple_basis, as_df=True, analysis_report_date=analysis_report_date)
    pcd = res["peer_comp_detail"]; bands_wide = res["peer_multiple_bands_wide"]; comp_bands = res["peer_comp_bands"]
    # Write CSVs
    base = ticker_dir(target_ticker)
    uri1 = write_df_csv(pcd, base / f"peer_comp_detail_{target_ticker}.csv")
    uri2 = write_df_csv(bands_wide, base / f"peer_multiple_bands_wide_{target_ticker}.csv")
    uri3 = write_df_csv(comp_bands, base / f"peer_comp_bands_{target_ticker}.csv")
    # Also return a compact JSON
    content = [text_item(f"Peer multiples for target={target_ticker} (peers={', '.join(tickers)}, multiple_basis={multiple_basis})")]
    content.extend([
        file_resource(uri1, "text/csv"),
        file_resource(uri2, "text/csv"),
        file_resource(uri3, "text/csv"),
    ])
    # Lightweight JSON summary (bands wide)
    content.append(text_item(json.dumps({"bands_index": list(map(str, bands_wide.index)), "columns": list(map(str, bands_wide.columns))}, indent=2)))
    # Return the full dict structure too (as text) so it can be piped into price_from_peer_multiples if needed
    content.append(text_item(json.dumps({
        "target_ticker": res.get("target_ticker"),
        "multiple_basis": res.get("multiple_basis"),
        "metric_basis_map": res.get("metric_basis_map"),
        "notes": res.get("notes"),
        "peer_quality_diagnostics": res.get("peer_quality_diagnostics"),
        "peer_comp_detail": to_records_df(pcd),
        "peer_multiple_bands_wide": json.loads(bands_wide.to_json()),
        "peer_comp_bands": to_records_df(comp_bands)
    })[:120000]))  # guard oversized messages
    return content

@app.tool(name="price_from_peer_multiples", description="Compute implied per-share price bands (P25/P50/P75) from peer_multiples output.")
def tool_price_from_peer_multiples(peer_multiples_json: Optional[Dict[str, Any]]=None, tickers: Optional[List[str]]=None, target_ticker: Optional[str]=None, include_target: bool=False, multiple_basis: Literal["ttm","forward_pe"]="ttm", analysis_report_date: Optional[str]=None, save_csv: bool=False):
    if peer_multiples_json is None:
        if not (tickers and target_ticker):
            raise ValueError("Provide either peer_multiples_json OR (tickers + target_ticker).")
        pm = vit.peer_multiples(tickers, target_ticker=target_ticker, include_target=include_target, multiple_basis=multiple_basis, as_df=True, analysis_report_date=analysis_report_date)
    else:
        pm = peer_multiples_json
    out_df = vit.price_from_peer_multiples(pm, ticker=target_ticker, analysis_report_date=analysis_report_date, save_csv=False, as_df=True)
    content = [text_item("Implied price bands from peers"), text_item(json.dumps(to_records_df(out_df), indent=2))]
    if save_csv:
        t = (target_ticker or pm.get("target_ticker") or "TARGET").upper()
        uri = write_df_csv(out_df, ticker_dir(t) / f"price_from_peers_{t}.csv")
        content.append({"type": "resource", "uri": uri, "mimeType": "text/csv"})
    return content

@app.tool(name="plot_peer_metric_boxplot", description="Boxplot of peer metric (PE|PS|EV_EBITDA) with target overlay; returns PNG.")
def tool_plot_peer_metric_boxplot(tickers: List[str], target_ticker: str, metric: Literal["PE","PS","EV_EBITDA"]="PE", include_target_in_stats: bool=False, multiple_basis: Literal["ttm","forward_pe"]="ttm"):
    peers_all = sorted({target_ticker.upper(), *(t.upper() for t in tickers)})
    pm = vit.peer_multiples(peers_all, target_ticker=target_ticker, include_target=include_target_in_stats, multiple_basis=multiple_basis, as_df=True)
    pcd = pm["peer_comp_detail"]; bands = pm["peer_multiple_bands_wide"]
    out = ticker_dir(target_ticker) / f"peer_{metric}_box.png"
    vit.plot_peer_metric_boxplot(peer_comp_detail=pcd, peer_multiple_bands_wide=bands, metric=metric, target_ticker=target_ticker, include_target_in_stats=include_target_in_stats, save_path=str(out))
    #return [text_item(f"Peer {metric} distribution (target={target_ticker})"), *image_items_for_png(out)] # see earlier comments for other plotter functions
    return image_items_for_png(out)

# ============================
# EV / Market Cap
# ============================
@app.tool(name="compare_to_market_ev", description="Compute implied EV vs observed enterpriseValue from Yahoo.")
def tool_compare_to_market_ev(ticker: str, years: Optional[int]=None, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, growth: Optional[float]=None, target_cagr_fallback: float=0.02, use_average_fcf_years: Optional[int]=3, volatility_threshold: float=0.5, assumptions_as_of: Optional[str]=None):
    df = vit.compare_to_market_ev(ticker, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, growth=growth, target_cagr_fallback=target_cagr_fallback, use_average_fcf_years=use_average_fcf_years, volatility_threshold=volatility_threshold, assumptions_as_of=assumptions_as_of, as_df=True)
    return [text_item(json.dumps(to_records_df(df), indent=2))]

@app.tool(name="plot_ev_observed_vs_implied", description="Plot observed EV vs implied EV (expects output from compare_to_market_ev); returns PNG.")
def tool_plot_ev_observed_vs_implied(ticker: Optional[str]=None, ev_df_json: Optional[List[Dict[str, Any]]]=None, years: Optional[int]=None, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, growth: Optional[float]=None, target_cagr_fallback: float=0.02, use_average_fcf_years: Optional[int]=3, volatility_threshold: float=0.5, assumptions_as_of: Optional[str]=None):
    if ev_df_json is None:
        if not ticker:
            raise ValueError("Provide either ev_df_json OR ticker.")
        ev_df = vit.compare_to_market_ev(ticker, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, growth=growth, target_cagr_fallback=target_cagr_fallback, use_average_fcf_years=use_average_fcf_years, volatility_threshold=volatility_threshold, assumptions_as_of=assumptions_as_of, as_df=True)
    else:
        import pandas as pd
        ev_df = pd.DataFrame(ev_df_json)
    t = ticker or (ev_df["ticker"].iloc[0] if "ticker" in ev_df.columns else "TICKER")
    out = ticker_dir(str(t)) / f"{t}_ev_observed_vs_implied.png"
    vit.plot_ev_observed_vs_implied(ev_df, save_path=str(out))
    #return [text_item(f"Observed vs implied EV for {t}"), *image_items_for_png(out)] # see earlier comments for other plotter functions
    return image_items_for_png(out)

@app.tool(name="compare_to_market_cap", description="Compare implied EQUITY VALUE vs observed MARKET CAP (Yahoo).")
def tool_compare_to_market_cap(ticker_or_evdf: Union[str, List[Dict[str, Any]]], years: Optional[int]=None, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, growth: Optional[float]=None, target_cagr_fallback: float=0.02, use_average_fcf_years: Optional[int]=3, volatility_threshold: float=0.5, assumptions_as_of: Optional[str]=None):
    # Accept a ticker string OR an ev_df_json (list[dict]) from compare_to_market_ev
    import pandas as pd
    if isinstance(ticker_or_evdf, str):
        df = vit.compare_to_market_cap(ticker_or_evdf, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, growth=growth, target_cagr_fallback=target_cagr_fallback, use_average_fcf_years=use_average_fcf_years, volatility_threshold=volatility_threshold, assumptions_as_of=assumptions_as_of, as_df=True)
    else:
        ev_df = pd.DataFrame(ticker_or_evdf)
        df = vit.compare_to_market_cap(ev_df, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, growth=growth, target_cagr_fallback=target_cagr_fallback, use_average_fcf_years=use_average_fcf_years, volatility_threshold=volatility_threshold, assumptions_as_of=assumptions_as_of, as_df=True)
    return [text_item(json.dumps(to_records_df(df), indent=2))]

@app.tool(name="plot_market_cap_observed_vs_implied_equity_val", description="Plot observed Market Cap vs implied Equity Value; returns PNG.")
def tool_plot_mktcap_vs_implied_equity(ticker: Optional[str]=None, evcap_df_json: Optional[List[Dict[str, Any]]]=None, years: Optional[int]=None, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, growth: Optional[float]=None, target_cagr_fallback: float=0.02, use_average_fcf_years: Optional[int]=3, volatility_threshold: float=0.5, assumptions_as_of: Optional[str]=None):
    import pandas as pd
    if evcap_df_json is None:
        if not ticker:
            raise ValueError("Provide either evcap_df_json OR ticker.")
        evcap_df = vit.compare_to_market_cap(ticker, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, growth=growth, target_cagr_fallback=target_cagr_fallback, use_average_fcf_years=use_average_fcf_years, volatility_threshold=volatility_threshold, assumptions_as_of=assumptions_as_of, as_df=True)
    else:
        evcap_df = pd.DataFrame(evcap_df_json)
    t = ticker or (evcap_df["ticker"].iloc[0] if "ticker" in evcap_df.columns else "TICKER")
    out = ticker_dir(str(t)) / f"{t}_mktcap_vs_implied_equity.png"
    vit.plot_market_cap_observed_vs_implied_equity_val(evcap_df, save_path=str(out))
    #return [text_item(f"Observed Market Cap vs Implied Equity Value for {t}"), *image_items_for_png(out)] # see earlier comments for other plotter functions
    return image_items_for_png(out)

# ============================
# DCF
# ============================
@app.tool(name="dcf_three_scenarios", description="Compute Low/Mid/High per-share DCF for a ticker.")
def tool_dcf_three_scenarios(ticker: str, peer_tickers: Optional[List[str]]=None, years: int=5, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, target_cagr_fallback: float=0.02, fcf_window_years: Optional[int]=3, manual_baseline_fcf: Optional[float]=None, manual_growth_rates: Optional[List[float]]=None, assumptions_as_of: Optional[str]=None):
    df = vit.dcf_three_scenarios(ticker, peer_tickers=peer_tickers, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, target_cagr_fallback=target_cagr_fallback, fcf_window_years=fcf_window_years, manual_baseline_fcf=manual_baseline_fcf, manual_growth_rates=manual_growth_rates, assumptions_as_of=assumptions_as_of, as_df=True)
    return [text_item(json.dumps(to_records_df(df), indent=2))]

@app.tool(name="dcf_sensitivity_grid", description="Build DCF sensitivity table (WACC x terminal growth) as wide + long outputs.")
def tool_dcf_sensitivity_grid(
    ticker: str,
    years: int=5,
    risk_free_rate: float=0.045,
    equity_risk_premium: float=0.060,
    growth: Optional[float]=None,
    target_cagr_fallback: float=0.02,
    use_average_fcf_years: Optional[int]=3,
    assumptions_as_of: Optional[str]=None,
    wacc_values: Optional[List[float]]=None,
    growth_values: Optional[List[float]]=None,
    save_csv: bool=False,
):
    res = vit.dcf_sensitivity_grid(
        ticker,
        years=years,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        growth=growth,
        target_cagr_fallback=target_cagr_fallback,
        use_average_fcf_years=use_average_fcf_years,
        assumptions_as_of=assumptions_as_of,
        wacc_values=wacc_values,
        growth_values=growth_values,
        as_df=True,
    )
    grid_long = res["grid_long"]
    grid_wide = res["grid_wide"]
    content = [text_item(f"DCF sensitivity grid for {ticker} (rows=growth, cols=WACC)")]
    payload = {
        "ticker": res.get("ticker"),
        "analysis_report_date": res.get("analysis_report_date"),
        "inputs_used": res.get("inputs_used"),
        "notes": res.get("notes"),
        "grid_wide": json.loads(grid_wide.to_json()),
        "grid_long": to_records_df(grid_long),
    }
    content.append(text_item(json.dumps(payload)[:120000]))
    if save_csv:
        base = ticker_dir(ticker)
        uri_wide = write_df_csv(grid_wide.reset_index(), base / f"dcf_sensitivity_grid_wide_{ticker}.csv")
        uri_long = write_df_csv(grid_long, base / f"dcf_sensitivity_grid_long_{ticker}.csv")
        content.append(file_resource(uri_wide, "text/csv"))
        content.append(file_resource(uri_long, "text/csv"))
    return content

@app.tool(name="plot_dcf_scenarios_vs_price", description="Plot DCF scenarios vs current price; returns PNG.")
def tool_plot_dcf_vs_price(ticker: str, peer_tickers: Optional[List[str]]=None, years: int=5, risk_free_rate: float=0.045, equity_risk_premium: float=0.060, target_cagr_fallback: float=0.02, fcf_window_years: Optional[int]=3, manual_baseline_fcf: Optional[float]=None, manual_growth_rates: Optional[List[float]]=None, assumptions_as_of: Optional[str]=None):
    # Build audit DF via dcf_three_scenarios and get current price snapshot
    audit_df = vit.dcf_three_scenarios(ticker, peer_tickers=peer_tickers, years=years, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, target_cagr_fallback=target_cagr_fallback, fcf_window_years=fcf_window_years, manual_baseline_fcf=manual_baseline_fcf, manual_growth_rates=manual_growth_rates, assumptions_as_of=assumptions_as_of, as_df=True)
    p1d, _, _ = vit._price_snapshots(ticker)
    out = ticker_dir(ticker) / f"{ticker}_dcf_vs_price.png"
    vit.plot_dcf_scenarios_vs_price(audit_df, p1d, save_path=str(out))
    #return [text_item(f"DCF vs Price for {ticker}"), *image_items_for_png(out)] # see earlier comments for other plotter functions
    return image_items_for_png(out)

# ============================
# Health
# ============================
@app.tool(name="health_to_tables", description="Normalize a health report (new or legacy schema) into a tidy table.")
def tool_health_to_tables(health_report: Any, sort: bool=True):
    df = vit.health_to_tables(health_report, sort=sort, as_df=True)
    return [text_item(json.dumps(to_records_df(df), indent=2))]

@app.tool(name="health_to_markdown", description="Render a health report to Markdown.")
def tool_health_to_markdown(health_report: Any):
    md = vit.health_to_markdown(health_report)
    return [text_item(md)]

# -------------- main --------------
if __name__ == "__main__":
    import sys, traceback
    try:
        # STDIO is the default transport for both MCP implementations used above.
        app.run()
    except KeyboardInterrupt:
        # graceful shutdown on Ctrl+C or client-initiated stop
        pass
    except Exception as e:
        # make sure errors show up in Claude Desktop's log
        print("FATAL in server.py:", e, file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        raise  # or: sys.exit(1)
