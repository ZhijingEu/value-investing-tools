from __future__ import annotations

import math
import re
from typing import List, Tuple, Optional, Dict, Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vitlib.utils import _is_num, _safe_float, _today_iso, _provider_ticker
from vitlib.fundamentals import compute_profitability_timeseries, compute_liquidity_timeseries
from vitlib.peers import _price_snapshots
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

    # Accept legacy aliases from older callers while keeping the public contract stable.
    if sort_by == "metric":
        sort_by = "name"
    elif sort_by == "ticker":
        sort_by = "name"

    # Decide final order
    if sort_by == "family":
        # Primary: family order; Secondary: within-family order; Tertiary: metric label as fallback
        long_df = long_df.sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])
        ordered_labels = (
            long_df
            .drop_duplicates(subset=["Metric", "MetricLabel"])
            .sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])["MetricLabel"]
            .tolist()
        )
    elif sort_by == "name":
        long_df = long_df.sort_values(["MetricLabel"])
        ordered_labels = (
            long_df
            .drop_duplicates(subset=["Metric", "MetricLabel"])
            .sort_values(["MetricLabel"])["MetricLabel"]
            .tolist()
        )
    elif sort_by == "avg":
        # Compute metric means and order by descending average
        means = long_df.groupby("Metric", observed=False)["Score"].mean().sort_values(ascending=False)
        long_df["__avg_order__"] = long_df["Metric"].map(means.to_dict())
        long_df = long_df.sort_values(["__avg_order__", "MetricLabel"])
        ordered_labels = (
            long_df
            .drop_duplicates(subset=["Metric", "MetricLabel"])
            .sort_values(["__avg_order__", "MetricLabel"])["MetricLabel"]
            .tolist()
        )
    elif sort_by == "none":
        # Preserve caller metric order, useful for deterministic report templates.
        metric_to_label = (
            long_df.drop_duplicates(subset=["Metric", "MetricLabel"])
            .set_index("Metric")["MetricLabel"]
            .to_dict()
        )
        ordered_labels = [metric_to_label[m] for m in selected_cols if m in metric_to_label]
    else:
        # default to family if unknown
        long_df = long_df.sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])
        ordered_labels = (
            long_df
            .drop_duplicates(subset=["Metric", "MetricLabel"])
            .sort_values(["__fam_idx__", "__within_idx__", "MetricLabel"])["MetricLabel"]
            .tolist()
        )

    # Lock x-axis order via an ordered Categorical on MetricLabel
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
        tkr = _provider_ticker(ticker)
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
        tkr = _provider_ticker(ticker)
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

__all__ = [
    'plot_peer_metric_boxplot',
    'plot_ev_observed_vs_implied',
    'plot_market_cap_observed_vs_implied_equity_val',
    'plot_dcf_scenarios_vs_price',
    'plot_metrics_multiples',
    'plot_scores_clustered',
    'metric_rank_key',
    'plot_single_metric_ts',
    'plot_metrics_family_ts',
    'plot_multi_tickers_multi_metrics_ts',
]
