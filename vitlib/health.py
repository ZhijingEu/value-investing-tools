from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import pandas as pd

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

__all__ = [
    'health_to_tables',
    'health_to_markdown',
    'save_health_report_excel',
]
