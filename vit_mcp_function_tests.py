import json
from pprint import pprint
import pandas as pd

from ValueInvestingTools import (
    peer_multiples,
    compare_to_market_ev,
    compare_to_market_cap,
    dcf_sensitivity_grid,
    to_records,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

analysis_date = "2026-02-23"


def _as_jsonable(obj, analysis_date):
    if isinstance(obj, pd.DataFrame):
        return to_records(obj, analysis_report_date=analysis_date)
    return obj


def show_dict(d, title=None, analysis_date=None):
    if title:
        print("\n" + title)
    payload = _as_jsonable(d, analysis_date) if analysis_date else d
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


# 1) peer_multiples (forward PE, exclude target)
peers = ["MSFT", "GOOGL", "AMZN", "AAPL"]
out_peer = peer_multiples(
    peers,
    target_ticker="AAPL",
    include_target=False,
    multiple_basis="forward_pe",
    analysis_report_date=analysis_date,
)
show_dict(out_peer, "peer_multiples (AAPL vs MSFT/GOOGL/AMZN, forward_pe)", analysis_date)
print("\npeer_quality_diagnostics:")
pprint(out_peer.get("peer_quality_diagnostics", {}))

# 2) compare_to_market_ev
out_ev = compare_to_market_ev("MSFT", analysis_report_date=analysis_date)
show_dict(out_ev, "compare_to_market_ev (MSFT)", analysis_date)
print("\nValuation_Confidence:")
pprint(out_ev.get("data", [{}])[0].get("Valuation_Confidence", {}))

# 3) compare_to_market_cap
out_cap = compare_to_market_cap("MSFT", analysis_report_date=analysis_date)
show_dict(out_cap, "compare_to_market_cap (MSFT)", analysis_date)
conf = out_cap.get("data", [{}])[0].get("Valuation_Confidence", {})
assumptions = out_cap.get("data", [{}])[0].get("Assumptions_Used", {})
print("\nValuation_Confidence:")
pprint(conf)
print("\nassumptions_snapshot_id:", assumptions.get("assumptions_snapshot_id"))

# 4) dcf_sensitivity_grid
out_grid = dcf_sensitivity_grid("MSFT", analysis_report_date=analysis_date, as_df=False)
show_dict(out_grid, "dcf_sensitivity_grid (MSFT)", analysis_date)
print("\nGrid wide keys:", list(out_grid.get("grid_wide", {}).keys())[:10])
print("Grid long sample rows:")
long = out_grid.get("grid_long", {})
rows = long.get("data", []) if isinstance(long, dict) else (long or [])
for row in rows[:5]:
    print(row)
