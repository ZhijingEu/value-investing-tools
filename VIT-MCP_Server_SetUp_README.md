# Value Investing Tools — MCP Server Guide (STDIO MCP)

This repo ships with a local Model Context Protocol (MCP) server that exposes a curated subset of VIT functions over STDIO. It works with Claude Desktop, Codex, and other MCP-capable clients. The server produces file URIs for artifacts (PNGs/CSVs/XLSX) and leaves the full Python API available via `import vit`.

**Note:** The long-running orchestrator function is intentionally not exposed as an MCP tool. Use the granular tools below or the Python API.

Inline chart rendering from the Value-Investing MCP tools is currently unreliable in some clients (including Claude Desktop).  
Plots **are still generated** and written as PNGs to your configured output folder (`VIT_OUTPUT_DIR`, e.g., `./output/<TICKER>/…`). In this state, the image may not appear inline in the chat even though the tool returns it.

**Workarounds (until the patch lands):**
- Open the saved PNG(s) directly from the `output` folder or from the file URI returned by the tool.  
- Ask the assistant to **embed the returned image** using Markdown (e.g., `![title](data:image/png;base64,...)` or `![title](file:///path/to.png)`), then provide the analysis.  
- Or ask the assistant to **plot the results another way** (e.g., a different rendering approach or a simple data table) — this usually works.

Thanks for your patience while we ship a proper inline-image fix.

## What's exposed

### Data & fundamentals

- `_price_snapshots` — 1-day average price, as-of date, notes.
- `historical_average_share_prices` — average prices over 1/30/90/180 days.
- `historical_growth_metrics` — multi-year CAGRs (revenue, net income, FCF).
- `compute_fundamentals_actuals` — normalized fundamentals (annual / TTM).
- `compute_fundamentals_scores` — growth/profitability/reinvestment/risk scoring (optionally merged with actuals).

### Charts (PNG written to `./output/<TICKER>/…`)

- `plot_single_metric_ts` — one metric over time for one ticker.
- `plot_metrics_family_ts` — a family of metrics over time for one ticker.
- `plot_multi_tickers_multi_metrics_ts` — compare a family across multiple tickers.
- `plot_scores_clustered` — clustered bar chart of fundamentals scores.
- `plot_peer_metric_boxplot` — PE/PS/EV_EBITDA distribution with target overlay.
- `plot_ev_observed_vs_implied` — observed EV vs implied EV (expects `compare_to_market_ev` output).
- `plot_market_cap_observed_vs_implied_equity_val` — observed market cap vs implied equity value (expects `compare_to_market_cap` output).
- `plot_dcf_scenarios_vs_price` — DCF scenarios (Low/Mid/High) versus current price.

### Peers / valuation

- `peer_multiples` — per-peer ratios + valuation bands (detail & wide).
- `price_from_peer_multiples` — implied per-share P25/P50/P75 from peer bands.
- `compare_to_market_ev` — implied EV vs observed EV.
- `compare_to_market_cap` — implied equity value vs observed market cap.
- `dcf_sensitivity_grid` — WACC x terminal growth sensitivity grid (wide + long).

### Health

- `health_to_tables` — normalize a health report into a tidy table.
- `health_to_markdown` — render a health report to Markdown.

## Installation (Windows example)

```bash
# 1) clone / open the project folder
cd C:\Users\<yourUserName>\OneDrive\Desktop\MCP_Servers\value-investing-tools-mcp

# 2) create a virtual environment
py -3 -m venv .\vit-env
.\vit-env\Scripts\Activate

# 3) install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4) quick import sanity check
python -c "import vit; print('vit OK:', hasattr(vit, 'compute_fundamentals_actuals'))"

# 5) quick data sanity check (non-MCP)
python -c "import ValueInvestingTools as vit; print(vit.fundamentals_ttm_vs_average('MSFT').head())"
```

### Optional (versioned filename)

If you keep a versioned library filename, set an env var so vit finds it:

```powershell
$env:VIT_LIB_BASENAME="ValueInvestingTools.py"
```

(Alternatively, rename the file to `ValueInvestingTools.py` — the wrapper looks for both.)

## Configure Claude Desktop (Local MCP server)

Open **Settings → Developer → Local MCP Servers** and add one of the two configurations:

### Option A: absolute path to server.py

```json
{
  "mcpServers": {
    "value-investing-tools": {
      "command": "C:\\Users\\<yourUserName>\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp\\vit-env\\Scripts\\python.exe",
      "args": ["C:\\Users\\<yourUserName>\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp\\server.py"],
      "env": {
        "VIT_OUTPUT_DIR": "C:\\Users\\<yourUserName>\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp\\output",
        "VIT_LIB_BASENAME": "ValueInvestingTools.py",
        "INLINE_IMAGES": "0"
      }
    }
  }
}
```

### Option B: relative script + cwd

```json
{
  "mcpServers": {
    "value-investing-tools": {
      "command": "C:\\Users\\<yourUserName>\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp\\vit-env\\Scripts\\python.exe",
      "args": ["server.py"],
      "cwd": "C:\\Users\\<yourUserName>\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp",
      "env": {
        "VIT_OUTPUT_DIR": "C:\\Users\\zhiji\\OneDrive\\Desktop\\MCP_Servers\\value-investing-tools-mcp\\output",
        "VIT_LIB_BASENAME": "ValueInvestingTools.py",
        "INLINE_IMAGES": "1",
        "INLINE_IMAGE_MAX_BYTES": "2000000"
      }
    }
  }
}
```
Note: `INLINE_IMAGES=1` tells the server to include a base64 image item in addition to the file URI. If you prefer file-only artifacts set it to `0`.

After saving, fully quit and relaunch Claude Desktop so it reloads the local servers.

## How it works (I/O & artifacts)

- **Transport:** STDIO (clients launch the process and speak MCP over stdin/stdout).
- **Images:** plotting tools save PNGs to `./output/<TICKER>/…` and return absolute file URIs (`file:///C:/…`) so Claude can display them inline.
- **Tables:** data tools return JSON in chat; many also write CSV and attach them as resources.
- **Excel:** when produced (e.g., health reports), a file resource to `.xlsx` is returned.
- **Matplotlib:** the server selects the headless backend (Agg) so charts render without a display.

## Other MCP clients (non-Claude)

Any MCP-capable client that can launch a local STDIO server can use this tool. The only requirement is a command+args configuration equivalent to:

```bash
<python> server.py
```

You may also set:
- `VIT_OUTPUT_DIR` to control artifact location.
- `VIT_LIB_BASENAME` if you renamed the library file.

## Natural-language starter prompts

You don't need to name tools explicitly — try phrasing like this:

### Fundamentals & scores

- "I want to understand the growth, profitability, reinvestment, and risk profile for MSFT on an annual basis."
- "Compare AAPL vs MSFT fundamentals scores on a TTM basis and show me the table."

### Time-series visuals

- "Plot Microsoft's profitability metrics over time on an annual basis."
- "Chart Operating_Margin for AAPL over time (annual)."
- "Compare NVDA, AMD, INTC on their risk metrics over time (annual)."

### Peers & price-to-X

- "Build a peer multiples view for AAPL vs MSFT, GOOGL, AMZN, and exclude AAPL from the peer stats."
- "From that peer set, what are AAPL's implied price bands (median and quartiles)?"
- "Show me a PE distribution chart for AAPL against MSFT, GOOGL, AMZN."

### EV / Market cap

- "How does MSFT's enterprise value compare to an implied EV from cash flows? Also draw the chart."
- "Translate that into implied equity value vs observed market cap for MSFT and plot it."

### DCF

- "Run a quick three-scenario DCF for NVDA (peers: GOOGL, META, AMZN, MSFT) and overlay today's price in a chart."

### Prices & history

- "What's NVDA's current 1-day average price and price as-of date?"
- "Give me 1/30/90/180-day average prices for AAPL and MSFT, and save a CSV."

### Health

- "Tabulate this health report and sort it:" (paste the dict)
- "Render this health report to Markdown."

If Claude replies conversationally instead of running a tool, add "use your value-investing-tools server" or say "plot it / return a table" to nudge tool usage.

For `price_from_peer_multiples`, either reference the previous `peer_multiples` result or provide tickers + target_ticker so the server can compute it on the fly.

## Troubleshooting

### "relative path can't be expressed as a file URI"
Make sure you're on the latest `server.py`. It returns absolute `file:///…` URIs and never re-parses them as relative paths.

### Server exits immediately / can't open server.py
In your Claude config, use Option A (absolute path) or Option B (cwd set). Passing only "server.py" without cwd may make Python look in Claude's app folder.

### module 'vit' has no attribute '_price_snapshots'
The vit wrapper re-exports `_price_snapshots`/`_price_snapshots_ext`. Ensure `vit/__init__.py` is the version that exposes those names, or set `VIT_LIB_BASENAME` to your library filename.

### OneDrive path typos
It's **OneDrive** (capital D). **OneDRive** will fail.

### Network/data freshness
Some tools use yfinance. Ensure you're online for price/quote fetches.

## Using the Python API (notebooks)

The MCP server is optional — you can always use the library directly:

```python
import vit

# fundamentals
vit.compute_fundamentals_actuals(["MSFT","AAPL"], basis="annual", as_df=True)

# peer multiples → implied price bands
pm = vit.peer_multiples(["AAPL","MSFT","GOOGL","AMZN"], target_ticker="AAPL", as_df=True)
vit.price_from_peer_multiples(pm, ticker="AAPL", as_df=True)

# DCF plot vs price
df = vit.dcf_three_scenarios("NVDA", peer_tickers=["GOOGL","META","AMZN","MSFT"], as_df=True)
p1d, _, _ = vit._price_snapshots("NVDA")
vit.plot_dcf_scenarios_vs_price(df, p1d)
```

If Jupyter starts from a different working directory, add the repo path to `sys.path` or install the package into your environment.
