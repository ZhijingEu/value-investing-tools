# ADR-0006: MCP Transport – stdio MCP for Claude Desktop, HTTP/SSE for ChatGPT Developer Mode

## Status
Proposed (Draft)

## Context
The Value Investing Tools project exposes analytical and valuation functions to local AI agents  
(e.g., Claude Desktop, ChatGPT Developer Mode) through the **Model Context Protocol (MCP)**.  
At present, all tools operate via a **stdio (standard I/O) transport**, which is compatible with Claude Desktop’s local tool execution.

However, ChatGPT Developer Mode and other LLM environments require an **HTTP or SSE (Server-Sent Events)** transport layer.  
To future-proof the project, both modes need to be supported without breaking existing local workflows.

---

## Decision
Adopt a **dual-mode transport design** for MCP integration:

| Mode | Purpose | Environment | Protocol |
|------|----------|--------------|-----------|
| **stdio** | Default local execution | Claude Desktop / local MCP servers | JSON-RPC via stdin/stdout |
| **HTTP/SSE** | Remote or hosted access | ChatGPT Developer Mode / web tunneling (e.g., ngrok) | HTTP POST + SSE stream responses |

---

## Implementation Plan

1. **Maintain existing local server (`server.py`)**
   - Keep stdio transport as the default.  
   - Expose new fundamentals, Piotroski, and valuation functions via MCP tool registration:
     ```python
     mcp.register_tool("compute_fundamentals_score", compute_fundamentals_score)
     mcp.register_tool("piotroski_fscore", piotroski_fscore)
     mcp.register_tool("compare_to_market_ev", compare_to_market_ev)
     ```
   - Continue to run using:
     ```bash
     python server.py
     ```

2. **Create an HTTP/SSE companion (`server_http.py`)**
   - Implement a lightweight FastAPI or Flask app exposing identical endpoints.  
   - Each endpoint mirrors a corresponding MCP tool, returning streamed results via SSE.  
   - Example route:
     ```python
     @app.post("/compute_fundamentals_score")
     async def compute_fundamentals_endpoint(req: FundamentalsRequest):
         result = compute_fundamentals_score(**req.dict())
         return StreamingResponse(yield_json(result), media_type="text/event-stream")
     ```

3. **Connection Options**
   - Local users: continue running stdio MCP (no change).  
   - ChatGPT Developer Mode users:  
     - Run `server_http.py` locally.  
     - Optionally expose via tunneling (e.g., ngrok, localtunnel) to obtain a public URL.  
     - Register endpoint under Developer Mode “Tools” with corresponding schema.

4. **Shared Tool Registry**
   - Maintain a single tool manifest (e.g., `tools_manifest.json`) listing available endpoints and transports.  
   - Example:
     ```json
     {
       "compute_fundamentals_score": {"stdio": true, "http": "/compute_fundamentals_score"},
       "dcf_three_scenarios": {"stdio": true, "http": "/dcf_three_scenarios"}
     }
     ```

5. **Response Contract**
   - Both transports must return identical JSON payloads, including:
     - `method` and `profile` metadata
     - `pillar_scores`, `overall_scores`, `reference_scores`
     - `data_health` section
     - Optional streaming output (for long-running valuations)

6. **Security & Isolation**
   - HTTP version runs locally by default (no public exposure).  
   - Explicit tunneling or reverse proxy required for remote access.  
   - No persistent data storage or authentication in v1 (for simplicity).

---

## Alternatives Considered
1. **Keep only stdio transport.**  
   *Rejected:* limits integration with ChatGPT Developer Mode and other web-based LLM tools.  
2. **Adopt WebSocket transport instead of SSE.**  
   *Rejected:* unnecessary complexity for unidirectional event streams.  
3. **Use cloud-hosted MCP.**  
   *Rejected:* violates local-first design principle and offline usability.

---

## Consequences
- **Short-term:** seamless local integration with Claude Desktop remains functional.  
- **Medium-term:** ChatGPT Developer Mode support enabled via optional HTTP/SSE server.  
- **Long-term:** architecture flexible enough to support multi-agent orchestration (both local and hosted).  
- Provides consistent API contracts across transports, ensuring interoperability.

---

## References
- Anthropic, *Model Context Protocol (MCP) Specification*, 2024.  
- OpenAI, *ChatGPT Developer Mode Tool Interface Documentation*, 2025.  
- FastAPI Documentation – [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)  
