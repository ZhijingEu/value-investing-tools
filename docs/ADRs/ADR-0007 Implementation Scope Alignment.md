# ADR-0007: Implementation Scope Alignment (Current Model vs Planned Model)

## Status
Accepted

## Date
2026-02-22

## Context
Repository docs and ADRs describe a planned 4x4 fundamentals redesign (ADR-0001) and Piotroski module (ADR-0002).  
The current production code in `ValueInvestingTools.py` still exposes the existing 12-metric scoring pipeline via:
- `compute_fundamentals_actuals(...)`
- `compute_fundamentals_scores(...)`

This mismatch can cause contributor confusion and incorrect assumptions about what is currently available in APIs and MCP tools.

## Decision
Adopt an explicit two-track documentation policy:
1. **Current implementation track**: README and contributor docs must describe the shipped 12-metric model and available APIs.
2. **Planned architecture track**: ADR-0001/0002/0004/0005/0006 remain valid design targets and stay clearly marked as proposed.

Until the 4x4 redesign lands in code:
- no documentation should imply Altman/Beneish/Piotroski are already part of live scoring outputs,
- roadmap phases are treated as targets, not completion claims.

## Consequences
- Improves contributor correctness and review quality.
- Reduces contract drift between docs, MCP tool behavior, and library outputs.
- Enables incremental implementation of planned ADRs without breaking user trust.
