# North Star: VIT Product Positioning

## Positioning Statement
ValueInvestingTools (VIT) is an **LLM-native valuation engine** for equity research workflows.

Its job is to convert raw financial data into **auditable** valuation outputs:
- fundamentals diagnostics and scoring
- peer multiple comparisons and implied price bands
- EV -> Equity bridge logic
- DCF scenario analysis with explicit assumptions and health notes

VIT is designed to run both:
- as a Python library for notebooks/scripts
- as a local STDIO MCP server that an LLM can compose with other finance/data MCPs

## What VIT Should Be Best At
- Transparent valuation calculations (not black-box predictions)
- Assumption stamping (`as_of`, defaults used, overrides)
- Data-quality/health diagnostics attached to outputs
- Reproducible tool contracts for agent workflows (JSON/table/artifact outputs)
- Practical analyst workflows: fundamentals -> peers -> valuation triangulation

## Explicit Non-Goals (for now)
- Portfolio optimization / sizing / risk budgeting
- Backtesting or execution platform
- Full market-data terminal breadth (news, options, macro, alt data, broker APIs)
- Replacing institutional research systems end-to-end

These may be handled by other tools/MCP servers that an LLM orchestrates alongside VIT.

## Scope Guardrails
Prefer features that strengthen VIT's valuation-engine niche:
- Better valuation methods and diagnostics
- Better data-source adapters feeding the same valuation schema
- Better output contracts and test coverage
- Better reproducibility (assumptions, timestamps, health flags)

Deprioritize features that mainly expand breadth without improving valuation quality:
- large catalogs of raw data endpoints
- portfolio/trading workflows
- unrelated analytics domains

## Decision Rubric (use before adding features)
Approve a feature when most answers are "yes":
1. Does it improve valuation quality, auditability, or reproducibility?
2. Does it support MCP/tool composition for equity research agents?
3. Can it fit the current workflow (fundamentals, peers, EV/DCF, health) without turning VIT into a platform?
4. Can it be tested deterministically with a stable output contract?

If the main value is portfolio construction, trading, or broad data aggregation, it likely belongs outside VIT.

## Near-Term Direction
- Sprint 1: North Star + provider/test foundations
- Sprint 2: valuation-quality upgrades (forward comps, sensitivity, confidence diagnostics)
- Sprint 3: hardening, modularization, release discipline

Use `docs/Roadmap.md` for implementation sequencing and ADRs for major design decisions.
