# Documentation Index

This folder contains public documentation intended for users and contributors.

## Start Here
- `../README.md` - usage, installation, and function reference.
- `../Methodology.md` - theoretical foundations, assumptions, and override rationale.

## Product & Planning
- `NORTH_STAR.md` - positioning, scope guardrails, decision rubric.
- `Roadmap.md` - phased delivery plan and status.

## Architecture Decisions
- `ADRs/` - decision records for major design choices.

## MCP Usage
- `mcp/README.md` - MCP configuration templates and guidance.

## Release
- `RELEASE_PROCESS.md` - versioning and release workflow.
- `RELEASE_NOTES_0.50.0.md` - release notes for current version.

## Internal Working Docs
Internal working notes live under `docs/internal/` and are not tracked in git.

## Contributor Expectations (Summary)
- New modules should include function signatures, a minimal input/output example, and ADR references when relevant.
- Keep plots deterministic where possible (stable ordering, no randomness).

## MCP Tool Docstring Checklist (Stub)
- Purpose: one sentence on what the tool returns.
- Key inputs: list of the most important parameters and defaults.
- Overrides: mention override dicts if supported (and the key names).
- Interpretation: short guidance on reading outputs and warnings.
