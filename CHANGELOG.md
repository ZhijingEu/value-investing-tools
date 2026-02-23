# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Placeholder for upcoming Sprint 3/4 changes.

## [0.50.0] - 2026-02-23

### Added
- Modular core library under `vitlib/` (utils, fundamentals, peers, valuation, health, plots, orchestrator) while preserving the `ValueInvestingTools.py` facade.
- Forward PE mode for peer multiples, peer comparability diagnostics, DCF sensitivity grid, valuation confidence metadata, and assumptions snapshot IDs.
- MCP contract tests for peer multiples and valuation confidence outputs.
- Demo notebook sections for forward PE, diagnostics, valuation confidence, and sensitivity grid.

### Changed
- `ValueInvestingTools.py` now re-exports the modular core while keeping the public API stable.
