# Release Notes 0.50.0 (2026-02-23)

## Summary
This release modularizes the core library into `vitlib/` while preserving the public API via the `ValueInvestingTools.py` facade. It also adds valuation diagnostics (confidence + assumptions snapshots), peer comparability diagnostics, and a DCF sensitivity grid, plus updated MCP contract tests and demo notebook examples.

## Highlights
- Modular core: `vitlib/utils`, `fundamentals`, `peers`, `valuation`, `health`, `plots`, `orchestrator`.
- Peer multiples: forward PE mode + `peer_quality_diagnostics`.
- Valuation outputs: `Valuation_Confidence` and `assumptions_snapshot_id`.
- DCF: `dcf_sensitivity_grid` for WACC x terminal growth sensitivity.
- MCP: expanded contract tests for peer multiples and valuation confidence.
- Demo notebook updated to showcase new outputs.

## Migration Notes
No breaking changes to the public API. Existing imports like:

```python
import ValueInvestingTools as vit
```

continue to work.

If you imported internal helpers directly from `ValueInvestingTools.py`, prefer the new module locations:
- Fundamentals: `vitlib.fundamentals`
- Peers: `vitlib.peers`
- Valuation: `vitlib.valuation`
- Health: `vitlib.health`
- Plots: `vitlib.plots`
- Orchestrator: `vitlib.orchestrator`
- Utilities: `vitlib.utils`

Test patches that mocked internal functions should patch the new module paths (e.g., `vitlib.valuation._calculate_wacc`).

## Validation
Recommended:
```bash
python -m py_compile server.py ValueInvestingTools.py vit/__init__.py vitlib/*.py
python -m unittest discover -s tests -p "test_*.py" -v
```
