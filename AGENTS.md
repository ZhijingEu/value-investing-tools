# Repository Guidelines

## Project Structure & Module Organization
- `ValueInvestingTools.py`: main analytics library (fundamentals, peer multiples, EV/DCF, plotting helpers).
- `server.py`: MCP stdio server exposing library functions as tools.
- `vit/__init__.py`: import wrapper that re-exports the active `ValueInvestingTools` module.
- `providers/`: provider adapter scaffolding (`base.py`, `yahoo.py`) for future data-source abstraction.
- `docs/`: roadmap, developer notes, ADRs (`docs/ADRs/*.md`), and `docs/NORTH_STAR.md` (product positioning / scope guardrails).
- `output/`: generated charts/CSVs (gitignored artifacts).
- `tests/`: `unittest`-based regression and contract checks.

## Build, Test, and Development Commands
- Create isolated env (preferred for all testing/improvements):
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Syntax smoke check: `python -m py_compile server.py ValueInvestingTools.py vit\__init__.py`
- Run MCP server locally: `python server.py`
- Quick import check: `python -c "import vit; print(vit.__version__)"`

## Coding Style & Naming Conventions
- Target Python 3.10+ style, 4-space indentation, and PEP 8 defaults.
- Use `snake_case` for functions/variables, `UPPER_CASE` for constants, and clear metric names.
- Keep public function signatures stable; document behavior changes in `README.md` and relevant ADR/docs files.
- Prefer small, focused helpers over adding more monolithic logic blocks.

## Testing Guidelines
- Current baseline uses `unittest` (`python -m unittest discover -s tests -p "test_*.py" -v`).
- `pytest` is acceptable for future additions, but keep CI commands and docs aligned.
- New tests should live under `tests/` and follow `test_*.py` naming.
- Prioritize deterministic unit tests around scoring logic, schema contracts, and tool I/O shaping.
- If network data is required, isolate with fixtures/mocks where possible and include smoke tests separately.

## Commit & Pull Request Guidelines
- Keep commit subjects short and descriptive (history style: concise update/refinement messages).
- Preferred format: imperative present tense, e.g., `Refine scoring edge-case handling`.
- PRs should include: what changed, why, affected files/functions, and sample output paths when plots/CSVs change.

## Security & Configuration Tips
- Always run testing/improvement work in an isolated venv (`.venv` or `vit-env`), never system Python.
- Never commit secrets. `.gitignore` already excludes `*.env`, `*secrets*.json`, local Claude config files, env folders, and generated outputs.
- If new sensitive config files are introduced, add ignore rules immediately.
