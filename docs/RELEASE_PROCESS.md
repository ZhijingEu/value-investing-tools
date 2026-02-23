# Release Process (Semantic Versioning)

This project follows Semantic Versioning: `MAJOR.MINOR.PATCH`.

## Versioning Rules
- **MAJOR**: breaking API changes (function names, signatures, payload schema changes).
- **MINOR**: backward-compatible features (new tools, new output fields, new modules).
- **PATCH**: backward-compatible fixes (bug fixes, docs corrections, test-only changes).

## Required Updates Per Release
1. Update `ValueInvestingTools.py`:
   - Set `__version__ = "X.Y.Z"`.
2. Update `CHANGELOG.md`:
   - Move items from `[Unreleased]` into a new `X.Y.Z - YYYY-MM-DD` section.
3. Run tests:
   - `python -m unittest discover -s tests -p "test_*.py" -v`
4. Tag the release:
   - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`

## Release Checklist
- [ ] Version updated in `ValueInvestingTools.py`.
- [ ] `CHANGELOG.md` updated for the release.
- [ ] Tests pass locally.
- [ ] Tag created.
- [ ] Release notes drafted (summary of Added/Changed/Fixed).
