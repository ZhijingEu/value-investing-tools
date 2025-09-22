"""
vit: friendly import wrapper for ValueInvestingTools

- Lets users simply `import vit` while the actual file can be versioned/renamed.
- You can override the filename via env: VIT_LIB_BASENAME=ValueInvestingTools_Rev49.1.py
"""

from __future__ import annotations
from importlib.machinery import SourceFileLoader
from pathlib import Path
import os as _os

_here = Path(__file__).resolve().parent
_root = _here.parent

# 1) Honor env override first
_env_name = _os.environ.get("VIT_LIB_BASENAME", "").strip()

# 2) Fallback candidates if env not set
_candidates = [
    _env_name or "",                     # maybe ""
    "ValueInvestingTools.py",            # suffixless
    "ValueInvestingTools_Rev49.1.py",    # versioned
]

_lib_path = None
for name in _candidates:
    if not name:
        continue
    p = (_root / name).resolve()
    if p.exists():
        _lib_path = p
        break

if _lib_path is None:
    raise FileNotFoundError(
        "vit: could not find the ValueInvestingTools module. Tried:\n  - "
        + "\n  - ".join(str((_root / n).resolve()) for n in _candidates if n)
        + "\nSet VIT_LIB_BASENAME to the correct filename if you renamed it."
    )

_mod = SourceFileLoader("valueinvestingtools", str(_lib_path)).load_module()

# Re-export public API
_public = {k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}

# Explicitly expose the private helpers the MCP server uses
for _name in ("_price_snapshots", "_price_snapshots_ext"):
    if hasattr(_mod, _name):
        _public[_name] = getattr(_mod, _name)

globals().update(_public)

# Best-effort version tag
__version__ = getattr(_mod, "__version__", f"wrapped@{_lib_path.name}")