"""PhysicalView: interactive GPU-backed viser UI for the full SimAny real-to-sim pipeline.

Importing this package puts the configured SimAny checkout (agents/, robo/, models/) on
``sys.path`` so the viewer works without a hand-set PYTHONPATH. Entry point:
``python -m physicalview.app``; docs in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

__version__ = "0.2.0"


def _bootstrap_simany_root() -> None:
    try:
        import yaml
        from physicalview.config import DEFAULT_CONFIG, resolve_simany_root
        raw = yaml.safe_load(_Path(DEFAULT_CONFIG).read_text()) or {} if _Path(DEFAULT_CONFIG).exists() else {}
        root = resolve_simany_root(raw)
    except Exception:  # noqa: BLE001 - never break import
        root = _Path(_os.environ.get("SIMANY_ROOT", str(_Path(__file__).resolve().parents[1] / "backends/simany")))
    r = str(root)
    if root.is_dir() and r not in _sys.path:
        _sys.path.append(r)


_bootstrap_simany_root()
