"""Portable interpreter selection. Each GPU dependency stack stays isolated."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNTIMES = {
    "control": ("PHIROOM_CONTROL_PY", None),
    "main": ("SIMANY_PY", ".venv/bin/python"),
    "sam3": ("SIMANY_SAM3_PY", ".envs/sam3/bin/python"),
    "gsplat": ("SIMANY_GSPLAT_PY", ".envs/gsplat/bin/python"),
    "sam3d": ("SIMANY_SAM3D_PY", ".envs/sam3d-objects/bin/python"),
    "trellis2": ("SIMANY_TRELLIS2_PY", ".envs/trellis2/bin/python"),
    "qwen": ("QWEN_PY", ".envs/qwen/bin/python"),
    "h5": ("SIMANY_H5_PY", ".envs/h5/bin/python"),
}


def source_root(root: str | None = None) -> Path:
    candidates = [
        root or os.environ.get("PHIROOM_ROOT"),
        str(Path.cwd()),
        str(Path(__file__).absolute().parents[2]),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().absolute()
            if (path / "run/phiroom.sh").is_file():
                return path
            if candidate == candidates[0]:
                raise ValueError(f"PhiRoom source root missing at {path}")
    raise ValueError(
        "Run inside the full source checkout or specify --root /path/to/PhiRoom"
    )


def load_config(path: str | None) -> dict:
    cfg = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(cfg, dict) or set(cfg) - {"runtimes", "env"}:
        raise ValueError("runtime config supports only 'runtimes' and 'env' objects")
    for field in ("runtimes", "env"):
        values = cfg.setdefault(field, {})
        if not isinstance(values, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or "\0" in k + v
            for k, v in values.items()
        ):
            raise ValueError(f"{field} must map strings to strings")
    if set(cfg["runtimes"]) - RUNTIMES.keys():
        raise ValueError("unknown runtime alias in configuration")
    # Only pipeline context goes in receipts; credentials belong in the process env.
    if any(
        not (k.startswith("SIMANY_") or k in {"CUDA_VISIBLE_DEVICES", "MUJOCO_GL"})
        or any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
        for k in cfg["env"]
    ):
        raise ValueError(
            "env accepts non-secret SIMANY_* context, CUDA_VISIBLE_DEVICES, MUJOCO_GL"
        )
    return cfg


def interpreter(runtime: str, root: Path, cfg: dict) -> str:
    variable, relative = RUNTIMES[runtime]
    value = cfg.get("runtimes", {}).get(runtime) or os.environ.get(variable)
    if not value:
        value = str(root / relative) if relative else sys.executable
    path = Path(value).expanduser()
    # Do not resolve symlinks: doing so silently escapes a virtual environment.
    return str(path if path.is_absolute() else root / path)
