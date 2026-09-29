"""GPU detection and environment compatibility.

``detect_gpu()`` never raises: on a node without a GPU it returns ``GpuInfo(present=False)``.
``env_compatible(config, env_key, gpu)`` answers whether a stage interpreter's compiled
kernels can run on the local GPU (compute capability listed in
``config.env_arch_support``). The Slurm ``gpu_type_for(env_key)`` picks the first remote
GPU type whose compute capability the env supports (preferring ``default_remote_gpu``).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict

from physicalview.config import StudioConfig


@dataclass(frozen=True)
class GpuInfo:
    present: bool
    name: str = ""
    compute_cap: str = ""          # e.g. "8.6"
    memory_mib: int = 0
    driver: str = ""
    index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def detect_gpu(index: int = 0) -> GpuInfo:
    """Query nvidia-smi (no torch import, so this works in any interpreter)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return GpuInfo(present=False)
    try:
        out = subprocess.run(
            [exe, f"--id={index}", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (subprocess.SubprocessError, OSError):
        return GpuInfo(present=False)
    if out.returncode != 0 or not out.stdout.strip():
        return GpuInfo(present=False)
    name, cc, mem, drv = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    try:
        mem_i = int(float(mem))
    except ValueError:
        mem_i = 0
    return GpuInfo(present=True, name=name, compute_cap=cc, memory_mib=mem_i,
                   driver=drv, index=index)


def env_compatible(config: StudioConfig, env_key: str, gpu: GpuInfo) -> bool:
    """True when the env's compiled kernels cover the local GPU's compute capability.

    An env with no entry in ``env_arch_support`` is treated as CPU-only/compatible.
    """
    archs = config.env_arch_support.get(env_key)
    if archs is None:
        return True
    if not gpu.present:
        return False
    return gpu.compute_cap in archs


def gpu_type_for(config: StudioConfig, env_key: str) -> str | None:
    """Remote Slurm GPU type whose compute capability the env supports, or None."""
    archs = config.env_arch_support.get(env_key)
    if archs is None:
        return config.default_remote_gpu
    order = [config.default_remote_gpu] + [k for k in config.gpu_targets if k != config.default_remote_gpu]
    for key in order:
        target = config.gpu_targets.get(key)
        if target is not None and target.compute_cap in archs:
            return key
    return None


def compatibility_matrix(config: StudioConfig, gpu: GpuInfo) -> dict[str, dict]:
    """Per-env dict: {local: bool, remote_gpu: str|None} — shown in the Jobs tab."""
    return {key: {"local": env_compatible(config, key, gpu),
                  "remote_gpu": gpu_type_for(config, key)}
            for key in config.interpreters}


def main(argv=None) -> int:  # pragma: no cover - convenience CLI
    import argparse
    from physicalview.config import load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    gpu = detect_gpu()
    print(json.dumps({"gpu": gpu.to_dict(), "matrix": compatibility_matrix(cfg, gpu)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
