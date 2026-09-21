"""SimFactory backend registry: every pipeline BLOCK is a folder, every
BASELINE in it registers under a name, and a YAML config selects one
(LLaMA-Factory's model/method registry pattern, applied to scene blocks).

Blocks: reconstruction, segmentation, inpainting, generation, simulation,
evaluation. A backend is any callable `run(ctx) -> None` (usually a thin
adapter that shells out to the battle-tested agents/models/robo modules with
the right env + interpreter); `available(ctx)` may veto (missing weights,
wrong scene source) with a reason string instead of failing mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BLOCKS = ("reconstruction", "segmentation", "inpainting", "generation",
          "simulation", "evaluation")

_REGISTRY: dict[str, dict[str, "Backend"]] = {b: {} for b in BLOCKS}


@dataclass
class Backend:
    block: str
    name: str
    run: Callable
    help: str = ""
    available: Callable | None = None   # ctx -> None | reason string


def register(block: str, name: str, help: str = "",
             available: Callable | None = None):
    if block not in _REGISTRY:
        raise KeyError(f"unknown block {block!r}; blocks: {BLOCKS}")

    def deco(fn):
        _REGISTRY[block][name] = Backend(block, name, fn, help, available)
        return fn
    return deco


def get(block: str, name: str) -> Backend:
    try:
        return _REGISTRY[block][name]
    except KeyError:
        opts = ", ".join(sorted(_REGISTRY.get(block, {})))
        raise SystemExit(f"[simfactory] no backend {name!r} for block "
                         f"{block!r}; available: {opts}") from None


def catalog() -> dict[str, dict[str, "Backend"]]:
    return _REGISTRY


@dataclass
class Context:
    """Everything a backend needs, resolved once from the config."""
    root: Path                       # repo root
    scene: str                       # scene id / name
    source: str                      # video | scannetpp | behavior | droid
    config: dict                     # full parsed YAML
    out: Path                        # $SIMANY_OUT
    scene_dir: Path                  # dataset-root/data/<scene>
    splats_root: Path
    scannetpp_root: Path
    options: dict = field(default_factory=dict)
    dry_run: bool = False

    def opt(self, key, default=None):
        return self.options.get(key, default)
