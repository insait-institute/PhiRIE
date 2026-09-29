"""Post-install sanity check: can Isaac Sim actually boot headless on this
machine, and does OmniGibson's own object-state machinery work at all (that's
the piece the whole validation depends on, per the Inside/OnTop == pure
kinematic-state finding from BEHAVIOR-1K's bddl_utils.py).

Run with: OMNIGIBSON_HEADLESS=1 python smoke_test.py
"""
import os

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import sys

print("[smoke] python", sys.version)

import torch  # noqa: E402

print("[smoke] torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[smoke] gpu:", torch.cuda.get_device_name(0))

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

print("[smoke] omnigibson", og.__version__ if hasattr(og, "__version__") else "?")
print("[smoke] gm.HEADLESS =", gm.HEADLESS)

# We only need physics + pose-based BDDL predicates (Inside/OnTop/...), no
# rendering - the default viewer camera setup pulls in a viewport/Hydra
# render pipeline that segfaults on NVIDIA drivers outside Isaac Sim's
# validated range (see headless_stubs.py). Skip it entirely rather than
# chase every
# rendering-extension dependency.
gm.RENDER_VIEWER_CAMERA = False

# omni.replicator.core (needed for ground-plane semantic labeling, random
# debug colors on primitives, etc.) pulls in omni.kit.widget.viewport, whose
# Hydra-engine init segfaults on such drivers at stage-creation
# time - same root cause as the earlier menu segfault, just a different
# extension tripping it. None of this is needed for BDDL pose checks, so
# stub the whole module rather than chasing every call site individually.
import sys  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402


def _stub_random_colours(N, enable_random=True, num_channels=4, start=0):
    rng = np.random.default_rng(0)
    colours = rng.integers(0, 255, size=(N, num_channels), dtype=np.uint8)
    if enable_random:
        rng.shuffle(colours)
    return colours


_replicator_pkg = types.ModuleType("omni.replicator")
_replicator_core_stub = types.ModuleType("omni.replicator.core")
_replicator_core_stub.random_colours = _stub_random_colours
_replicator_pkg.core = _replicator_core_stub
sys.modules["omni.replicator"] = _replicator_pkg
sys.modules["omni.replicator.core"] = _replicator_core_stub

import omnigibson.simulator as _og_sim_mod  # noqa: E402

_og_sim_mod.add_semantic_label = lambda *args, **kwargs: None

og.launch()
print("[smoke] og.launch() OK - Isaac Sim booted headless")

from omnigibson.object_states import OnTop, Inside  # noqa: E402,F401  (import check only)
from omnigibson.scenes.scene_base import Scene  # noqa: E402

print("[smoke] OnTop/Inside importable")

# PrimitiveObject is NOT used by the real harness (import_and_run.py uses
# DatasetObject exclusively) - it needs omni.kit.material/MDL commands we
# deliberately stripped out, so it's not a representative test here.
scene = Scene(use_floor_plane=True, use_skybox=False, include_robots=False)
og.sim.import_scene(scene)
og.sim.play()
og.sim.step()
print("[smoke] Scene imported + sim.play() + step() OK")

print("[smoke] ALL OK")
