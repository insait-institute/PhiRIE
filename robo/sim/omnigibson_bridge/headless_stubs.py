"""Shared workarounds for running OmniGibson/Isaac Sim 5.1.0 headless on this
cluster's driver (595.58.03, outside Isaac Sim 5.1's validated 580.65.06).

Root cause (see omnigibson_5_1_0.kit's comment for the fuller writeup): any
Kit extension that touches the UI/menu/viewport machinery
(omni.kit.menu.utils, omni.kit.widget.viewport's Hydra-engine init, the
viewport menubar's USD watcher) segfaults natively at stage-creation or
stage-open time on this driver, even with headless=True - "headless" only
suppresses the native window, it doesn't skip loading/initializing those
extensions. We deliberately excluded them all from omnigibson_5_1_0.kit's
[dependencies], which means the handful of OmniGibson code paths that
casually reach for one of them (semantic labels, debug colors, the viewport
USD watcher used during og.clear()) need a stand-in. None of these are used
for pose-based BDDL predicate checks, so every stub here is a safe no-op for
this validation's purposes - just enough surface for the call sites that
still reach for them to succeed instead of crashing.

Call apply() once, right after `import omnigibson`, before `og.launch()`.
"""
import sys
import types

import numpy as np


def apply():
    from omnigibson.macros import gm

    # No rendering needed for pose-based BDDL checks.
    gm.RENDER_VIEWER_CAMERA = False
    # Rigid-body COM tensor queries (get_coms(), hit during every object's
    # _post_load()) raised "Failed to get rigid body coms from backend" with
    # GPU dynamics off - the CPU PhysX pipeline in this Isaac Sim build
    # doesn't back that tensor API call for plain rigid bodies.
    gm.USE_GPU_DYNAMICS = True

    def _stub_random_colours(N, enable_random=True, num_channels=4, start=0):
        rng = np.random.default_rng(0)
        colours = rng.integers(0, 255, size=(N, num_channels), dtype=np.uint8)
        if enable_random:
            rng.shuffle(colours)
        return colours

    class _ModifyNamespace:
        @staticmethod
        def semantics(prim, label_dict, mode="replace"):
            pass  # no-op: semantic labels are unused for BDDL pose checks

    _replicator_pkg = types.ModuleType("omni.replicator")
    _replicator_core_stub = types.ModuleType("omni.replicator.core")
    _replicator_core_stub.__path__ = []  # mark as a package so .functional resolves
    _replicator_core_stub.random_colours = _stub_random_colours
    _replicator_functional_stub = types.ModuleType("omni.replicator.core.functional")
    _replicator_functional_stub.modify = _ModifyNamespace()
    _replicator_core_stub.functional = _replicator_functional_stub
    _replicator_pkg.core = _replicator_core_stub
    sys.modules["omni.replicator"] = _replicator_pkg
    sys.modules["omni.replicator.core"] = _replicator_core_stub
    sys.modules["omni.replicator.core.functional"] = _replicator_functional_stub

    # og.clear() (called internally by convert_urdf_to_usd, and available for
    # us to call directly) stops/starts a viewport menubar USD watcher around
    # teardown. We never load that extension, so there's genuinely no
    # watcher to stop/start - no-op start()/stop() is behaviorally correct,
    # not just a hack.
    class _UsdWatcher:
        @staticmethod
        def start():
            pass

        @staticmethod
        def stop():
            pass

    _menubar_utils_stub = types.ModuleType("omni.kit.viewport.menubar.core.utils")
    _menubar_utils_stub.usd_watch = _UsdWatcher()
    _menubar_core_stub = types.ModuleType("omni.kit.viewport.menubar.core")
    _menubar_core_stub.__path__ = []
    _menubar_core_stub.utils = _menubar_utils_stub
    _menubar_stub = types.ModuleType("omni.kit.viewport.menubar")
    _menubar_stub.__path__ = []
    _menubar_stub.core = _menubar_core_stub
    sys.modules["omni.kit.viewport.menubar"] = _menubar_stub
    sys.modules["omni.kit.viewport.menubar.core"] = _menubar_core_stub
    sys.modules["omni.kit.viewport.menubar.core.utils"] = _menubar_utils_stub

    # We don't load any real omni.kit.viewport.* extension, so the parent
    # package may not exist in sys.modules either - stub it too (defensively;
    # if a real one somehow got loaded, don't clobber it).
    if "omni.kit.viewport" not in sys.modules:
        _viewport_stub = types.ModuleType("omni.kit.viewport")
        _viewport_stub.__path__ = []
        _viewport_stub.menubar = _menubar_stub
        sys.modules["omni.kit.viewport"] = _viewport_stub
    else:
        sys.modules["omni.kit.viewport"].menubar = _menubar_stub
