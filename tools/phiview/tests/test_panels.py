"""Panel modules import without viser and their pure helpers behave (viser is only
imported inside build())."""
from __future__ import annotations

from pathlib import Path


def test_panel_pure_helpers_importable_without_viser():
    from physicalview.panels import generate_panel, inpaint_panel, jobs_panel
    assert "no jobs yet" in jobs_panel.jobs_table([])
    assert generate_panel.object_id_from_name("obj_07") == 7
    assert "select an object" in generate_panel.proposal_cards(None)
    rec = {"stamp": "20260904T120000Z", "selection": {"objects": [3], "regions": None}, "backend": "lama"}
    assert inpaint_panel.version_label(rec) == "20260904T120000Z · obj_03 · lama"
    assert inpaint_panel.read_versions(Path("/nonexistent/out")) == []


def test_all_panels_expose_build():
    import importlib
    for name in ("scene_panel", "generate_panel", "inpaint_panel", "robot_panel", "jobs_panel"):
        mod = importlib.import_module(f"physicalview.panels.{name}")
        assert callable(getattr(mod, "build"))
