"""Tests for the GT-free fallback added to agents/discover/s0_select_frame.py
and agents/assets/s5_align.py.

Context (docs/ICRA_RESEARCH_CONTRACT.md's post-freeze correction #9,
baselines/simfoundry_repro.py's module docstring "IMPORTANT SCOPE NOTE"):
both stages used to hard-call agents.core.common.load_gt_instances(), which
only resolves ScanNet++'s scans/segments.json + scans/segments_anno.json.
On BEHAVIOR/oracle `recon_scenes` scenes that either crashes outright, or --
worse -- would leak hidden oracle GT into the baseline's own construction if
a same-shaped file ever existed there (exactly what oracle/gt_export.py's
lockdown test guards against). This suite checks the new GT-free fallback:
  1. triggers (missing GT files, or an explicit --no-gt/SIMANY_NO_GT=1
     override) instead of crashing, and is deterministic;
  2. never touches load_gt_instances() or any `gt/`-shaped /
     *segments*/*anno*-shaped path, even when such files physically exist
     on disk right next to the (redirected-away) real ones;
  3. leaves the ORIGINAL GT-driven behavior byte-for-byte unchanged when GT
     instance files ARE available (regression coverage).

Fixtures build a tiny synthetic scene (an icosphere standing in for a single
"mug" GT instance, exactly like ScanNet++'s real convention of tagging
sub-regions of one scanned mesh via segments.json/segments_anno.json) plus
two colmap cameras, entirely under tmp_path, and monkeypatch
agents.core.common's module-level path constants -- the same
monkeypatch.setattr(module, "CONST", ...) pattern already used in
tests/test_contract_validation.py.

NOTE on agents.core.common.load_intrinsics()/load_colmap_w2c(): their path
arguments default to the module-level TRANSFORMS_JSON/COLMAP_IMAGES_TXT
constants, but those defaults are bound at function-definition time (a
plain `def f(x=SOME_MODULE_GLOBAL)`), so monkeypatching the module
attribute alone does NOT redirect a later no-argument call
(`C.load_intrinsics()`) the way it does for functions that reference a
global directly in their body (e.g. load_gt_instances(), which correctly
picks up a monkeypatched SEGMENTS_JSON). `_patch_common()` below works
around this the same way any caller would: by replacing the C.load_intrinsics
/C.load_colmap_w2c *callables* themselves with a thin wrapper bound to the
fixture paths, restored automatically by monkeypatch after each test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import trimesh
from PIL import Image

from agents.assets import s5_align as s5
from agents.core import common as C
from agents.discover import s0_select_frame as s0

IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


# ============================================================== fixture I/O

def _write_intrinsics(scene_dir: Path, w=640, h=480, f=500.0) -> Path:
    d = scene_dir / "dslr" / "nerfstudio"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "transforms_undistorted.json"
    p.write_text(json.dumps(
        {"fl_x": f, "fl_y": f, "cx": w / 2, "cy": h / 2, "w": w, "h": h}))
    return p


def _write_images_txt(scene_dir: Path, cams: dict) -> Path:
    """cams: {name: (qw, qx, qy, qz, tx, ty, tz)}"""
    d = scene_dir / "dslr" / "colmap"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (name, (qw, qx, qy, qz, tx, ty, tz)) in enumerate(cams.items(), 1):
        lines.append(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}")
        lines.append("0.0 0.0 -1")  # placeholder POINTS2D line (must be non-blank)
    p = d / "images.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def _write_images(scene_dir: Path, names, textured=None, size=(640, 480)):
    """`textured` frames get ORB-rich pseudo-random content; everything
    else is flat (zero ORB features) -- see feature_density() sanity check."""
    textured = textured or set()
    d = scene_dir / "dslr" / "resized_undistorted_images"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in names:
        if name in textured:
            arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        else:
            arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
        Image.fromarray(arr).save(d / name)
    return d


def _write_mug_mesh(scene_dir: Path, center=(0.0, 0.0, 1.0), radius=0.05):
    """A small icosphere standing in for a 'mug' GT instance -- its own
    vertices/faces are BOTH the raycast-able scene mesh and the tagged GT
    instance, matching ScanNet++'s real segments/segments_anno convention
    of tagging sub-regions of one scanned mesh (not a separate asset)."""
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    mesh.apply_translation(center)
    scans = scene_dir / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    mesh_path = scans / "mesh_aligned_0.05.ply"
    mesh.export(str(mesh_path))
    n = len(mesh.vertices)
    (scans / "segments.json").write_text(
        json.dumps({"segIndices": list(range(n))}))
    (scans / "segments_anno.json").write_text(json.dumps({"segGroups": [
        {"objectId": 1, "label": "mug", "segments": list(range(n))}]}))
    return mesh_path


def _write_object_fixture(out_dir: Path, gt_object_id=None):
    """objects/obj_00: a small box "generated asset" + a small-sphere
    partial "observation" cloud near world (0,0,1) -- enough for
    align_object() to run to completion and produce real numbers,
    independent of whatever s0/s3 upstream would have produced."""
    odir = out_dir / "objects" / "obj_00"
    odir.mkdir(parents=True, exist_ok=True)
    asset = trimesh.creation.box(extents=(0.08, 0.08, 0.10))
    asset.export(str(odir / "trellis_mesh.ply"))
    pts, _ = trimesh.sample.sample_surface(
        trimesh.creation.icosphere(subdivisions=1, radius=0.05), 500)
    pts = np.asarray(pts) + np.array([0.0, 0.0, 1.0])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(odir / "points.ply"), pc)
    meta = {"index": 0, "label": "mug", "extent": [0.1, 0.1, 0.1],
            "gt_object_id": gt_object_id}
    (out_dir / "objects" / "objects.json").write_text(json.dumps([meta]))
    return odir


def _patch_common(monkeypatch, scene_dir: Path, out_dir: Path, with_gt: bool):
    """Redirect agents.core.common's scene-dependent constants at fixture
    paths, working around load_intrinsics/load_colmap_w2c's def-time-bound
    defaults (see module docstring) by replacing the callables themselves."""
    transforms_json = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    images_txt = scene_dir / "dslr" / "colmap" / "images.txt"
    orig_load_intrinsics = C.load_intrinsics
    orig_load_colmap_w2c = C.load_colmap_w2c
    monkeypatch.setattr(
        C, "load_intrinsics",
        lambda tj=transforms_json: orig_load_intrinsics(tj))
    monkeypatch.setattr(
        C, "load_colmap_w2c",
        lambda it=images_txt: orig_load_colmap_w2c(it))
    monkeypatch.setattr(C, "IMAGES_DIR",
                         scene_dir / "dslr" / "resized_undistorted_images")
    monkeypatch.setattr(C, "OUT", out_dir)

    scans = scene_dir / "scans"
    if with_gt:
        mesh_path = scans / "mesh_aligned_0.05.ply"
        monkeypatch.setattr(C, "SEGMENTS_JSON", scans / "segments.json")
        monkeypatch.setattr(C, "SEGMENTS_ANNO_JSON", scans / "segments_anno.json")
        monkeypatch.setattr(C, "MESH_PLY", mesh_path)
        monkeypatch.setattr(C, "PIPELINE_MESH_PLY", mesh_path)
    else:
        # The recon_scenes/BEHAVIOR layout never has a scans/ dir at all --
        # point at a sibling dir guaranteed to never exist, deliberately NOT
        # "scans/" (some tests plant decoy GT-shaped files at the
        # conventional scans/ and gt/ locations to prove the fallback
        # doesn't wander into them via anything other than these constants).
        no_gt = scene_dir / "_no_gt_here"
        monkeypatch.setattr(C, "SEGMENTS_JSON", no_gt / "segments.json")
        monkeypatch.setattr(C, "SEGMENTS_ANNO_JSON", no_gt / "segments_anno.json")
        monkeypatch.setattr(C, "MESH_PLY", no_gt / "mesh_aligned_0.05.ply")
        monkeypatch.setattr(C, "PIPELINE_MESH_PLY", no_gt / "mesh_aligned_0.05.ply")


@pytest.fixture(autouse=True)
def _reset_s5_mesh_cache():
    """s5_align.scene_mesh_arrays() caches by presence only, not by which
    mesh path was used -- fine in real usage (one scene per process) but a
    cross-test contamination hazard here, where several tests in this file
    point PIPELINE_MESH_PLY at different fixture meshes within one process."""
    s5._MESH_CACHE.clear()
    yield
    s5._MESH_CACHE.clear()


# ============================================================ s0: GT-free trigger

def test_gt_free_fallback_triggers_and_is_deterministic(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_intrinsics(scene_dir)
    names = ["flat_a.jpg", "flat_b.jpg", "rich.jpg"]
    _write_images_txt(scene_dir, {n: IDENTITY_Q + (0.0, 0.0, float(i))
                                   for i, n in enumerate(names)})
    _write_images(scene_dir, names, textured={"rich.jpg"})
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=False)

    assert not s0.gt_instances_available()

    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()
    best1 = s0.select_frame_gt_free(K, W, H, w2c_all)
    best2 = s0.select_frame_gt_free(K, W, H, w2c_all)

    assert best1["mode"] == "gt_free_heuristic"
    assert best1["frame"] == "rich.jpg"       # most ORB features wins
    assert best1 == best2                     # same inputs -> same choice, twice

    # end-to-end through main(): no GT files present, no crash
    s0.main([])
    rep = json.loads((out_dir / "frame" / "rep_frame.json").read_text())
    assert rep["mode"] == "gt_free_heuristic"
    assert rep["frame"] == "rich.jpg"
    assert (out_dir / "frame" / "rich.jpg").exists()


def test_no_gt_env_var_and_cli_flag_force_fallback_even_with_gt_present(
        tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_intrinsics(scene_dir)
    _write_mug_mesh(scene_dir)
    names = ["frame_a.jpg", "frame_b.jpg"]
    cams = {"frame_a.jpg": IDENTITY_Q + (0.0, 0.0, 0.5),
            "frame_b.jpg": IDENTITY_Q + (0.0, 0.0, 5.0)}
    _write_images_txt(scene_dir, cams)
    _write_images(scene_dir, names, textured={"frame_a.jpg"})
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=True)

    assert s0.gt_instances_available()  # GT genuinely IS available here

    # 1. explicit --no-gt CLI flag wins over available GT
    s0.main(["--no-gt"])
    rep = json.loads((out_dir / "frame" / "rep_frame.json").read_text())
    assert rep["mode"] == "gt_free_heuristic"

    # 2. SIMANY_NO_GT=1 has the same effect with no CLI flag
    monkeypatch.setenv("SIMANY_NO_GT", "1")
    assert s0.no_gt_requested() is True
    s0.main([])
    rep2 = json.loads((out_dir / "frame" / "rep_frame.json").read_text())
    assert rep2["mode"] == "gt_free_heuristic"


# ============================================================ s5: GT-free trigger

def test_s5_reports_not_applicable_without_gt(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    odir = _write_object_fixture(out_dir, gt_object_id=None)
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=False)

    assert not s5.gt_instances_available()
    monkeypatch.setattr(C, "load_gt_instances",
                         lambda: (_ for _ in ()).throw(
                             AssertionError("must not be called in GT-free mode")))

    s5.main([])  # must not crash

    aligned = json.loads((odir / "aligned.json").read_text())
    assert aligned["eval"] == {
        "value": None, "status": "not_applicable",
        "reason": s5.GT_NOT_APPLICABLE_REASON,
    }
    # the Sim(3) pose-alignment part still ran and produced real numbers
    assert np.isfinite(aligned["scale"])
    assert np.all(np.isfinite(np.asarray(aligned["T"])))
    assert np.isfinite(aligned["chamfer_med_m"])


def test_s5_no_gt_override_forces_not_applicable_even_with_gt_present(
        tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_mug_mesh(scene_dir)
    odir = _write_object_fixture(out_dir, gt_object_id=1)
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=True)

    assert s5.gt_instances_available()
    s5.main(["--no-gt"])

    aligned = json.loads((odir / "aligned.json").read_text())
    assert aligned["eval"]["status"] == "not_applicable"
    assert aligned["eval"]["value"] is None


# ==================================================== GT-available regression

def test_gt_available_still_takes_original_path_for_s0(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_intrinsics(scene_dir)
    _write_mug_mesh(scene_dir)
    names = ["frame_a.jpg", "frame_b.jpg"]
    cams = {"frame_a.jpg": IDENTITY_Q + (0.0, 0.0, 0.5),
            "frame_b.jpg": IDENTITY_Q + (0.0, 0.0, 5.0)}
    _write_images_txt(scene_dir, cams)
    _write_images(scene_dir, names)
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=True)

    assert s0.gt_instances_available()
    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()

    direct = s0.select_frame_gt(K, W, H, w2c_all)
    s0.main([])
    rep = json.loads((out_dir / "frame" / "rep_frame.json").read_text())

    assert "mode" not in rep            # GT path never tags gt_free_heuristic
    assert rep["frame"] == direct["frame"] == "frame_a.jpg"
    assert rep["visible"] and rep["visible"][0]["label"] == "mug"
    assert rep == direct                # main() dispatches to the unmodified GT path verbatim


def test_gt_available_still_takes_original_path_for_s5(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_mug_mesh(scene_dir)
    odir = _write_object_fixture(out_dir, gt_object_id=1)
    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=True)

    assert s5.gt_instances_available()
    s5.main([])

    aligned = json.loads((odir / "aligned.json").read_text())
    assert "eval" in aligned
    assert aligned["eval"].get("status") is None    # NOT the not_applicable shape
    assert set(aligned["eval"].keys()) >= {"f1@20mm", "f1@40mm", "chamfer_mean_m"}
    assert aligned["eval"]["f1@20mm"]["f1"] >= 0.0


# ======================================================== safety: no GT reads

def _forbidden_paths(opened_paths):
    out = []
    for p in opened_paths:
        parts = {seg.lower() for seg in Path(p).parts}
        name = Path(p).name.lower()
        if "gt" in parts or "segments" in name or "anno" in name:
            out.append(p)
    return out


def test_s0_gt_free_fallback_never_touches_gt_shaped_paths(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    _write_intrinsics(scene_dir)
    names = ["flat_a.jpg", "rich.jpg"]
    _write_images_txt(scene_dir, {n: IDENTITY_Q + (0.0, 0.0, float(i))
                                   for i, n in enumerate(names)})
    _write_images(scene_dir, names, textured={"rich.jpg"})

    # Decoy forbidden files physically present at their natural conventional
    # locations -- proves the fallback doesn't wander into them via some
    # path OTHER than the (redirected-away) C.* constants below.
    scans = scene_dir / "scans"
    scans.mkdir(parents=True)
    (scans / "segments.json").write_text(json.dumps({"segIndices": [0]}))
    (scans / "segments_anno.json").write_text(json.dumps({"segGroups": []}))
    gtdir = scene_dir / "gt"
    gtdir.mkdir(parents=True)
    (gtdir / "gt_objects.json").write_text("{}")
    decoy_mesh = gtdir / "mesh_gt.ply"
    trimesh.creation.icosphere(subdivisions=1, radius=0.05).export(str(decoy_mesh))

    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=False)
    # even mis-point the mesh-source constant AT a gt/ file -- is_gt_path()
    # must refuse to touch it regardless of who set it.
    monkeypatch.setattr(C, "PIPELINE_MESH_PLY", decoy_mesh)
    assert not s0.gt_instances_available()
    assert s0.is_gt_path(decoy_mesh)

    monkeypatch.setattr(
        C, "load_gt_instances",
        lambda: (_ for _ in ()).throw(
            AssertionError("load_gt_instances() must not be called by the GT-free path")))

    opened = []
    real_open = Path.open

    def _tracking_open(self, *a, **k):
        opened.append(str(self))
        return real_open(self, *a, **k)
    monkeypatch.setattr(Path, "open", _tracking_open)

    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()
    best = s0.select_frame_gt_free(K, W, H, w2c_all)

    assert best["mode"] == "gt_free_heuristic"
    assert best["coverage_score"] is None   # mesh at a gt/ path -> skipped entirely

    forbidden = _forbidden_paths(opened)
    assert forbidden == [], f"GT-free frame selection touched: {forbidden}"


def test_s5_gt_free_fallback_never_touches_gt_shaped_paths(tmp_path, monkeypatch):
    scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
    odir = _write_object_fixture(out_dir, gt_object_id=None)

    scans = scene_dir / "scans"
    scans.mkdir(parents=True)
    (scans / "segments.json").write_text(json.dumps({"segIndices": [0]}))
    (scans / "segments_anno.json").write_text(json.dumps({"segGroups": []}))
    gtdir = scene_dir / "gt"
    gtdir.mkdir(parents=True)
    (gtdir / "gt_objects.json").write_text("{}")

    _patch_common(monkeypatch, scene_dir, out_dir, with_gt=False)
    assert not s5.gt_instances_available()

    monkeypatch.setattr(
        C, "load_gt_instances",
        lambda: (_ for _ in ()).throw(
            AssertionError("load_gt_instances() must not be called by the GT-free path")))

    opened = []
    real_open = Path.open

    def _tracking_open(self, *a, **k):
        opened.append(str(self))
        return real_open(self, *a, **k)
    monkeypatch.setattr(Path, "open", _tracking_open)

    s5.main([])  # must not crash, must not call load_gt_instances

    forbidden = _forbidden_paths(opened)
    assert forbidden == [], f"s5 GT-free path touched: {forbidden}"

    aligned = json.loads((odir / "aligned.json").read_text())
    assert aligned["eval"]["status"] == "not_applicable"
