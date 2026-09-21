"""CPU tests for physicalview.scene_state (synthetic trees + real result sets)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
import pytest

from physicalview import scene_state as S
from physicalview.config import load_config

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"


def _cfg_for(tmp_path: Path):
    """Real config with every filesystem root redirected into tmp_path."""
    cfg = load_config()
    return dataclasses.replace(
        cfg, repo_root=tmp_path, outputs_root=tmp_path / "outputs",
        scannetpp_root=tmp_path / "scannetpp", splats_root=tmp_path / "splats",
        studio_out=tmp_path / "studio")


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _make_result_set(root: Path, name: str, n_obj: int = 2, n_rej: int = 1) -> Path:
    out = root / "outputs" / name
    objs, aligned_all = [], []
    for i in range(n_obj):
        meta = {"index": i, "label": f"thing{i}", "aabb": [[0, 0, 0], [1, 1, 1]], "centroid": [0.5, 0.5, 0.5]}
        objs.append(meta)
        T = np.eye(4).tolist()
        al = {"index": i, "label": meta["label"], "scale": 1.0, "T": T, "tier": "A",
              "world_dims": [0.1, 0.2, 0.3], "vmesh_source": "trellis"}
        if i < n_rej:
            al["rejected"] = "size ratio 4.10 outside (0.4, 2.5)"
            al["tier"] = "C"
        aligned_all.append(al)
        odir = out / "objects" / f"obj_{i:02d}"
        _write(odir / "meta.json", meta)
        _write(odir / "aligned.json", al)
        (odir / "trellis").mkdir(parents=True, exist_ok=True)
        (odir / "trellis" / "trellis_mesh.ply").write_bytes(b"")
        (odir / "trellis" / "trellis_gs.ply").write_bytes(b"")
        _write(odir / "trellis" / "aligned.json", {**al, "size_ratio_vs_obs": 0.9})
    _write(out / "objects" / "objects.json", objs)
    _write(out / "objects" / "aligned_all.json", aligned_all)
    return out


# ------------------------------------------------------------- pure logic ----

@pytest.mark.parametrize("name,kind,scene", [
    ("c50d2d1d42_factory", "factory", "c50d2d1d42"),
    ("c50d2d1d42_auto", "auto", "c50d2d1d42"),
    ("09c1414f1b_full", "full", "09c1414f1b"),
    ("09c1414f1b_rowC", "other", "09c1414f1b"),
    ("09c1414f1b_rowC2", "other", "09c1414f1b"),
    ("09c1414f1b", "other", "09c1414f1b"),
    ("droid_iris_mon_apr_17_16_03_25_2023", "droid", "droid_iris_mon_apr_17_16_03_25_2023"),
    ("video_pilot_a29cccc784", "video", "pilot_a29cccc784"),
    ("behavior_task0002_severe", "behavior", "behavior_task0002_severe"),
    # outputs/behavior_task-0020 was built as scene behavior_task0020 (dashes stripped)
    ("behavior_task-0020", "behavior", "behavior_task0020"),
])
def test_classify_result_dir(name, kind, scene):
    assert S.classify_result_dir(name) == (kind, scene)


def test_parse_timings_last_block_wins(tmp_path):
    p = tmp_path / "timings.txt"
    p.write_text("s4 TRELLIS image-to-3D 196\nfactory_align (register) 242\n\n"
                 "s4 TRELLIS image-to-3D 35\nnot a timing line\nfactory_align (register) 363\n")
    t = S.parse_timings(p)
    assert t == {"s4 TRELLIS image-to-3D": 35.0, "factory_align (register)": 363.0}
    assert list(t) == ["s4 TRELLIS image-to-3D", "factory_align (register)"]
    assert S.parse_timings(tmp_path / "missing.txt") == {}


# --------------------------------------------------------------- synthetic ----

def test_discover_synthetic_tree(tmp_path):
    cfg = _cfg_for(tmp_path)
    _make_result_set(tmp_path, "abc_factory")
    _make_result_set(tmp_path, "abc_auto", n_obj=3, n_rej=0)
    (tmp_path / "outputs" / "abc_auto" / "inpaint").mkdir(parents=True)
    (tmp_path / "outputs" / "abc_auto" / "inpaint" / "clean_background.ply").write_bytes(b"")
    _write(tmp_path / "outputs" / "abc_auto" / "sim_export" / "scene.xml", "<mujoco/>")
    _write(tmp_path / "outputs" / "abc_auto" / "sim_export" / "pi05_tasks.json", {"tasks": []})
    _make_result_set(tmp_path, "xyz", n_obj=1, n_rej=0)          # plain dir -> other
    droid = tmp_path / "outputs" / "droid_lab_1"                     # recon-only build
    (droid / "recon").mkdir(parents=True)
    (droid / "derived_mesh.ply").write_bytes(b"")
    (tmp_path / "outputs" / "junk_dir").mkdir()                       # ignored
    (tmp_path / "outputs" / "file.log").write_text("x")             # ignored
    (tmp_path / "splats").mkdir()
    for sid in ("abc", "unb"):
        (tmp_path / "splats" / f"{sid}.ply").write_bytes(b"")
    _write(tmp_path / "scannetpp" / "splits" / "nvs_sem_val.txt", "abc\nunb\nnosplat\n")
    (tmp_path / "scannetpp" / "data" / "abc" / "scans").mkdir(parents=True)
    (tmp_path / "scannetpp" / "data" / "abc" / "scans" / "mesh_aligned_0.05.ply").write_bytes(b"")
    (tmp_path / "data" / "recon_scenes" / "data" / "droid_lab_1").mkdir(parents=True)

    rs = S.discover_result_sets(cfg)
    by = {r.name: r for r in rs}
    assert set(by) == {"abc_factory", "abc_auto", "xyz", "droid_lab_1", "unb"}
    assert [r.kind for r in rs[:2]] == ["auto", "factory"] or [r.kind for r in rs[:2]] == ["factory", "auto"]
    assert rs[-1].kind == "unbuilt" and rs[-1].name == "unb"
    f = by["abc_factory"]
    assert (f.kind, f.scene_id, f.n_objects, f.n_accepted) == ("factory", "abc", 2, 1)
    assert f.scene_dir == tmp_path / "scannetpp" / "data" / "abc"
    assert f.splat_ply == tmp_path / "splats" / "abc.ply"
    assert f.mesh_ply == tmp_path / "scannetpp" / "data" / "abc" / "scans" / "mesh_aligned_0.05.ply"
    assert not f.has_clean_bg and not f.has_sim_export and not f.has_tasks
    a = by["abc_auto"]
    assert (a.n_objects, a.n_accepted, a.has_clean_bg, a.has_sim_export, a.has_tasks) == (3, 3, True, True, True)
    assert by["xyz"].kind == "other"
    d = by["droid_lab_1"]
    assert d.kind == "droid" and d.n_objects == 0
    assert d.scene_dir == tmp_path / "data" / "recon_scenes" / "data" / "droid_lab_1"
    assert d.mesh_ply == droid / "derived_mesh.ply"
    assert d.splat_ply is None
    u = by["unb"]
    assert u.splat_ply == tmp_path / "splats" / "unb.ply" and u.scene_dir is None


def test_discover_missing_outputs_root(tmp_path):
    cfg = _cfg_for(tmp_path)
    assert S.discover_result_sets(cfg) == []


@pytest.mark.backend
def test_load_scene_and_reload_synthetic(tmp_path):
    cfg = _cfg_for(tmp_path)
    out = _make_result_set(tmp_path, "abc_factory")
    _write(out / "drop_v2.json", {"objects": [{"name": "obj_01", "label": "thing1", "drift_m": 0.001, "stable": True}]})
    _write(out / "report.json", {"n_instances": 2, "objects": [{"index": 0, "tier": "C"}, {"index": 1, "tier": "A"}]})
    (out / "timings.txt").write_text("stage a 1\nstage b 2\n")
    (out / "objects" / "obj_01" / "collision").mkdir()
    for i in range(3):
        (out / "objects" / "obj_01" / "collision" / f"part_{i:02d}.obj").write_bytes(b"")
    rs = next(r for r in S.discover_result_sets(cfg) if r.name == "abc_factory")
    st = S.load_scene(cfg, rs)
    assert st.K is None and st.W == 0 and st.cameras == {}     # no scene dir -> graceful
    assert st.splat_gs is None and st.clean_bg_gs is None and st.mesh is None
    assert st.tasks is None and st.scene_xml is None and st.sim_export == out / "sim_export"
    assert st.timings == {"stage a": 1.0, "stage b": 2.0}
    assert st.report["n_instances"] == 2
    assert set(st.objects) == {"obj_00", "obj_01"}
    o0, o1 = st.objects["obj_00"], st.objects["obj_01"]
    assert not o0.accepted and o0.rejected_reason.startswith("size ratio") and o0.tier == "C"
    assert o1.accepted and o1.rejected_reason is None and o1.tier == "A"
    assert np.allclose(o1.T_world, np.eye(4)) and o1.world_dims == [0.1, 0.2, 0.3]
    assert o1.chosen_source == "trellis" and list(o1.proposals) == ["trellis"]
    assert o1.proposals["trellis"].evidence["size_ratio"] == 0.9
    assert o1.collision_parts == 3 and o0.collision_parts == 0
    assert o1.drop_test == {"drift_m": 0.001, "stable": True} and o1.stable is True
    assert o0.drop_test is None and o0.stable is None
    assert o1.artifacts["collision"] and o1.artifacts["trellis_mesh"] and not o1.artifacts["physics"]
    assert o1.f1_20 is None
    assert st.edited_poses == {}

    # nothing changed -> no ids
    assert S.reload_objects(st) == []
    # a job wrote physics + a new object -> both reported
    _write(out / "objects" / "obj_01" / "physics.json", {"mass_kg": 0.3})
    objs = json.loads((out / "objects" / "objects.json").read_text())
    objs.append({"index": 2, "label": "new"})
    _write(out / "objects" / "objects.json", objs)
    (out / "objects" / "obj_02").mkdir()
    changed = S.reload_objects(st)
    assert changed == ["obj_01", "obj_02"]
    assert st.objects["obj_01"].physics == {"mass_kg": 0.3}
    assert not st.objects["obj_02"].accepted and st.objects["obj_02"].aligned is None


@pytest.mark.backend
def test_object_canonical_gs_missing(tmp_path):
    cfg = _cfg_for(tmp_path)
    _make_result_set(tmp_path, "abc_factory")
    # drop the (empty) gaussians so the lookup fails cleanly
    (tmp_path / "outputs" / "abc_factory" / "objects" / "obj_01" / "trellis" / "trellis_gs.ply").unlink()
    rs = S.discover_result_sets(cfg)[0]
    st = S.load_scene(cfg, rs)
    assert st.objects["obj_01"].proposals["trellis"].gs_ply is None
    with pytest.raises(FileNotFoundError):
        S.object_canonical_gs(st, "obj_01")
    with pytest.raises(KeyError):
        S.object_canonical_gs(st, "obj_99")


# ---------------------------------------------------------------- real data ----

needs_outputs = pytest.mark.skipif(
    not (OUTPUTS / "c50d2d1d42_factory" / "objects" / "objects.json").exists(),
    reason="real outputs/ not available")


@pytest.fixture(scope="module")
def real_sets():
    if not (OUTPUTS / "c50d2d1d42_factory" / "objects" / "objects.json").exists():
        pytest.skip("real outputs/ not available")
    cfg = load_config()
    t0 = time.time()
    rs = S.discover_result_sets(cfg)
    return cfg, rs, time.time() - t0


@needs_outputs
def test_real_discovery_fast_and_classified(real_sets):
    cfg, rs, dt = real_sets
    assert dt < 3.0, f"discovery took {dt:.2f}s"
    by = {r.name: r for r in rs}
    assert by["c50d2d1d42_factory"].kind == "factory"
    assert by["c50d2d1d42_factory"].n_objects == 18 and by["c50d2d1d42_factory"].n_accepted == 16
    assert by["c50d2d1d42_factory"].has_clean_bg and by["c50d2d1d42_factory"].has_tasks
    assert by["c50d2d1d42_auto"].kind == "auto"
    kinds = [r.kind for r in rs]
    first_other = next((i for i, k in enumerate(kinds) if k not in ("factory", "auto")), len(kinds))
    assert all(k in ("factory", "auto") for k in kinds[:first_other])
    assert all(k not in ("factory", "auto") for k in kinds[first_other:])
    if "droid_iris_mon_apr_17_16_03_25_2023" in by:
        d = by["droid_iris_mon_apr_17_16_03_25_2023"]
        assert d.kind == "droid" and d.mesh_ply is not None and d.mesh_ply.name == "derived_mesh.ply"
    if "video_pilot_a29cccc784" in by:
        assert by["video_pilot_a29cccc784"].kind == "video"
        assert by["video_pilot_a29cccc784"].scene_id == "pilot_a29cccc784"


@needs_outputs
def test_real_load_factory_scene(real_sets):
    cfg, rs, _ = real_sets
    r = next(x for x in rs if x.name == "c50d2d1d42_factory")
    st = S.load_scene(cfg, r, load_splats=False)
    assert len(st.objects) == 18 and sum(o.accepted for o in st.objects.values()) == 16
    if r.scene_dir is not None:
        assert st.K.shape == (3, 3) and st.W == 1752 and st.H == 1168 and len(st.cameras) > 100
        w2c = next(iter(st.cameras.values()))
        assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-5)
    assert st.tasks is not None and len(st.tasks["tasks"]) >= 1 and st.scene_xml is not None
    assert "s4 TRELLIS image-to-3D" in st.timings
    o = st.objects["obj_00"]
    assert o.accepted and o.tier == "A" and o.chosen_source == "sam3d"
    assert set(o.proposals) == {"trellis", "rvg", "sam3d"}
    assert o.proposals["sam3d"].gs_ply is not None and o.proposals["sam3d"].gs_ply.name == "sam3d_gs.ply"
    assert o.proposals["rvg"].evidence["sym_chamfer_m"] == pytest.approx(0.02447062708785491)
    assert o.proposals["sam3d"].evidence.get("gen_seconds") == pytest.approx(14.941953182220459)
    assert "f1_20" in o.proposals["trellis"].eval_only_fields
    assert o.f1_20 == pytest.approx(0.9841011835221212)
    assert o.stable is True and o.collision_parts == 2 and o.artifacts["inpainted"]
    rej = st.objects["obj_01"]
    assert not rej.accepted and "size ratio" in rej.rejected_reason


@needs_outputs
def test_real_rvg_winner_and_canonical_gs(real_sets):
    cfg, rs, _ = real_sets
    r = next((x for x in rs if x.name == "27dd4da69e_factory"), None)
    if r is None:
        pytest.skip("27dd4da69e_factory absent")
    st = S.load_scene(cfg, r, load_splats=False)
    o = st.objects["obj_01"]
    assert o.chosen_source == "rvg" and set(o.proposals) == {"trellis", "rvg"}
    assert o.proposals["rvg"].dir == o.dir / "rvg" and o.proposals["trellis"].dir == o.dir / "trellis"
    assert o.proposals["rvg"].evidence["gen_seconds"] == pytest.approx(16.07576274871826)
    assert o.proposals["trellis"].evidence["tier"] == "B" and o.proposals["rvg"].evidence["tier"] == "A"
    assert S.canonical_gs_path(o)[0] == "rvg"
    gs = S.object_canonical_gs(st, "obj_01")
    assert gs["means"].shape[1] == 3 and gs["means"].shape[0] > 1000 and gs["means"].device.type == "cpu"
    assert S.object_canonical_gs(st, "obj_01") is gs          # cached
    gt = S.object_canonical_gs(st, "obj_01", "trellis")
    assert gt["means"].shape[0] != gs["means"].shape[0]


@needs_outputs
@pytest.mark.parametrize("name", ["c50d2d1d42_auto", "droid_iris_mon_apr_17_16_03_25_2023", "video_pilot_a29cccc784"])
def test_real_other_kinds_load(real_sets, name):
    cfg, rs, _ = real_sets
    r = next((x for x in rs if x.name == name), None)
    if r is None:
        pytest.skip(f"{name} absent")
    st = S.load_scene(cfg, r, load_splats=False)
    assert len(st.objects) == r.n_objects
    assert sum(o.accepted for o in st.objects.values()) == r.n_accepted
    acc = [o for o in st.objects.values() if o.accepted]
    assert acc and all(o.T_world.shape == (4, 4) for o in acc)
    # flat legacy layout: trellis assets in the object dir itself
    assert all("trellis" in o.proposals and o.proposals["trellis"].gs_ply is not None for o in acc)
    if r.scene_dir is not None:
        assert st.K is not None and len(st.cameras) > 0
    assert st.timings
