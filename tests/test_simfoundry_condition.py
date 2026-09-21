"""Tests for robo/eval/simfoundry_condition.py -- the adapter that turns a
finished `baselines/simfoundry_repro.py` output directory into the third
`mujoco_paired` condition (Task 09 follow-up; design context:
docs/MUJOCO_PAIRED_PROTOCOL.md, configs/experiments/icra_contract_v1.yaml's
mujoco_paired.admissible_baselines).

Two groups of tests:

  1. Pure-logic / subprocess tests against robo.eval.simfoundry_condition
     directly -- no GPU needed (export_mjcf.py --test only steps physics
     headlessly via robo.sim.room_collision.benchmark_model, never
     constructs a mujoco.Renderer/EGL context).
  2. Integration tests that feed an incomplete/missing simfoundry_repro
     build through robo.eval.paired_runner's OWN condition machinery
     (build_env / run_matrix) and check it is classified as
     Outcome.BUILD_FAILURE -- also GPU-free, because that classification
     happens before any DroidSimEnv is ever constructed.

Run: .venv/bin/python -m pytest -q tests/test_simfoundry_condition.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robo.eval import episode_log as elog
from robo.eval import paired_runner as pr
from robo.eval import simfoundry_condition as sfcond

pytestmark = pytest.mark.filterwarnings("ignore")


# ========================================================== synthetic fixture
# A minimal `objects/` tree in the SAME shape agents.discover.s3_lift ->
# agents.assets.s5_align/s6_physics always produce (objects.json + per-object
# aligned.json/physics.json/object.urdf/collision/part_*.obj), which is what
# robo/sim/export_mjcf.py actually reads -- regardless of which pipeline
# configuration (full SimAny, or the simfoundry_repro ablation) produced it.

def _cube_obj(half: float) -> str:
    """A tiny watertight cube, centered at the origin, side `2*half` --
    valid input for both MuJoCo's mesh compiler and export_mjcf.py's own
    trimesh convex-hull usability check (>=4 vertices, hull volume >=
    1e-9 m^3)."""
    h = half
    verts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
             (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
             (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]
    lines = [f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}" for v in verts]
    lines += [f"f {f[0]} {f[1]} {f[2]}" for f in faces]
    return "\n".join(lines) + "\n"


_SPECS = [
    (0, "mug", (0.30, 0.05, 0.55), (0.08, 0.08, 0.10), 0.05, 0.6),
    (1, "bowl", (0.30, -0.15, 0.53), (0.12, 0.12, 0.06), 0.10, 0.5),
]


def _write_recon_dir(root: Path, specs=_SPECS) -> Path:
    """Write a standalone `objects/` tree at `root` -- deliberately NOT
    nested under any `sim_export`/`frame`/`masks` directory, since
    export_mjcf.py reads only `objects/` (see module docstring's claim,
    checked against robo/sim/export_mjcf.py directly: it never touches
    `frame/`, `masks/`, `depth/`)."""
    objdir = root / "objects"
    objdir.mkdir(parents=True)
    rows = []
    for index, label, pos, dims, mass, friction in specs:
        name = f"obj_{index:02d}"
        odir = objdir / name
        (odir / "collision").mkdir(parents=True)
        T = [[1.0, 0.0, 0.0, pos[0]], [0.0, 1.0, 0.0, pos[1]],
             [0.0, 0.0, 1.0, pos[2]], [0.0, 0.0, 0.0, 1.0]]
        (odir / "aligned.json").write_text(json.dumps(
            {"T": T, "world_dims": list(dims), "tier": "A", "rejected": False}))
        (odir / "physics.json").write_text(json.dumps(
            {"mass_kg": mass, "friction": friction, "restitution": 0.3}))
        (odir / "object.urdf").write_text("<robot name='stub'/>")  # existence-only
        half = min(dims) / 2.0 * 0.9
        (odir / "mesh_sim.obj").write_text(_cube_obj(half))
        (odir / "collision" / "part_0.obj").write_text(_cube_obj(half))
        half_dims = [d / 2.0 for d in dims]
        aabb = [[pos[i] - half_dims[i] for i in range(3)],
                [pos[i] + half_dims[i] for i in range(3)]]
        rows.append({"index": index, "label": label, "aabb": aabb})
    (objdir / "objects.json").write_text(json.dumps(rows))
    return root


def _geom_ids_for_body(model, name):
    bid = model.body(name).id
    start = int(model.body_geomadr[bid])
    n = int(model.body_geomnum[bid])
    return list(range(start, start + n))


# =============================================== pure-logic / path resolution

def test_default_baseline_dir_strips_factory_suffix():
    got = sfcond.default_baseline_dir("c50d2d1d42_factory")
    assert got == sfcond.ROOT / "outputs" / "c50d2d1d42_baselines" / "simfoundry_repro"


def test_default_baseline_dir_leaves_bare_scene_id_unchanged():
    got = sfcond.default_baseline_dir("c50d2d1d42")
    assert got == sfcond.ROOT / "outputs" / "c50d2d1d42_baselines" / "simfoundry_repro"


def test_load_manifest_returns_none_when_absent(tmp_path):
    assert sfcond.load_manifest(tmp_path / "does_not_exist") is None


def test_load_manifest_returns_none_on_unparseable_json(tmp_path):
    (tmp_path / sfcond.MANIFEST_FILENAME).write_text("{not json")
    assert sfcond.load_manifest(tmp_path) is None


@pytest.mark.parametrize("manifest", [
    None,
    {"build_success": False},
    {"build_success": None},
    {},  # missing key entirely
])
def test_is_build_complete_false_cases(manifest):
    assert sfcond.is_build_complete(manifest) is False


def test_is_build_complete_true_only_for_exact_true():
    assert sfcond.is_build_complete({"build_success": True}) is True


# ================================================== require_complete_build

def test_require_complete_build_raises_when_manifest_missing(tmp_path):
    baseline_dir = tmp_path / "not_started"
    with pytest.raises(sfcond.SimFoundryReproNotReady, match="no simfoundry_repro_manifest.json"):
        sfcond.require_complete_build(baseline_dir)


def test_require_complete_build_raises_with_failed_stage_reason(tmp_path):
    """Mirrors the REAL manifests currently on disk for the 6 in-flight
    scenes (outputs/<scene>_baselines/simfoundry_repro/
    simfoundry_repro_manifest.json, all build_success: false as of this
    writing, all failing at 's2 DA3 metric depth' with a HF cache-miss) --
    the reason string must name the actual failed stage, not a generic
    'build failed'."""
    baseline_dir = tmp_path / "failed_build"
    baseline_dir.mkdir()
    (baseline_dir / sfcond.MANIFEST_FILENAME).write_text(json.dumps({
        "build_success": False,
        "stage_log": [
            {"stage": "s0 select representative frame",
             "module": "agents.discover.s0_select_frame", "returncode": 0},
            {"stage": "s2 DA3 metric depth", "module": "models.s2_depth",
             "returncode": 1},
        ],
    }))
    with pytest.raises(sfcond.SimFoundryReproNotReady, match="s2 DA3 metric depth"):
        sfcond.require_complete_build(baseline_dir)


def test_require_complete_build_returns_manifest_when_flag_true(tmp_path):
    baseline_dir = tmp_path / "finished_build"
    baseline_dir.mkdir()
    manifest = {"build_success": True, "stage_log": []}
    (baseline_dir / sfcond.MANIFEST_FILENAME).write_text(json.dumps(manifest))
    got = sfcond.require_complete_build(baseline_dir)
    assert got == manifest


# ======================================================== instance_inventory

def test_instance_inventory_matches_reference_scene_shape(tmp_path):
    """Same [{object_id, label, tier, asset_hash}] shape
    robo.eval.reference_scene.instance_inventory returns for the
    "reference" condition -- robo.eval.paired_runner._scene_manifest_hash
    treats both the same way."""
    from robo.eval import reference_scene as refscene

    baseline_dir = _write_recon_dir(tmp_path / "baseline")
    rows = sfcond.instance_inventory(baseline_dir)
    assert sorted(r["object_id"] for r in rows) == ["obj_00", "obj_01"]
    for r in rows:
        assert set(r) == {"object_id", "label", "tier", "asset_hash"}
        assert r["asset_hash"] is None

    # robo.eval.reference_scene reads the identical objects/ layout and
    # must agree on which objects are structurally valid.
    ref_rows = refscene.instance_inventory(baseline_dir)
    assert sorted(r["object_id"] for r in rows) == sorted(r["object_id"] for r in ref_rows)


def test_instance_inventory_empty_when_no_objects_json(tmp_path):
    assert sfcond.instance_inventory(tmp_path / "nothing_here") == []


# ============================================ scene export structural shape

def test_scene_export_produces_scene_xml(tmp_path):
    baseline_dir = _write_recon_dir(tmp_path / "sf_repro_build")
    result = sfcond.build_scene_export(baseline_dir, "unit_test_scene",
                                       collision_mode="shim")
    assert result["returncode"] == 0, result["stderr_tail"]
    assert result["scene_xml"].exists()
    assert result["settle_json"].exists()  # --test always requested


def test_scene_export_is_idempotent_without_force(tmp_path):
    baseline_dir = _write_recon_dir(tmp_path / "sf_repro_build2")
    first = sfcond.build_scene_export(baseline_dir, "unit_test_scene",
                                      collision_mode="shim")
    assert first["skipped"] is False
    second = sfcond.build_scene_export(baseline_dir, "unit_test_scene",
                                       collision_mode="shim")
    assert second["skipped"] is True


def test_scene_export_structural_equivalence_with_export_mjcf_shape(tmp_path):
    """The core acceptance criterion: given the SAME object shape,
    robo.eval.simfoundry_condition.build_scene_export produces a scene.xml
    the same way robo/sim/export_mjcf.py would for a "simany" factory
    build with that shape -- same collision-group convention (visual
    group 2 / collision group 3), same contype/conaffinity scheme, same
    body mass -- NOT byte-identical (different out-dir paths/scene ids
    bake into <compiler meshdir> and the model name).
    """
    import mujoco

    dir_simany_like = _write_recon_dir(tmp_path / "simany_like")
    dir_sfrepro_like = _write_recon_dir(tmp_path / "sfrepro_like")

    res_a = sfcond.build_scene_export(dir_simany_like, "scene_a", collision_mode="shim")
    res_b = sfcond.build_scene_export(dir_sfrepro_like, "scene_b", collision_mode="shim")
    assert res_a["returncode"] == 0, res_a["stderr_tail"]
    assert res_b["returncode"] == 0, res_b["stderr_tail"]

    model_a = mujoco.MjModel.from_xml_path(str(res_a["scene_xml"]))
    model_b = mujoco.MjModel.from_xml_path(str(res_b["scene_xml"]))

    # not byte-identical: absolute meshdir/model name differ by construction
    assert res_a["scene_xml"].read_text() != res_b["scene_xml"].read_text()

    for name in ("obj_00", "obj_01"):
        gids_a = _geom_ids_for_body(model_a, name)
        gids_b = _geom_ids_for_body(model_b, name)
        assert len(gids_a) == len(gids_b) == 2  # 1 visual + 1 collision part

        groups_a = sorted(int(model_a.geom_group[g]) for g in gids_a)
        groups_b = sorted(int(model_b.geom_group[g]) for g in gids_b)
        assert groups_a == groups_b == [2, 3]  # visual=2, collision=3 convention

        contype_a = sorted(int(model_a.geom_contype[g]) for g in gids_a)
        contype_b = sorted(int(model_b.geom_contype[g]) for g in gids_b)
        assert contype_a == contype_b
        conaff_a = sorted(int(model_a.geom_conaffinity[g]) for g in gids_a)
        conaff_b = sorted(int(model_b.geom_conaffinity[g]) for g in gids_b)
        assert conaff_a == conaff_b

        bid_a, bid_b = model_a.body(name).id, model_b.body(name).id
        assert model_a.body_mass[bid_a] == pytest.approx(model_b.body_mass[bid_b])


# ========================================== "not finished yet" -> build_failure
# Task requirement: an incomplete/missing simfoundry_repro_manifest.json (or
# build_success: false) must be a distinguishable coverage failure
# (Outcome.BUILD_FAILURE), never a silent skip or an unhelpful crash, when
# fed through robo.eval.paired_runner's OWN condition-selection machinery.

def _minimal_scene_cfg(tmp_path, baseline_dir, scene_id="sf_missing_scene"):
    tasks_json = tmp_path / f"{scene_id}_tasks.json"
    tasks_json.write_text(json.dumps({"scene": scene_id, "tasks": []}))
    return {"id": scene_id, "tasks_json": str(tasks_json),
            "simfoundry_repro_dir": str(baseline_dir)}


def test_build_env_missing_manifest_raises_build_failure_error(tmp_path):
    scene_cfg = _minimal_scene_cfg(tmp_path, tmp_path / "baseline_not_started")
    with pytest.raises(elog.BuildFailureError, match="no simfoundry_repro_manifest.json"):
        pr.build_env(scene_cfg, "simfoundry_repro", tmp_path)


def test_build_env_build_success_false_raises_build_failure_error(tmp_path):
    baseline_dir = tmp_path / "baseline_failed"
    baseline_dir.mkdir()
    (baseline_dir / sfcond.MANIFEST_FILENAME).write_text(json.dumps({
        "build_success": False,
        "stage_log": [{"stage": "s2 DA3 metric depth", "module": "models.s2_depth",
                        "returncode": 1}],
    }))
    scene_cfg = _minimal_scene_cfg(tmp_path, baseline_dir, scene_id="sf_failed_scene")
    with pytest.raises(elog.BuildFailureError, match="s2 DA3 metric depth"):
        pr.build_env(scene_cfg, "simfoundry_repro", tmp_path)


def test_build_env_never_raises_a_bare_unclassified_exception(tmp_path):
    """Task requirement, verbatim: 'never a silent skip or a crash with an
    unhelpful trace.' Whatever escapes build_env for this condition must be
    elog.BuildFailureError specifically -- not e.g. a bare KeyError/
    FileNotFoundError bubbling up from deep inside export_mjcf.py or
    resolve_condition."""
    scene_cfg = _minimal_scene_cfg(tmp_path, tmp_path / "totally_absent")
    try:
        pr.build_env(scene_cfg, "simfoundry_repro", tmp_path)
    except Exception as e:  # noqa: BLE001 -- this IS the assertion
        assert isinstance(e, elog.BuildFailureError), (
            f"expected elog.BuildFailureError, got {type(e).__name__}: {e}")
        assert str(e), "error message must not be empty"
    else:
        pytest.fail("expected build_env to raise for a not-yet-built scene")


def test_run_matrix_logs_build_failure_for_every_planned_episode(tmp_path):
    """End-to-end through robo.eval.paired_runner.run_matrix (the actual
    "condition machinery" the task asks this to be exercised through):
    a scene whose simfoundry_repro build hasn't produced a manifest yet
    must occupy the coverage denominator with outcome build_failure, same
    as the existing simany/reference build-failure test
    (tests/test_paired_runner.py::test_coverage_includes_failed_builds).
    """
    scene_id = "sf_matrix_scene"
    tasks_json = tmp_path / "suite.json"
    tasks_json.write_text(json.dumps({
        "scene": scene_id,
        "tasks": [{
            "task_id": f"{scene_id}__t0", "target": "obj_00",
            "target_label": "mug", "any_instance": False, "receptacle": None,
            "region": {"cx": 0.0, "cy": 0.0, "hx": 0.1, "hy": 0.1,
                       "zlo": 0.0, "zhi": 1.0},
            "instructions": {"default": "x", "vague": "x", "specific": "x"},
        }],
    }))
    scene_cfg = {"id": scene_id, "tasks_json": str(tasks_json),
                "simfoundry_repro_dir": str(tmp_path / "no_manifest_yet")}
    config = {
        "policy": "scripted_sinusoid", "seeds": [0], "episodes": 1,
        "horizon_s": 1.0, "conditions": ["simfoundry_repro"],
        "scenes": [scene_cfg], "out_dir": str(tmp_path / "run"),
    }

    summary = pr.run_matrix(config, tmp_path / "run")

    assert summary["coverage_by_outcome"] == {"build_failure": 1}
    ledger = elog.EpisodeLedger(tmp_path / "run" / "episode_ledger.jsonl").load_completed()
    (rec,) = ledger.values()
    assert rec.condition == "simfoundry_repro"
    assert rec.outcome == elog.Outcome.BUILD_FAILURE
    assert rec.error and "no simfoundry_repro_manifest.json" in rec.error


def test_run_matrix_real_failed_job_manifest_is_build_failure(tmp_path):
    """Smoke test against REAL data: as of this writing, jobs
    784635-784639 (outputs/{27dd4da69e,45b0dac5e3,578511c8a9,7b6477cb95,
    825d228aec}_baselines/simfoundry_repro/simfoundry_repro_manifest.json)
    have all already finished with build_success: false (a node-local HF
    cache miss on DA3METRIC-LARGE at s2_depth). This is the "not finished
    successfully" branch exercised against a real manifest instead of a
    synthetic one; skipped if none of those manifests exist yet (e.g. run
    from a fresh checkout with no outputs/ tree at all).
    """
    real_dirs = [sfcond.default_baseline_dir(f"{s}_factory") for s in
                 ("27dd4da69e", "45b0dac5e3", "578511c8a9", "7b6477cb95", "825d228aec")]
    real_dirs = [d for d in real_dirs if (d / sfcond.MANIFEST_FILENAME).exists()]
    if not real_dirs:
        pytest.skip("no real simfoundry_repro_manifest.json on disk yet "
                    "(jobs 784635-784640 not landed in this checkout)")
    baseline_dir = real_dirs[0]
    manifest = sfcond.load_manifest(baseline_dir)
    if sfcond.is_build_complete(manifest):
        pytest.skip(f"{baseline_dir} has already build_success: true -- "
                    "covered by the happy-path tests instead")

    scene_cfg = {"id": baseline_dir.parent.name.replace("_baselines", "") + "_factory",
                "tasks_json": str(tmp_path / "unused_tasks.json"),
                "simfoundry_repro_dir": str(baseline_dir)}
    (tmp_path / "unused_tasks.json").write_text(
        json.dumps({"scene": scene_cfg["id"], "tasks": []}))

    with pytest.raises(elog.BuildFailureError):
        pr.build_env(scene_cfg, "simfoundry_repro", tmp_path)
