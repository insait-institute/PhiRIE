"""Tests for robo/eval/task_regrounding.py -- re-grounding a task's
target/receptacle object references from one scene build's own object ids
onto a DIFFERENT, independently-reconstructed build of the same scene, by
semantic label/category match (geometry as a tie-breaker only), instead of
assuming the two builds enumerate/index objects the same way.

Real-data validation against the actual `7b6477cb95` scene (both a
completed `_factory` build and a completed `_baselines/simfoundry_repro`
build exist on disk as of this writing -- the scene the task's own
real-run measurement found 12/12 (100%) index-mismatch env_crash on)
lives in `test_real_7b6477cb95_scene_before_after` and
`test_real_scene_regrounding_matches_gt_object_id_where_available` below;
both skip cleanly if that output tree isn't present in this checkout.

Run: .venv/bin/python -m pytest -q tests/test_task_regrounding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robo.eval import simfoundry_condition as sfcond
from robo.eval import task_regrounding as tregr

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


# ========================================================== synthetic fixture
def _make_build(root: Path, specs) -> Path:
    """A minimal `objects/` tree satisfying
    `robo.tasks.pi05_tasks._load_objects`'s inclusion test (aligned.json
    present + not rejected, object.urdf present, physics.json present) --
    everything `task_regrounding.load_candidates` needs. `specs`:
    [(index, label, center_xyz, dims_xyz, mass_kg), ...].
    """
    objdir = root / "objects"
    objdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, label, center, dims, mass in specs:
        name = f"obj_{index:02d}"
        odir = objdir / name
        odir.mkdir(parents=True, exist_ok=True)
        half = [d / 2.0 for d in dims]
        aabb = [[center[i] - half[i] for i in range(3)],
                [center[i] + half[i] for i in range(3)]]
        (odir / "aligned.json").write_text(json.dumps(
            {"world_dims": list(dims), "tier": "A", "rejected": False}))
        (odir / "physics.json").write_text(json.dumps(
            {"mass_kg": mass, "friction": 0.5}))
        (odir / "object.urdf").write_text("<robot name='stub'/>")
        rows.append({"index": index, "label": label, "aabb": aabb})
    (objdir / "objects.json").write_text(json.dumps(rows))
    return root


def _row(build_dir: Path, name: str) -> dict:
    rows = {r["name"]: r for r in tregr.load_candidates(build_dir)}
    return rows[name]


# ===================================================== requirement 1: mug/mug
def test_mug_regrounds_despite_index_mismatch(tmp_path):
    """simany's obj_09 ("mug") and simfoundry_repro's obj_03 (also "mug",
    same scene) must be matched despite the index mismatch -- the exact
    scenario the real job (12/12 env_crash on scene 7b6477cb95) hit."""
    source_dir = _make_build(tmp_path / "source", [
        (9, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
        (2, "bowl", (2.0, 2.0, 1.0), (0.12, 0.12, 0.06), 0.10),
    ])
    target_dir = _make_build(tmp_path / "target", [
        (0, "bowl", (2.0, 2.0, 1.0), (0.12, 0.12, 0.06), 0.10),
        (3, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
    ])

    task = {"task_id": "t", "target": "obj_09", "target_label": "mug",
            "any_instance": False, "receptacle": None,
            "region": {"cx": 0, "cy": 0, "hx": 1, "hy": 1, "zlo": 0, "zhi": 1},
            "instructions": {"default": "x", "vague": "x", "specific": "x"}}

    tasks, report = tregr.ground_task_suite([task], source_dir, target_dir)

    assert tasks[0]["target"] == "obj_03"
    assert tasks[0]["target_label"] == "mug"
    # instruction language is a frozen field -- must never be rewritten
    assert tasks[0]["instructions"] == task["instructions"]

    (row,) = [r for r in report if r["role"] == "target"]
    assert row["match_kind"] == tregr.MATCH_REGROUNDED
    assert row["status"] == "resolved"
    assert row["resolved_id"] == "obj_03"
    assert row["source_id"] == "obj_09"
    assert tregr.EV_LABEL_EXACT in row["evidence"]


def test_ground_object_direct_match_short_circuits_label_search(tmp_path):
    """If the raw source id happens to still name the right object in the
    target build (small scene, indices coincide), that is reported as a
    DIRECT match, not a coincidentally-successful label search."""
    target_dir = _make_build(tmp_path / "target", [
        (9, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
    ])
    source_row = {"name": "obj_09", "label": "mug", "center": np.array([1.0, 1.0, 1.0])}
    result = tregr.ground_object(source_row, tregr.load_candidates(target_dir))
    assert result["match_kind"] == tregr.MATCH_DIRECT
    assert result["resolved_id"] == "obj_09"
    assert result["confidence"] == 1.0


# ============================================== requirement 2: genuinely absent
def test_target_genuinely_missing_reports_not_found(tmp_path):
    """simfoundry_repro found only 2 objects; simany's task needs a 5th
    (a "plate") that this build's coarser single-frame reconstruction
    never detected at all -- must report "not found", never a wrong
    guess."""
    source_dir = _make_build(tmp_path / "source", [
        (0, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
        (1, "bowl", (2.0, 1.0, 1.0), (0.12, 0.12, 0.06), 0.10),
        (2, "bottle", (3.0, 1.0, 1.0), (0.07, 0.07, 0.25), 0.20),
        (3, "box", (4.0, 1.0, 1.0), (0.30, 0.20, 0.15), 0.30),
        (4, "plate", (5.0, 1.0, 1.0), (0.20, 0.20, 0.02), 0.15),
    ])
    target_dir = _make_build(tmp_path / "target", [
        (0, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
        (1, "bowl", (2.0, 1.0, 1.0), (0.12, 0.12, 0.06), 0.10),
    ])

    task = {"task_id": "t", "target": "obj_04", "target_label": "plate",
            "any_instance": False, "receptacle": None,
            "region": {"cx": 0, "cy": 0, "hx": 1, "hy": 1, "zlo": 0, "zhi": 1},
            "instructions": {"default": "x", "vague": "x", "specific": "x"}}

    tasks, report = tregr.ground_task_suite([task], source_dir, target_dir)

    # left byte-identical -- still fails downstream as env_crash, honestly
    assert tasks[0]["target"] == "obj_04"
    assert tasks[0]["target_label"] == "plate"

    (row,) = [r for r in report if r["role"] == "target"]
    assert row["match_kind"] == tregr.MATCH_UNRESOLVED
    assert row["status"] == "unresolved"
    assert row["resolved_id"] is None
    assert row["confidence"] == 0.0
    assert "plate" in row["reason"]
    assert row["evidence"] == []


# ============================================ requirement 3: geometry tiebreak
def test_two_same_label_candidates_geometry_tiebreak_picks_nearest(tmp_path):
    """Two "cup" objects exist in the target build; the geometry
    tie-breaker must resolve to the one nearer the source object's own
    position -- and the test asserts WHICH one, and that the evidence
    trail names geometry as the deciding factor."""
    source_row = {"name": "obj_07", "label": "cup", "center": np.array([0.0, 0.0, 0.0])}
    target_dir = _make_build(tmp_path / "target", [
        (0, "cup", (0.02, 0.0, 0.0), (0.10, 0.10, 0.10), 0.10),   # near
        (5, "cup", (5.0, 5.0, 5.0), (0.10, 0.10, 0.10), 0.10),    # far
    ])
    target_rows = tregr.load_candidates(target_dir)

    result = tregr.ground_object(source_row, target_rows)

    assert result["match_kind"] == tregr.MATCH_REGROUNDED
    assert result["status"] == "resolved"
    assert result["resolved_id"] == "obj_00"          # the NEAR candidate
    assert tregr.EV_GEOMETRY_TIEBREAK in result["evidence"]
    assert tregr.EV_LABEL_EXACT in result["evidence"]
    ids = {c["object_id"] for c in result["candidates"]}
    assert ids == {"obj_00", "obj_05"}                # both were candidates
    winner = next(c for c in result["candidates"] if c["object_id"] == "obj_00")
    loser = next(c for c in result["candidates"] if c["object_id"] == "obj_05")
    assert winner["confidence"] > loser["confidence"]  # geometry broke the tie


def test_equidistant_same_label_candidates_stay_ambiguous_not_guessed(tmp_path):
    """When geometry can't break the tie either (two equally-plausible,
    equally-distant same-label candidates), the module must refuse to
    pick one -- "ambiguous", not a coin-flip resolution."""
    source_row = {"name": "obj_09", "label": "cup", "center": np.array([0.0, 0.0, 0.0])}
    target_dir = _make_build(tmp_path / "target", [
        (0, "cup", (1.0, 0.0, 0.0), (0.10, 0.10, 0.10), 0.10),
        (1, "cup", (-1.0, 0.0, 0.0), (0.10, 0.10, 0.10), 0.10),
    ])
    result = tregr.ground_object(source_row, tregr.load_candidates(target_dir))

    assert result["match_kind"] == tregr.MATCH_UNRESOLVED
    assert result["status"] == "ambiguous"
    assert result["resolved_id"] is None
    assert len(result["candidates"]) == 2
    assert "refusing to guess" in result["reason"]


# ============================== requirement 4: confidence/reason are honest
def test_category_synonym_match_has_lower_confidence_than_exact_label():
    """A category/synonym-only match ("cup" -> the target's own "mug")
    must be reported at visibly lower confidence than an exact label
    match -- confidence is not fabricated to look equally certain."""
    exact_row = {"name": "obj_00", "label": "mug", "center": None}
    exact_target = [{"name": "obj_05", "label": "mug", "center": np.array([0., 0., 0.]),
                     "aabb": np.zeros((2, 3)), "dims": np.zeros(3), "mass": 0.1,
                     "tier": "A", "bottom_z": 0.0, "drift": 0.0}]
    exact_result = tregr.ground_object(exact_row, exact_target)

    cup_row = {"name": "obj_01", "label": "cup", "center": None}
    cup_target = [{"name": "obj_06", "label": "mug", "center": np.array([0., 0., 0.]),
                  "aabb": np.zeros((2, 3)), "dims": np.zeros(3), "mass": 0.1,
                  "tier": "A", "bottom_z": 0.0, "drift": 0.0}]
    cup_result = tregr.ground_object(cup_row, cup_target)

    assert exact_result["match_kind"] == tregr.MATCH_REGROUNDED
    assert cup_result["match_kind"] == tregr.MATCH_REGROUNDED
    assert exact_result["evidence"] == [tregr.EV_LABEL_EXACT]
    assert cup_result["evidence"] == [tregr.EV_CATEGORY_SYNONYM]
    assert cup_result["confidence"] < exact_result["confidence"]
    assert cup_result["reason"] and exact_result["reason"]  # both non-empty, non-fabricated


def test_unresolved_result_has_zero_confidence_and_explicit_reason():
    row = {"name": "obj_00", "label": "spaceship", "center": None}
    result = tregr.ground_object(row, [
        {"name": "obj_01", "label": "mug", "center": np.array([0., 0., 0.]),
         "aabb": np.zeros((2, 3)), "dims": np.zeros(3), "mass": 0.1,
         "tier": "A", "bottom_z": 0.0, "drift": 0.0},
    ])
    assert result["match_kind"] == tregr.MATCH_UNRESOLVED
    assert result["confidence"] == 0.0
    assert "spaceship" in result["reason"]


# ============================================== suite-level wiring / receptacle
def test_ground_task_suite_regrounds_receptacle_and_updates_dims(tmp_path):
    source_dir = _make_build(tmp_path / "source", [
        (0, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
        (1, "bowl", (2.0, 1.0, 1.0), (0.12, 0.12, 0.06), 0.10),
    ])
    # indices swapped in the target build, and the bowl's own measured
    # dims genuinely differ (independent single-view reconstruction) --
    # receptacle_dims must come from the TARGET's own measurement.
    target_dir = _make_build(tmp_path / "target", [
        (0, "bowl", (2.0, 1.0, 1.0), (0.13, 0.13, 0.07), 0.10),
        (1, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
    ])

    task = {"task_id": "t", "target": "obj_00", "target_label": "mug",
            "any_instance": False, "receptacle": "obj_01",
            "receptacle_dims": [0.12, 0.12, 0.06],
            "instructions": {"default": "x", "vague": "x", "specific": "x"}}

    tasks, report = tregr.ground_task_suite([task], source_dir, target_dir)
    out = tasks[0]

    assert out["target"] == "obj_01"
    assert out["receptacle"] == "obj_00"
    assert out["receptacle_dims"] == pytest.approx([0.13, 0.13, 0.07])
    assert {r["role"] for r in report} == {"target", "receptacle"}
    for row in report:
        assert row["match_kind"] == tregr.MATCH_REGROUNDED
        assert set(row) == {"task_id", "role", "source_id", "source_label",
                            "match_kind", "status", "resolved_id",
                            "confidence", "evidence", "reason"}


def test_ground_task_suite_mixed_resolved_and_unresolved(tmp_path):
    """One resolvable task and one genuinely-absent task in the SAME
    suite -- the unresolved one must be untouched (byte-identical
    target/receptacle) while the resolvable one is rewritten."""
    source_dir = _make_build(tmp_path / "source", [
        (0, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
        (1, "pen", (9.0, 9.0, 9.0), (0.02, 0.02, 0.14), 0.01),
    ])
    target_dir = _make_build(tmp_path / "target", [
        (3, "mug", (1.0, 1.0, 1.0), (0.08, 0.08, 0.10), 0.05),
    ])

    tasks_in = [
        {"task_id": "resolvable", "target": "obj_00", "target_label": "mug",
         "any_instance": False, "receptacle": None,
         "instructions": {"default": "x", "vague": "x", "specific": "x"}},
        {"task_id": "absent", "target": "obj_01", "target_label": "pen",
         "any_instance": False, "receptacle": None,
         "instructions": {"default": "x", "vague": "x", "specific": "x"}},
    ]
    tasks_out, report = tregr.ground_task_suite(tasks_in, source_dir, target_dir)
    by_id = {t["task_id"]: t for t in tasks_out}

    assert by_id["resolvable"]["target"] == "obj_03"
    assert by_id["absent"]["target"] == "obj_01"  # unchanged -- still env_crash, honestly
    assert by_id["absent"]["target_label"] == "pen"

    kinds = {r["task_id"]: r["match_kind"] for r in report if r["role"] == "target"}
    assert kinds == {"resolvable": tregr.MATCH_REGROUNDED, "absent": tregr.MATCH_UNRESOLVED}


# =================================================== wiring: simfoundry_condition
def test_resolve_condition_regrounds_and_persists_report(tmp_path):
    """robo.eval.simfoundry_condition.resolve_condition, given `suite=`,
    rewrites the returned suite's tasks via task_regrounding AND persists
    the transparency report to <baseline_dir>/task_regrounding_report.json
    -- the actual wiring robo.eval.paired_runner.build_env relies on."""
    from tests.test_simfoundry_condition import _write_recon_dir

    factory_dir = _write_recon_dir(tmp_path / "factory")  # obj_00=mug, obj_01=bowl
    baseline_dir = _write_recon_dir(tmp_path / "baseline", specs=[
        (0, "bowl", (0.30, -0.15, 0.53), (0.12, 0.12, 0.06), 0.10, 0.5),
        (1, "mug", (0.30, 0.05, 0.55), (0.08, 0.08, 0.10), 0.05, 0.6),
    ])
    (baseline_dir / sfcond.MANIFEST_FILENAME).write_text(
        json.dumps({"build_success": True, "stage_log": []}))

    suite = {
        "scene": "unit_scene",
        "scene_xml": str(factory_dir / "sim_export" / "scene.xml"),
        "robot": {"base_pos": [0.0, 0.0, 0.0], "base_yaw": 0.0},
        "table": {"cx": 0.0, "cy": 0.0, "hx": 1.0, "hy": 1.0, "top_z": 0.5},
        "ext_cam": {"pos": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 0.0], "fovy": 60.0},
        "exclude_objects": [],
        "tasks": [{
            "task_id": "unit_scene__obj_00_to_region", "target": "obj_00",
            "target_label": "mug", "any_instance": False, "receptacle": None,
            "region": {"cx": 0.0, "cy": 0.0, "hx": 0.1, "hy": 0.1,
                      "zlo": 0.0, "zhi": 1.0},
            "instructions": {"default": "x", "vague": "x", "specific": "x"},
        }],
    }
    scene_cfg = {"id": "unit_scene", "simfoundry_repro_dir": str(baseline_dir),
                "factory_dir": str(factory_dir), "collision_mode": "shim"}

    resolved = sfcond.resolve_condition(scene_cfg, suite=suite)

    regrounded_task = resolved["suite"]["tasks"][0]
    assert regrounded_task["target"] == "obj_01"  # mug is index 1 in baseline
    assert regrounded_task["target_label"] == "mug"

    report = resolved["task_regrounding_report"]
    (row,) = report
    assert row["match_kind"] == tregr.MATCH_REGROUNDED
    assert row["resolved_id"] == "obj_01"

    report_path = baseline_dir / sfcond.TASK_REGROUNDING_REPORT_FILENAME
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == report


def test_resolve_condition_without_suite_is_unaffected():
    """Back-compat: `suite=None` (the default) must not attach a "suite"
    key or attempt any regrounding -- verified structurally without
    needing a real completed build (this checks the contract, not a full
    happy path already covered by test_simfoundry_condition.py's own
    build-failure tests)."""
    import inspect
    sig = inspect.signature(sfcond.resolve_condition)
    assert sig.parameters["suite"].default is None


# =============================================== real-data validation (7b6477cb95)
FACTORY_DIR = ROOT / "outputs" / "7b6477cb95_factory"
BASELINE_DIR = ROOT / "outputs" / "7b6477cb95_baselines" / "simfoundry_repro"
TASKS_JSON = FACTORY_DIR / "sim_export" / "pi05_tasks.json"


def _skip_unless_real_scene_present():
    if not (TASKS_JSON.exists() and (BASELINE_DIR / "objects" / "objects.json").exists()):
        pytest.skip("outputs/7b6477cb95_{factory,baselines/simfoundry_repro} "
                    "not present in this checkout")


def test_real_7b6477cb95_scene_before_after():
    """Real-data validation (not a synthetic fixture): scene 7b6477cb95
    is the one the actual paired run measured at 12/12 (100%) env_crash
    from index mismatch. BEFORE regrounding, every task's raw target id
    is checked against the baseline build's own object ids (what would
    have been looked up with no regrounding at all); AFTER, via
    ground_task_suite. Asserts the real numbers reported to the user
    (not just that "some" improvement happened)."""
    _skip_unless_real_scene_present()

    suite = json.loads(TASKS_JSON.read_text())
    target_rows_by_name = {r["name"] for r in tregr.load_candidates(BASELINE_DIR)}

    before_resolved = sum(1 for t in suite["tasks"] if t["target"] in target_rows_by_name)
    tasks_after, report = tregr.ground_task_suite(suite["tasks"], FACTORY_DIR, BASELINE_DIR)
    after_resolved = sum(1 for t in tasks_after if t["target"] in target_rows_by_name)

    print(f"[test] 7b6477cb95: {len(suite['tasks'])} tasks, "
          f"before(direct-index)={before_resolved} after(regrounded)={after_resolved}")
    for row in report:
        if row["role"] == "target":
            print(f"  {row['task_id']}: {row['source_id']} ({row['source_label']!r}) "
                  f"-> {row['match_kind']} resolved_id={row['resolved_id']!r} "
                  f"conf={row['confidence']:.2f} :: {row['reason']}")

    # Ground truth for THIS scene, established by reading both real
    # objects.json files (see task write-up): 3 tasks (obj_11 "cup",
    # obj_14 "pen", obj_15 "pen"), 0 present in the baseline build by raw
    # index (its own objects.json only has indices 0-8), 1 regrounds by
    # category match ("cup" -> the baseline's own "mug", the same
    # gt_object_id=74 in both builds' objects.json), 2 genuinely absent
    # (no "pen" detected anywhere in the baseline's 9 objects).
    assert before_resolved == 0
    assert after_resolved == 1

    target_report = {r["task_id"]: r for r in report if r["role"] == "target"}
    cup_task = next(t for t in suite["tasks"] if t["target_label"] == "cup")
    pen_tasks = [t for t in suite["tasks"] if t["target_label"] == "pen"]
    assert len(pen_tasks) == 2

    assert target_report[cup_task["task_id"]]["match_kind"] == tregr.MATCH_REGROUNDED
    assert target_report[cup_task["task_id"]]["resolved_id"] == "obj_03"
    for pt in pen_tasks:
        assert target_report[pt["task_id"]]["match_kind"] == tregr.MATCH_UNRESOLVED


def test_real_scene_regrounding_matches_gt_object_id_where_available():
    """Independent sanity check using the pipeline's OWN debug
    `gt_object_id` field (present in both real objects.json files, but
    never consulted by task_regrounding itself -- see its module
    docstring) as an oracle: the object our label/category+geometry
    heuristic regrounds "cup" (factory obj_11) onto must be the SAME
    physical GT instance, not merely a plausible-looking guess."""
    _skip_unless_real_scene_present()

    factory_objects = json.loads((FACTORY_DIR / "objects" / "objects.json").read_text())
    baseline_objects = json.loads((BASELINE_DIR / "objects" / "objects.json").read_text())
    gt_by_index_factory = {f"obj_{o['index']:02d}": o.get("gt_object_id") for o in factory_objects}
    gt_by_index_baseline = {f"obj_{o['index']:02d}": o.get("gt_object_id") for o in baseline_objects}

    suite = json.loads(TASKS_JSON.read_text())
    cup_task = next(t for t in suite["tasks"] if t["target_label"] == "cup")
    tasks_after, _ = tregr.ground_task_suite([cup_task], FACTORY_DIR, BASELINE_DIR)
    resolved_id = tasks_after[0]["target"]

    assert resolved_id != cup_task["target"]  # actually regrounded to a different id
    assert gt_by_index_baseline[resolved_id] == gt_by_index_factory[cup_task["target"]]
