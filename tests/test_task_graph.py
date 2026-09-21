"""Tests for robo/certification/task_graph.py and grounding.py (Task 11).

Fixture scope note (see final task report): the plan's real acceptance
criterion asks for hand-labeled graphs over "at least ten tasks." This file
is the unit-test pass against synthetic fixtures -- five tests/data/tasks/
*.yaml manifests over one shared synthetic scene
(tests/data/scenes/task_graph_fixture/scene.yaml), each isolating a
different role-resolution evidence path (label+language+geometry-anchor,
explicit spatial-hint tie-break, explicit benchmark_mapping override,
isolated receptacle ambiguity, and total missing-evidence ambiguity).
Scaling to ten *real* scene-graph-tested tasks is a follow-up once a real
build's objects.json is converted into this module's scene.yaml schema (no
such converter exists yet -- see configs/task_graph/pick_and_place_example.yaml).
"""
import json
from pathlib import Path

import pytest

from robo.certification import grounding as G
from robo.certification import task_graph as TG

ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "tests" / "data" / "scenes" / "task_graph_fixture"
TASK_DIR = ROOT / "tests" / "data" / "tasks"

# Every non-eligible object in the shared scene fixture: too far for the
# reach envelope and outside every camera frustum, never referenced by any
# task's rubric.role_refs. Must never appear anywhere in any built graph.
FAR_AWAY_IDS = {"obj_09", "obj_06"}


def _scene():
    return TG.load_scene(SCENE_DIR)


def _task(name):
    return TG.load_task(TASK_DIR / f"{name}.yaml")


def _graph(task_name):
    return TG.build_graph(_scene(), _task(task_name))


def _object_node_ids(graph):
    return {n["node_id"] for n in graph["nodes"] if n["kind"] == "object"}


def _assert_far_away_excluded(graph):
    ids = _object_node_ids(graph)
    assert ids.isdisjoint(FAR_AWAY_IDS), (
        f"distant/irrelevant objects leaked into the graph: {ids & FAR_AWAY_IDS}")
    for role, res in graph["role_resolutions"].items():
        hyp_ids = {h["object_id"] for h in res["hypotheses"]}
        assert hyp_ids.isdisjoint(FAR_AWAY_IDS), (
            f"distant object became a '{role}' hypothesis: {hyp_ids & FAR_AWAY_IDS}")


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def food_bussing_graph():
    return _graph("food_bussing")


# ------------------------------------------------------------- basic shape
def test_scene_and_task_fixtures_load():
    scene = _scene()
    assert scene["scene_id"] == "task_graph_fixture"
    assert len(scene["objects"]) == 8
    task = _task("food_bussing")
    assert task["task_id"] == "food_bussing_demo"


def test_graph_has_robot_and_camera_nodes(food_bussing_graph):
    kinds = {n["kind"] for n in food_bussing_graph["nodes"]}
    assert kinds == {"robot", "camera", "object"}
    cam_ids = {n["node_id"] for n in food_bussing_graph["nodes"] if n["kind"] == "camera"}
    assert cam_ids == {"camera:ext_cam", "camera:wrist_cam"}


def test_graph_json_roundtrips(food_bussing_graph):
    dumped = TG.to_json(food_bussing_graph)
    reloaded = json.loads(dumped)
    assert reloaded == food_bussing_graph


# --------------------------------------------------------- role resolution
def test_manipulated_object_resolves_to_evidenced_mug_not_distractor(food_bussing_graph):
    """food_bussing.yaml: obj_01 (on the placemat, near the tray) is the
    correct manipulated_object; obj_02 is a same-label ("mug") distractor
    with no rubric/geometry evidence beyond the label match. Distractor
    rejection must come from geometry-nearest-to-receptacle evidence, not an
    arbitrary tie-break."""
    res = food_bussing_graph["role_resolutions"]["manipulated_object"]
    assert res["status"] == "resolved"
    assert res["hypotheses"][0]["object_id"] == "obj_01"

    by_id = {h["object_id"]: h for h in res["hypotheses"]}
    assert "obj_02" in by_id, "the distractor should still be visible as a candidate"
    assert by_id["obj_02"]["object_id"] != res["hypotheses"][0]["object_id"]
    # obj_02 only ever accrues label/language/category evidence, never a
    # geometry bonus (it is the farthest same-label candidate from the
    # receptacle anchor) -- assert that directly, not just the final rank.
    assert G.EV_GEOMETRY_NEAREST not in by_id["obj_02"]["evidence"]
    assert G.EV_GEOMETRY_NEAREST in by_id["obj_01"]["evidence"]
    # And the winning margin is decisive, not a coin flip.
    assert by_id["obj_01"]["confidence"] - by_id["obj_02"]["confidence"] >= G.AMBIGUITY_MARGIN


def test_distractor_not_selected_via_spatial_hint_path(food_bussing_graph=None):
    """A second, independent distractor-rejection path: spatial_hint_disambiguation.yaml
    requests only manipulated_object (no receptacle anchor at all) and
    disambiguates obj_01 vs. obj_02 purely via an explicit rubric
    spatial_hints.manipulated_object: nearest_to_base qualifier. Also checks
    the qualifier doesn't let obj_08 ("cup", literally the closest object to
    the base but with zero label evidence) win on proximity alone."""
    graph = _graph("spatial_hint_disambiguation")
    res = graph["role_resolutions"]["manipulated_object"]
    assert res["status"] == "resolved"
    assert res["hypotheses"][0]["object_id"] == "obj_01"
    ranked_ids = [h["object_id"] for h in res["hypotheses"]]
    assert ranked_ids.index("obj_01") < ranked_ids.index("obj_02")
    assert ranked_ids.index("obj_01") < ranked_ids.index("obj_08")


def test_explicit_benchmark_mapping_overrides_geometry_and_label():
    """benchmark_mapping.yaml pins obj_02 (the 'wrong'/distractor mug in the
    other fixtures) as manipulated_object via rubric.role_refs -- an
    explicit human/benchmark id lookup table, which plan/11 requires
    grounding to honor even when it disagrees with geometry."""
    graph = _graph("benchmark_mapping")
    manip = graph["role_resolutions"]["manipulated_object"]
    assert manip["status"] == "resolved"
    assert manip["hypotheses"][0]["object_id"] == "obj_02"
    assert G.EV_BENCHMARK_MAPPING in manip["hypotheses"][0]["evidence"]

    recep = graph["role_resolutions"]["receptacle"]
    assert recep["status"] == "resolved"
    assert recep["hypotheses"][0]["object_id"] == "obj_00"
    assert G.EV_BENCHMARK_MAPPING in recep["hypotheses"][0]["evidence"]


def test_no_evaluation_result_leakage_in_role_refs():
    """role_refs is an explicit id lookup table (human/benchmark-authored),
    not a policy-evaluation result -- grounding.LANGUAGE_ROLES resolution
    must accept it. There is structurally no eval-result field anywhere in
    the scene/task manifest schema; this test additionally asserts that
    smuggling one in (a made-up 'eval_success' key) is simply never read."""
    task = _task("benchmark_mapping")
    task["eval_success"] = {"obj_02": 1.0, "obj_00": 0.0}  # not a real field; must be ignored
    scene = _scene()
    graph_with_bogus_field = TG.build_graph(scene, task)
    task.pop("eval_success")
    graph_without = TG.build_graph(scene, task)
    assert TG.to_json(graph_with_bogus_field) == TG.to_json(graph_without)


# ------------------------------------------------------------ eligibility
def test_far_away_objects_excluded_from_every_fixture():
    """Negative test (plan/11): 'a distant/irrelevant room object must be
    excluded unless it intersects camera view or swept workspace.' obj_09 is
    a same-*label* ("mug") distant distractor, distinct from the *nearby*
    distractor obj_02 covered above; obj_06 is a generically distant, unlike-
    category piece of furniture. Checked across every task fixture, since
    eligibility is task-independent geometry/camera evidence."""
    for name in ("food_bussing", "spatial_hint_disambiguation", "benchmark_mapping",
                 "ambiguous_receptacle", "missing_evidence"):
        _assert_far_away_excluded(_graph(name))


def test_eligibility_gate_ignores_category_match_alone():
    """Directly exercise the eligibility gate: obj_09 shares its label
    ("mug") with the correct manipulated_object, but is neither reachable
    nor camera-visible nor referenced -- eligible_object_ids must exclude it
    regardless of what any task asks for."""
    scene = _scene()
    task = _task("food_bussing")
    eligible, _ = TG.eligible_object_ids(scene, task)
    assert "obj_09" not in eligible
    assert "obj_06" not in eligible
    assert "obj_01" in eligible and "obj_02" in eligible  # sanity: gate isn't over-pruning


# --------------------------------------------------------------- support
def test_support_edge_and_role(food_bussing_graph):
    """obj_01 (mug) rests on obj_05 (placemat): bottom-of-mug z equals
    top-of-placemat z and the mug's xy center falls inside the placemat's
    footprint. The 'support' role should surface exactly this object."""
    edges = food_bussing_graph["edges"]
    assert {"src": "obj_01", "dst": "obj_05", "kind": "support"} in [
        {"src": e["src"], "dst": e["dst"], "kind": e["kind"]} for e in edges]

    support = food_bussing_graph["role_resolutions"]["support"]
    assert support["status"] == "resolved"
    assert support["hypotheses"][0]["object_id"] == "obj_05"


def test_support_role_unresolved_when_manipulated_object_rests_on_nothing_modeled():
    """benchmark_mapping.yaml's manipulated_object is obj_02, which the
    scene fixture does not place on any other eligible object -- 'support'
    should come back with zero hypotheses (unresolved), not a guess."""
    graph = _graph("benchmark_mapping")
    support = graph["role_resolutions"]["support"]
    assert support["status"] == "unresolved"
    assert support["hypotheses"] == []


# -------------------------------------------------------------- obstacle
def test_swept_workspace_obstacle_detected(food_bussing_graph):
    """obj_08 ('cup') sits almost exactly on the base->mug reach path and
    carries no manipulated_object label evidence at all -- it should surface
    purely through swept-workspace geometry as a collision-relevant obstacle."""
    obstacle = food_bussing_graph["role_resolutions"]["obstacle"]
    assert obstacle["status"] == "resolved"
    assert obstacle["multi_select"] is True
    obstacle_ids = {h["object_id"] for h in obstacle["hypotheses"]}
    assert "obj_08" in obstacle_ids
    for h in obstacle["hypotheses"]:
        assert G.EV_SWEPT_WORKSPACE in h["evidence"]
    # The manipulated object and receptacle themselves are never also
    # reported as their own obstacle.
    assert obstacle_ids.isdisjoint({"obj_01", "obj_00"})


# ---------------------------------------------------------- ambiguity
def test_ambiguous_receptacle_flagged_not_arbitrary():
    """ambiguous_receptacle.yaml pins manipulated_object (via role_refs) but
    gives no evidence at all for receptacle, and the scene has two
    receptacle-category objects (tray, bowl). Must come back 'ambiguous'
    with both hypotheses retained, never a confident single pick."""
    graph = _graph("ambiguous_receptacle")
    manip = graph["role_resolutions"]["manipulated_object"]
    assert manip["status"] == "resolved"
    assert manip["hypotheses"][0]["object_id"] == "obj_01"

    recep = graph["role_resolutions"]["receptacle"]
    assert recep["status"] == "ambiguous"
    assert {h["object_id"] for h in recep["hypotheses"]} == {"obj_00", "obj_07"}
    assert "receptacle" in graph["ambiguous_roles"]


def test_missing_evidence_flags_both_roles_ambiguous_not_arbitrary():
    """missing_evidence.yaml has no rubric at all and language the parser
    cannot match into either phrase. Both manipulated_object and receptacle
    must be flagged 'ambiguous' -- multiple weak, undifferentiated
    hypotheses -- rather than silently returning whichever object the
    residual geometry tie-break happens to rank first."""
    graph = _graph("missing_evidence")
    for role in ("manipulated_object", "receptacle"):
        res = graph["role_resolutions"][role]
        assert res["status"] == "ambiguous"
        assert len(res["hypotheses"]) >= 2
    assert {"manipulated_object", "receptacle"} <= set(graph["ambiguous_roles"])


def test_ambiguity_status_uses_raw_not_clipped_confidence():
    """Two hypotheses can both be well above a naive [0, 1] confidence
    ceiling and still be decisively separated -- finalize_hypotheses must
    rank/decide status on the unclipped additive score, not a value that
    would make two very-differently-evidenced candidates look tied."""
    hyps = {
        "a": {"confidence": 1.42, "evidence": ["x"]},
        "b": {"confidence": 1.12, "evidence": ["y"]},
    }
    res = G.finalize_hypotheses("manipulated_object", hyps, multi_select=False)
    assert res["status"] == "resolved"
    assert res["hypotheses"][0]["object_id"] == "a"
    assert res["hypotheses"][0]["confidence"] == pytest.approx(1.42)
    assert res["hypotheses"][1]["confidence"] == pytest.approx(1.12)


# ------------------------------------------------------------- determinism
def test_graph_is_deterministic_across_repeated_builds():
    """plan/11 acceptance criterion: 'graph is deterministic under a frozen
    scene/task manifest.' Rebuild from freshly-loaded manifests (not just a
    cached dict) twice and require byte-identical JSON."""
    for name in ("food_bussing", "ambiguous_receptacle", "missing_evidence"):
        g1 = TG.to_json(TG.build_graph(_scene(), _task(name)))
        g2 = TG.to_json(TG.build_graph(_scene(), _task(name)))
        assert g1 == g2


def test_graph_is_deterministic_across_process_invocations(tmp_path):
    """Same check, but through the actual CLI entry point (subprocess), to
    catch any accidental reliance on in-process state (e.g. leftover
    grounding.configure() overrides) that a pure-python call wouldn't."""
    import subprocess
    import sys

    outs = []
    for i in range(2):
        out_dir = tmp_path / f"run_{i}"
        proc = subprocess.run(
            [sys.executable, "-m", "robo.certification.task_graph",
             "--task", str(TASK_DIR / "food_bussing.yaml"),
             "--scene", str(SCENE_DIR),
             "--out-dir", str(out_dir), "--quiet"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outs.append((out_dir / "task_graph.json").read_text())
    assert outs[0] == outs[1]


# --------------------------------------------------------- config loading
def test_config_overrides_grounding_weights_and_thresholds():
    """configs/task_graph/default_thresholds.yaml should load and apply
    without error, and (since it's frozen to match the hardcoded defaults)
    must reproduce the identical graph -- proving --config is wired all the
    way through without silently being ignored would require a *different*
    config; this test additionally exercises that a genuinely different
    weight set changes the outcome."""
    cfg_path = ROOT / "configs" / "task_graph" / "default_thresholds.yaml"
    cfg = TG.load_config(cfg_path)
    assert cfg["grounding_weights"]["W_BENCHMARK_MAPPING"] == pytest.approx(G.W_BENCHMARK_MAPPING)

    previous = G.configure(cfg["grounding_weights"])
    try:
        graph_default_cfg = TG.build_graph(_scene(), _task("food_bussing"))
    finally:
        G.configure(previous)
    graph_hardcoded = TG.build_graph(_scene(), _task("food_bussing"))
    assert TG.to_json(graph_default_cfg) == TG.to_json(graph_hardcoded)

    # A deliberately different weight set (zero out the geometry tie-break
    # bonus) changes the manipulated_object outcome for food_bussing.yaml:
    # without it, obj_01 and obj_02 tie on label+language+category alone.
    previous = G.configure({"W_GEOMETRY_BONUS": 0.0})
    try:
        graph_no_geo_bonus = TG.build_graph(_scene(), _task("food_bussing"))
    finally:
        G.configure(previous)
    res = graph_no_geo_bonus["role_resolutions"]["manipulated_object"]
    assert res["status"] == "ambiguous"


# --------------------------------------------------------- accuracy report
def test_role_accuracy_report_mechanism():
    """plan/11 acceptance criterion: 'role accuracy and ambiguity rate are
    reported.' Build the hand-labeled expectation for every fixture above
    and run grounding.role_accuracy_report over them -- this is the
    reporting mechanism itself under test, not just its inputs."""
    cases = [
        {"role_resolutions": _graph("food_bussing")["role_resolutions"],
         "expected": {"manipulated_object": "obj_01", "receptacle": "obj_00"}},
        {"role_resolutions": _graph("spatial_hint_disambiguation")["role_resolutions"],
         "expected": {"manipulated_object": "obj_01"}},
        {"role_resolutions": _graph("benchmark_mapping")["role_resolutions"],
         "expected": {"manipulated_object": "obj_02", "receptacle": "obj_00"}},
        {"role_resolutions": _graph("ambiguous_receptacle")["role_resolutions"],
         "expected": {"manipulated_object": "obj_01", "receptacle": None}},
        {"role_resolutions": _graph("missing_evidence")["role_resolutions"],
         "expected": {"manipulated_object": None, "receptacle": None}},
    ]
    report = G.role_accuracy_report(cases)

    assert report["n_cases"] == 9  # 2+1+2+2+2 expected-role assertions across the 5 fixtures
    assert report["role_accuracy"] == pytest.approx(1.0), (
        "every hand-labeled expectation (including the two deliberately "
        "'correctly ambiguous' ones) should be met exactly")
    assert 0.0 < report["ambiguity_rate"] < 1.0
    assert report["per_role"]["manipulated_object"]["n"] == 5
    assert report["per_role"]["receptacle"]["n"] == 4

    # Persist the report next to the other runtime task_graph artifacts, as
    # the concrete "role accuracy / ambiguity rate are reported" artifact
    # (plan/11 acceptance criterion) -- not required for the assertions
    # above, but this is the mechanism a reviewer/CI step would read.
    report_dir = ROOT / "outputs" / "task_graph_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "role_accuracy_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))


# -------------------------------------------------------------- overlay
def test_save_graph_writes_json_and_overlay(tmp_path):
    graph = _graph("food_bussing")
    json_path, overlay_path = TG.save_graph(graph, tmp_path)
    assert json_path.exists()
    assert json.loads(json_path.read_text()) == graph
    assert overlay_path.exists()
    # Either the matplotlib PNG or the text-fallback overlay is acceptable
    # (plan/11 explicitly allows the text fallback as a documented trade-off).
    assert overlay_path.suffix in (".png", ".txt")
