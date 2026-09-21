from types import SimpleNamespace
import copy
import pytest

from robo.eval.e4_reset_eligibility import index_reset_eligibility
from robo.eval.harness_runner import _episode_jitter


def test_wrong_cpu_gate_hash_aborts_before_environment_or_failure_rows(tmp_path):
    from robo.eval import e4_candidate_screen as screen
    from robo.eval.e4_reset_eligibility import load_reset_eligibility
    bundle = tmp_path / "sealed"
    screen._publish_bundle(bundle,
        manifest_kind="e4_one_target_winner_smoke_artifacts",
        payloads={"gate.json": b"{}\n"}, manifest_fields={})
    config = {"policy": "scripted_sinusoid", "paper_mode": False,
              "jitter_first_episode": True, "jitter": screen.JITTER_XY_M,
              "cpu_reset_eligibility": {"bundle": str(bundle), "gate_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="hash differs"):
        load_reset_eligibility(config, None, [], root=tmp_path)


def _fixture():
    state = SimpleNamespace(scene_id="scene", task_id="task", ep=0,
                            reset_seed=42, reset_state_id="task__seed0__ep0")
    rows = [{"policy_id": p, "scene_id": "scene", "task_id": "task",
             "episode": 0, "reset_seed": 42, "cell_id": p,
             "passed": p == "A4", "checks": {"stability_passed": p == "A4"}}
            for p in ("A0", "A4")]
    declaration = dict(gate_sha256="a" * 64, source_commit="b" * 40, bundle="bundle")
    return rows, [state], {"fixed_single_path": "a0", "agentic": "a4"}, declaration


def test_both_arms_remain_planned_with_typed_failure_evidence():
    args = _fixture()
    indexed = index_reset_eligibility(*args)
    assert len(indexed) == 2
    assert indexed[("a0", args[1][0].reset_state_id)]["failure_type"] == "cpu_reset_validity_failure"
    assert indexed[("a4", args[1][0].reset_state_id)]["passed"]


@pytest.mark.parametrize("change", ["missing", "unknown", "duplicate"])
def test_missing_unknown_duplicate_resets_abort(change):
    rows, states, mapping, declaration = _fixture()
    if change == "missing":
        rows.pop()
    elif change == "unknown":
        rows[1]["reset_seed"] = 43
    else:
        rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError, match="reset"):
        index_reset_eligibility(rows, states, mapping, declaration)


def test_first_reset_jitter_is_explicit_and_legacy_default_unchanged():
    state = SimpleNamespace(ep=0)
    assert _episode_jitter({"jitter": .01}, state) == 0
    assert _episode_jitter({"jitter": .01, "jitter_first_episode": True}, state) == .01


@pytest.mark.parametrize("changed", ["task-bundle", "menagerie"])
def test_resealed_dependencies_must_match_cpu_evidence(tmp_path, monkeypatch, changed):
    import json
    from robo.eval import e4_candidate_screen as screen
    from robo.eval import e4_camera_scorer_gate as camera
    from robo.eval import e4_task_freeze
    from robo.eval.e4_reset_eligibility import load_reset_eligibility
    code = {"commit": "a" * 40, "dirty": False}
    gate = {"code": code, "study_scope": "one_target_engineering_smoke",
            "task_bundle": {"bundle_manifest_sha256": "b" * 64},
            "menagerie": {"sha256": "c" * 64}}
    bundle = tmp_path / "sealed"
    screen._publish_bundle(bundle, manifest_kind="e4_one_target_winner_smoke_artifacts",
        payloads={"gate.json": json.dumps(gate).encode()}, manifest_fields={})
    monkeypatch.setattr(screen, "_code_snapshot", lambda commit: code)
    monkeypatch.setattr(e4_task_freeze, "validate_task_bundle", lambda *a, **kw: {
        "bundle_manifest_sha256": ("x" if changed == "task-bundle" else "b") * 64})
    monkeypatch.setattr(camera, "_menagerie_snapshot", lambda *a, **kw: {
        "sha256": "y" * 64, "files": []})
    config = {"policy": "scripted_sinusoid", "jitter_first_episode": True,
              "jitter": screen.JITTER_XY_M,
              "scenes": [{"id": "scene", "task_freeze_manifest": "unused"}],
              "cpu_reset_eligibility": {"bundle": str(bundle),
                  "gate_sha256": screen._sha256(bundle / "gate.json"),
                  "source_commit": code["commit"]}}
    with pytest.raises(ValueError, match=changed + " digest differs"):
        load_reset_eligibility(config, None, [], root=tmp_path)


def test_prebuild_certificate_authenticates_cell_and_planned_jitter(tmp_path):
    import json
    from robo.eval import e4_candidate_screen as screen
    from robo.eval.e4_reset_eligibility import certify_prebuild_failure
    bundle = tmp_path / "cpu"
    jitter = {"offset_xy_m": [0.001, 0.002]}
    row = {"cell_id": "a0__task__seed0__ep0", "scene_id": "scene", "task_id": "task",
           "policy_id": "A0", "episode": 0, "reset_seed": 42,
           "reset_jitter": jitter, "passed": False, "checks": {"stability_passed": False}}
    gate = {"code": {"commit": "a" * 40, "dirty": False},
            "task_bundle": {"bundle_manifest_sha256": "b" * 64}}
    screen._publish_bundle(bundle, manifest_kind="e4_one_target_winner_smoke_artifacts",
        payloads={"gate.json": json.dumps(gate).encode(),
                  "metrics.jsonl": (json.dumps(row) + "\n").encode()}, manifest_fields={})
    declaration = {"bundle": str(bundle), "gate_sha256": screen._sha256(bundle / "gate.json"),
                   "source_commit": "a" * 40}
    a0 = {"id": "a0", "scene": "fixed_single_path"}
    a4 = {"id": "a4", "scene": "agentic"}
    spec = SimpleNamespace(raw={"cpu_reset_eligibility": declaration, "policy": "scripted_sinusoid"},
        treatments={"a0": SimpleNamespace(scene="fixed_single_path", to_dict=lambda:a0),
                    "a4": SimpleNamespace(scene="agentic", to_dict=lambda:a4)},
        comparisons=[], treatment_ids=lambda:["a0", "a4"])
    record = {"outcome": "build_failure", "treatment_id": "a0", "treatment": a0,
              "scene_id": "scene", "task_id": "task", "reset_seed": 42,
              "reset_state_id": "task__seed0__ep0",
              "construction_validity_evidence": {**declaration, "failure_type": "cpu_reset_validity_failure",
                  "passed": False, "failed_checks": ["stability_passed"], "cpu_cell_id": row["cell_id"]},
              "contract": {"reset_definition": {"reset_state_id": "task__seed0__ep0",
                  "reset_seed": 42, "episode_index": 0, "base_seed": 0, "jitter": copy.deepcopy(jitter)},
                  "construction_artifacts": {"task_freeze_manifest_sha256": "b" * 64}}}
    assert certify_prebuild_failure(record, spec, root=tmp_path)
    wrong_arm = copy.deepcopy(record)
    wrong_arm["treatment_id"] = "a4"
    with pytest.raises(ValueError, match="declared arm"):
        certify_prebuild_failure(wrong_arm, spec, root=tmp_path)
    wrong_arm["treatment"] = a4
    with pytest.raises(ValueError, match="measured CPU cell"):
        certify_prebuild_failure(wrong_arm, spec, root=tmp_path)
    from robo.eval.harness_validation import validate_saved_treatment_records
    telemetry = copy.deepcopy(record)
    telemetry["reset_provenance"] = {"malformed": "must not be discarded"}
    result = validate_saved_treatment_records([telemetry], spec,
        {"task__seed0__ep0"}, manifest_root=tmp_path)
    assert not result["ok"]
    assert any("must not contain measured reset" in v for v in result["violations"])
    assert telemetry["reset_provenance"] == {"malformed": "must not be discarded"}
    record["contract"]["reset_definition"]["jitter"]["offset_xy_m"][0] += .01
    with pytest.raises(ValueError, match="planned reset"):
        certify_prebuild_failure(record, spec, root=tmp_path)
