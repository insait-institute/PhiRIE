import hashlib
import json
from pathlib import Path

import pytest

from robo.manifest.hash import canonical_hash
from robo.roundtrip.milestone_report import package, sha, FIELDS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    pilot, build, comparison = [tmp_path / name for name in ("pilot", "build", "comparison")]
    ref_replay = tmp_path / "control/replay/REF_NATIVE_seed0"
    config = {"reset_seeds": [0, 1, 2, 3, 4], "horizon": 600}
    ch = canonical_hash(config)
    policy = {"checkpoint": "frozen", "normalization": "frozen", "rng": "paired"}
    def episode(directory, seed, treatment, success, kind="closed_loop_visual_policy", identity=None):
        directory.mkdir(parents=True)
        (directory / "initial_state.json").write_text("{}")
        (directory / "actions.json").write_text("[[0,1],[1,0]]")
        (directory / "trace.json.gz").write_bytes(b"synthetic trace fixture")
        (directory / "continuous.mp4").write_bytes(b"synthetic video fixture " + treatment.encode())
        result = {"episode_id": f"{treatment}:{seed}:{kind}", "treatment_id": treatment, "reset_seed": seed,
            "task_id": "PickPlaceCounterToSink", "execution_kind": kind, "success": success,
            "executed": True, "error": None, "ticks": 2, "horizon": 600, "wall_s": .2,
            "config_sha256": ch, "policy_identity": identity or policy,
            "video_path": str(directory / "continuous.mp4"), "video_error": None}
        write(directory / "result.json", result)
        write(directory / "harness_ledger.jsonl", result)
        return result
    native = [episode(pilot / f"episode_seed{i}", i, "REF_NATIVE", i != 3) for i in range(5)]
    write(pilot / "run_manifest.json", {"config": config, "config_sha256": ch, "code": {"commit": "a"*40}})
    write(pilot / "pilot_results.json", native)
    ident = tmp_path / "identity/identity_report.json"
    write(ident, {"passed": True, "initial_objects_equal": True, "initial_observation_diff": {},
        "u1_initial_observation_diff": {}, "u1_import": {"body_xml_equal": True},
        "u0": [{"observation_diff": {}, "state_max_abs": 0, "predicate_equal": True}],
        "u1": [{"observation_diff": {}, "state_max_abs": 0, "predicate_equal": True}]})
    bc = {"source_commit": "b"*40}
    write(build / "build_config.json", bc)
    (build / "target.obj").write_text("synthetic mesh")
    write(build / "build_manifest.json", {"config_sha256": hashlib.sha256(json.dumps(bc, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_commit": "b"*40, "source_hashes": {"target.obj": sha(build / "target.obj")}})
    ref = pilot / "episode_seed0"
    canonical = pilot / "canonical_seed0"
    write(canonical / "canonical_state.json", {})
    (canonical / "scene.xml").write_text("<mujoco/>")
    pair_dir = comparison / "paired/FIXED_NATIVE_seed0"
    fixed = episode(pair_dir / "episode", 0, "FIXED_NATIVE", True)
    reference = {"reference_episode": str(ref), "canonical_reference": str(canonical), "reset_seed": 0,
        "config_sha256": ch, "reference_files": {name: sha(ref / name) for name in ("result.json", "actions.json", "initial_state.json")},
        "canonical_files": {name: sha(canonical / name) for name in ("canonical_state.json", "scene.xml")}}
    write(pair_dir / "pair_receipt.json", {"state": "RECORDED", "planned_comparison_episodes": 1,
        "reference": reference, "comparison_result_sha256": sha(pair_dir / "episode/result.json"),
        "policy_identity": policy, "source_code": {"commit": "c"*40}})
    write(pair_dir / "import_receipt.json", {"source_hashes": {str(build / "target.obj"): sha(build / "target.obj")}})
    replay_dirs = [comparison / "replay/FIXED_NATIVE_seed0", ref_replay]
    replay_results = []
    for directory, success in zip(replay_dirs, [False, True]):
        result = episode(directory / "episode", 0, "REPLAY", success, "fixed_action_replay",
            {"object_pose_playback": False, "source": {"reference_result_sha256": sha(ref / "result.json"), "actions_sha256": sha(ref / "actions.json")}})
        metric = {"absolute_frame_correspondence": "NOT_ESTABLISHED", "position_rmse_cm": None,
            "rotation_error_deg": None, "final_position_error_cm": None, "relative_displacement_rmse_cm": 6.5 if not success else 0,
            "relative_rotation_increment_mean_deg": 23 if not success else 0,
            "relative_full_horizon_final_displacement_error_cm": 11 if not success else 0,
            "native_replay_outcome": result, "planned_action_steps": 2, "comparison_recorded_steps": 2,
            "matched_steps": 2, "matched_step_coverage": 1, "complete_trace": True,
            "provenance": {key: {name: sha(source / name) for name in ("result.json", "actions.json", "trace.json.gz")}
                           for key, source in (("reference_files", ref), ("comparison_files", directory / "episode"))}}
        write(directory / "replay_metrics.json", metric)
        replay_results.append(result)
    write(ref_replay.parent.parent / "full_action_identity.json", {"planned_steps": 2, "reference_steps": 2, "replay_steps": 2,
        "all_actions_equal": True, "all_qpos_equal": True, "all_qvel_equal": True,
        "all_observation_hashes_equal": True, "all_native_predicates_equal": True})
    diagnosis = tmp_path / "diagnosis.json"
    write(diagnosis, {"source_commit": "c"*40, "rows": [
        {"episode": str(directory), "source_trace_sha256": sha(directory / "trace.json.gz"),
         "logged_native_success": result["success"], "recomputed_native_success": result["success"],
         "treatment": result["treatment_id"], "target_origin_inside_sink": True,
         "gripper_to_target_origin_m": .26 if result["success"] else .24,
         "native_retreat_threshold_m": .25, "gripper_far": result["success"]}
        for directory, result in ((ref, native[0]), (pair_dir / "episode", fixed), (replay_dirs[0] / "episode", replay_results[0]))]})
    return dict(pilot=pilot, identity=ident, build=build, comparison=comparison,
                reference_replay=ref_replay, diagnosis=diagnosis, out=tmp_path / "report")


def test_preserves_failure_and_copies_video_without_reencoding(inputs):
    report = package(**inputs, include_replay_video=True)
    assert report["counts"][0] == {"population": "native_pilot", "planned": 5, "executed": 5, "successes": 4, "rollout_coverage": 1}
    assert len(report["native_pilot_outcomes"]) == 5
    assert report["native_pilot_outcomes"][3]["success"] is False
    assert len(report["episodes"]) == 8
    assert all(set(row) == set(FIELDS) for row in report["episodes"])
    assert report["episodes"][-2]["position_rmse_cm"] is None
    assert report["episodes"][-2]["relative_displacement_rmse_cm"] == 6.5
    for video in report["videos"]:
        assert (inputs["out"] / video["name"]).read_bytes() == Path(video["source"]).read_bytes()
    with pytest.raises(FileExistsError):
        package(**inputs)


@pytest.mark.parametrize("bad", ["pilot_drop", "policy", "action", "absolute", "identity", "diagnosis", "full_control"])
def test_rejects_broken_source_or_comparison_contract(inputs, bad):
    if bad == "pilot_drop":
        path = inputs["pilot"] / "pilot_results.json"; value = json.loads(path.read_text()); value.pop(); write(path, value)
    elif bad == "policy":
        path = inputs["comparison"] / "paired/FIXED_NATIVE_seed0/episode/result.json"
        value = json.loads(path.read_text()); value["policy_identity"] = {"checkpoint": "changed"}; write(path, value)
    elif bad == "action":
        (inputs["comparison"] / "replay/FIXED_NATIVE_seed0/episode/actions.json").write_text("[[99]]")
    elif bad == "absolute":
        path = inputs["comparison"] / "replay/FIXED_NATIVE_seed0/replay_metrics.json"
        value = json.loads(path.read_text()); value["position_rmse_cm"] = 0; write(path, value)
    elif bad == "identity":
        value = json.loads(inputs["identity"].read_text()); value["u0"][0]["state_max_abs"] = .1; write(inputs["identity"], value)
    elif bad == "diagnosis":
        value = json.loads(inputs["diagnosis"].read_text()); value["rows"][-1]["recomputed_native_success"] = True; write(inputs["diagnosis"], value)
    else:
        path = inputs["reference_replay"].parent.parent / "full_action_identity.json"
        value = json.loads(path.read_text()); value["all_qpos_equal"] = False; write(path, value)
    with pytest.raises(ValueError):
        package(**inputs)
    assert not inputs["out"].exists()
