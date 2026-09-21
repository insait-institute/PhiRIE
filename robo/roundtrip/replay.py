"""Canonical native robot-action replay and explicitly relative pose diagnostics.

Body origins need not correspond between native and reconstructed objects.
Absolute pose errors remain null. A single public TRAIN marker is attached in
each initial body frame; no object state is written after the initial import.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _action_array(actions) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 12 or len(result) == 0:
        raise ValueError("replay requires a nonempty N x 12 native robot action array")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite reference action")
    return np.ascontiguousarray(result)


class FixedActionPolicy:
    """A replay source, never a learned visual-policy episode."""
    is_visual_policy = False

    def __init__(self, actions, *, source_identity: dict):
        self._actions = _action_array(actions).copy()
        self._source_identity = copy.deepcopy(source_identity)
        self._index = None

    @property
    def metadata(self):
        return {"policy_type": "fixed_native_robot_action_replay",
                "source": self._source_identity, "planned_actions": len(self._actions),
                "action_array_sha256": hashlib.sha256(self._actions.tobytes()).hexdigest(),
                "action_dtype": "float64", "action_dimension": 12,
                "object_pose_playback": False}

    def reset(self, seed: int):
        self._index = 0

    def infer(self, observation):
        if self._index is None:
            raise RuntimeError("reset must precede replay")
        if self._index >= len(self._actions):
            raise IndexError("fixed action bank exhausted; no padding or replanning")
        action = self._actions[self._index].copy()
        self._index += 1
        return action


def run_replay_episode(adapter, *, actions_path: str | Path, config: dict,
                       reset_seed: int, out_dir: str | Path, treatment_id: str,
                       expected_actions_sha256: str, source_identity: dict | None = None):
    """Caller must complete the paired native reset/import before entry.

    Uses the existing canonical ledger/video/trace writer. A source trace shorter
    than the native policy horizon remains a declared fixed-prefix replay.
    """
    from robo.eval.harness_runner import run_native_episode
    actions_path = Path(actions_path)
    if sha256(actions_path) != expected_actions_sha256:
        raise ValueError("reference action bank content changed")
    actions = _action_array(json.loads(actions_path.read_text()))
    replay_config = copy.deepcopy(config)
    if config.get('schema_version')==2:
        if config['execution_protocol']!='full_horizon_feedback_diagnostic' or len(actions)!=config['horizon']:
            raise ValueError('v2 replay requires real reference actions for the full declared task horizon')
    else:
        replay_config["native_policy_horizon"] = config["horizon"]
        replay_config["horizon"] = len(actions)
        replay_config["replay_contract"] = {
            "source_actions_sha256": expected_actions_sha256,
            "fixed_action_count": len(actions), "early_success_termination": False,
            "native_task_success_scope": "unchanged predicate at the fixed replay prefix horizon",
        }
    identity = {**(source_identity or {}), "actions_path": str(actions_path.resolve()),
                "actions_sha256": expected_actions_sha256}
    record = run_native_episode(adapter, FixedActionPolicy(actions, source_identity=identity),
                                config=replay_config, reset_seed=reset_seed,
                                out_dir=Path(out_dir), treatment_id=treatment_id,
                                execution_kind="fixed_action_replay")
    replayed = np.asarray(json.loads((Path(out_dir)/"actions.json").read_text()), dtype=np.float64)
    if replayed.size == 0:
        replayed = replayed.reshape(0, 12)
    if replayed.shape != (record["ticks"], 12) or len(replayed) > len(actions):
        raise ValueError("canonical replay action/trace accounting differs")
    if replayed.tobytes() != actions[:len(replayed)].tobytes():
        raise ValueError("native replay changed reference robot actions")
    if len(replayed) < len(actions) and record.get("error") is None:
        raise ValueError("canonical replay terminated early without a recorded failure")
    return record


def _pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("body pose must be finite xyz + wxyz")
    q = pose[3:]
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
        raise ValueError("body quaternion must be unit wxyz")
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return pose[:3], rotation


def _relative_marker_series(initial, ticks, role, marker_world_m):
    if role not in initial.get("object_states", {}):
        return [], "missing_initial_target"
    position0, rotation0 = _pose_matrix(initial["object_states"][role])
    local_marker = rotation0.T @ (marker_world_m-position0)
    series = []
    last_time = 0.0
    for index, tick in enumerate(ticks):
        if tick.get("tick") != index:
            raise ValueError("trace tick roster must be a contiguous prefix")
        elapsed = float(tick["simulation_time_s"])-float(initial["time"])
        if not np.isfinite(elapsed) or elapsed <= last_time:
            raise ValueError("trace times must increase strictly after reset")
        if role not in tick.get("objects", {}):
            return series, "missing_target_in_trace"
        position, rotation = _pose_matrix(tick["objects"][role])
        marker = position + rotation @ local_marker
        series.append((elapsed, marker-marker_world_m, rotation @ rotation0.T))
        last_time = elapsed
    return series, None


def compare_relative_trajectories(reference_initial: dict, reference_ticks: list,
                                  comparison_initial: dict, comparison_ticks: list, *,
                                  role: str, marker_world_m, planned_steps: int,
                                  quaternion_convention: str = "wxyz",
                                  time_tolerance_s: float = 1e-8) -> dict:
    """Measure common-prefix response, never pretend unrelated origins align.

    Marker attachment uses initial poses solely in this privileged evaluator.
    It deliberately removes initial placement error and is a relative diagnostic.
    Rotation increments R(t) R(0)^T similarly remove constant body-frame basis.
    """
    if quaternion_convention != "wxyz":
        raise ValueError("native trace quaternion convention must be explicitly wxyz")
    marker = np.asarray(marker_world_m, dtype=np.float64)
    if marker.shape != (3,) or not np.isfinite(marker).all():
        raise ValueError("public marker must be a finite world point")
    if isinstance(planned_steps, bool) or not isinstance(planned_steps, int) or planned_steps <= 0:
        raise ValueError("planned fixed action count must be positive")
    if len(reference_ticks) > planned_steps or len(comparison_ticks) > planned_steps:
        raise ValueError("trace exceeds its predeclared action horizon")
    reference, ref_failure = _relative_marker_series(reference_initial, reference_ticks, role, marker)
    comparison, cmp_failure = _relative_marker_series(comparison_initial, comparison_ticks, role, marker)
    n = min(len(reference), len(comparison))
    times = []; position_errors = []; rotation_errors = []
    for (rt, rp, rr), (ct, cp, cr) in zip(reference[:n], comparison[:n]):
        if not np.isclose(rt, ct, atol=time_tolerance_s, rtol=0):
            raise ValueError("reference/comparison control timestamps differ; no interpolation")
        times.append(rt)
        position_errors.append(float(100*np.linalg.norm(rp-cp)))
        cosine = np.clip((np.trace(rr.T @ cr)-1)/2, -1., 1.)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    complete = n == planned_steps
    return {
        "metric_scope": "relative_public_marker_displacement_and_world_rotation_increment",
        "replacement_scope": "target_only_oracle_context_diagnostic",
        "absolute_frame_correspondence": "NOT_ESTABLISHED",
        "position_rmse_cm": None, "rotation_error_deg": None,
        "final_position_error_cm": None, "final_rotation_error_deg": None,
        "initial_estimation_error_preserved_in_headline": None,
        "relative_diagnostic_removes_initial_placement_error": True,
        "role": role, "quaternion_convention": quaternion_convention,
        "marker_world_m": marker.tolist(), "planned_action_steps": planned_steps,
        "reference_recorded_steps": len(reference_ticks), "comparison_recorded_steps": len(comparison_ticks),
        "matched_steps": n, "matched_step_coverage": n/planned_steps,
        "complete_trace": complete, "retained_duration_s": times[-1] if times else 0.,
        "reference_missing_target": ref_failure, "comparison_missing_target": cmp_failure,
        "relative_displacement_rmse_cm": float(np.sqrt(np.mean(np.square(position_errors)))) if n else None,
        "relative_rotation_increment_mean_deg": float(np.mean(rotation_errors)) if n else None,
        "relative_full_horizon_final_displacement_error_cm": position_errors[-1] if complete else None,
        "relative_full_horizon_final_rotation_increment_error_deg": rotation_errors[-1] if complete else None,
        "relative_prefix_final_displacement_error_cm": position_errors[-1] if n else None,
        "relative_prefix_final_rotation_increment_error_deg": rotation_errors[-1] if n else None,
        "times_s": times, "relative_displacement_errors_cm": position_errors,
        "relative_rotation_increment_errors_deg": rotation_errors,
        "missing_tail_rule": "unmeasured; no state or zero padding",
    }


def _public_marker_centroid(source):
    path = Path(source["path"])
    if sha256(path) != source["sha256"]:
        raise ValueError("public marker source hash differs")
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            points = archive[source.get("points_key", "points_world_m")]
    else:
        raise ValueError("public marker source must be an explicit world-point NPY or NPZ")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("public TRAIN target cloud must be finite nonempty N x 3 world points")
    return points.mean(axis=0)


def write_public_marker_contract(points_path, out_path, *, points_key="points_world_m"):
    """Freeze an evaluation marker from already-public TRAIN construction input."""
    source = {"path": str(Path(points_path).resolve()), "sha256": sha256(points_path),
              "points_key": points_key}
    contract = {"schema_version": 1, "kind": "relative_public_train_marker",
                "selection_rule": "TRAIN target pointcloud centroid",
                "marker_world_m": _public_marker_centroid(source).tolist(),
                "public_source": source}
    with Path(out_path).open("x") as stream:
        json.dump(contract, stream, indent=2); stream.write("\n")
    return contract


def load_frame_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get("kind") != "relative_public_train_marker" or contract.get("schema_version") != 1:
        raise ValueError("requires a declared public TRAIN marker contract")
    if contract.get("selection_rule") != "TRAIN target pointcloud centroid":
        raise ValueError("marker must be frozen from the public TRAIN target pointcloud")
    marker = np.asarray(contract["marker_world_m"], dtype=float)
    if marker.shape != (3,) or not np.isfinite(marker).all():
        raise ValueError("invalid public marker")
    centroid = _public_marker_centroid(contract["public_source"])
    if not np.array_equal(marker, centroid):
        raise ValueError("declared marker differs from the public TRAIN cloud centroid")
    return contract


def compare_episode_directories(reference_dir, comparison_dir, *, frame_contract_path, role="obj"):
    from robo.eval.episode_log import read_timeseries
    ref, cmp = Path(reference_dir), Path(comparison_dir)
    frame = load_frame_contract(frame_contract_path)
    reference_result = json.loads((ref/"result.json").read_text())
    comparison_result = json.loads((cmp/"result.json").read_text())
    if comparison_result.get("execution_kind") != "fixed_action_replay":
        raise ValueError("physical replay metrics require a typed canonical replay episode")
    source = comparison_result.get("policy_identity", {}).get("source", {})
    if source.get("reference_result_sha256") != sha256(ref/"result.json"):
        raise ValueError("replay does not bind the declared reference episode")
    if source.get("actions_sha256") != sha256(ref/"actions.json"):
        raise ValueError("replay does not bind the declared reference action bank")
    for field in ("scene_id", "task_id", "reset_seed"):
        if comparison_result.get(field) != reference_result.get(field):
            raise ValueError(f"replay {field} differs from source reference")
    reference_actions = _action_array(json.loads((ref/"actions.json").read_text()))
    reference_ticks = read_timeseries(ref/"trace.json.gz")
    comparison_ticks = read_timeseries(cmp/"trace.json.gz")
    actual = np.asarray(json.loads((cmp/"actions.json").read_text()), dtype=np.float64)
    if actual.size == 0: actual = actual.reshape(0, 12)
    if actual.shape != (len(comparison_ticks), 12) or len(reference_ticks) != len(reference_actions):
        raise ValueError("action/trace row accounting differs")
    if actual.tobytes() != reference_actions[:len(actual)].tobytes():
        raise ValueError("comparison robot actions do not match the source prefix")
    result = compare_relative_trajectories(
        json.loads((ref/"initial_state.json").read_text()), reference_ticks,
        json.loads((cmp/"initial_state.json").read_text()), comparison_ticks,
        role=role, marker_world_m=frame["marker_world_m"], planned_steps=len(reference_actions))
    result["provenance"] = {"frame_contract_sha256": sha256(frame_contract_path),
                            "reference_files": {name: sha256(ref/name) for name in
                                                ["actions.json", "trace.json.gz", "initial_state.json", "result.json"]},
                            "comparison_files": {name: sha256(cmp/name) for name in
                                                 ["actions.json", "trace.json.gz", "initial_state.json", "result.json"]}}
    result["native_replay_outcome"] = comparison_result
    result.update({"planned_replay_episodes": 1,
                   "executed_replay_episodes": int(bool(comparison_result.get("executed"))),
                   "completed_replay_episodes": int(len(comparison_ticks) == len(reference_actions)
                                                      and comparison_result.get("error") is None),
                   "reference_native_success": reference_result.get("success"),
                   "replay_native_success": comparison_result.get("success"),
                   "reference_native_policy_horizon": reference_result.get("horizon"),
                   "success_scope": "unchanged native predicate at fixed replay prefix, not necessarily native policy horizon"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--actions-sha256", required=True)
    parser.add_argument("--reference-episode", required=True)
    parser.add_argument("--canonical-reference", required=True)
    parser.add_argument("--frame-contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reset-seed", required=True, type=int)
    parser.add_argument("--treatment-id", required=True)
    parser.add_argument("--object-dir")
    parser.add_argument("--object-id", default="target")
    parser.add_argument("--role", default="obj")
    args = parser.parse_args(argv)
    from robo.roundtrip.spec import load_spec
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import load_reference_bundle, prepare_paired_adapter
    config = load_spec(args.config)
    bundle = load_reference_bundle(args.reference_episode, args.canonical_reference,
                                   config=config, reset_seed=args.reset_seed)
    load_frame_contract(args.frame_contract)
    if Path(args.actions).resolve() != (Path(args.reference_episode)/"actions.json").resolve():
        raise ValueError("action bank must be the declared reference episode's actions")
    if args.reset_seed not in config["reset_seeds"]:
        raise ValueError("reset seed is outside declared DEV roster")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    adapter = RoboCasaAdapter(config)
    try:
        imported = prepare_paired_adapter(adapter, bundle, object_dir=args.object_dir,
                                          object_id=args.object_id, role=args.role)
        (out/"import_receipt.json").write_text(json.dumps(imported, indent=2)+"\n")
        run_replay_episode(adapter, actions_path=args.actions, expected_actions_sha256=args.actions_sha256,
            config=config, reset_seed=args.reset_seed, out_dir=out/"episode", treatment_id=args.treatment_id,
            source_identity={"reference_result_sha256": sha256(Path(args.reference_episode)/"result.json"),
                             "canonical_reference": bundle["provenance"]})
    finally:
        adapter.close()
    metrics = compare_episode_directories(args.reference_episode, out/"episode",
                                         frame_contract_path=args.frame_contract, role=args.role)
    (out/"replay_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    row = {key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}
    with (out/"replay_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
