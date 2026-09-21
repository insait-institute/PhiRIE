"""Privileged, deterministic RoboCasa 6-TRAIN/2-held-out static capture smoke.

This acquisition driver consumes native camera configuration, never hidden
object poses, for its path. It delegates serialization and static checks to
capture_static; reconstruction workers receive only the resulting TRAIN mount.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from robo.manifest.hash import canonical_hash, git_snapshot
from robo.roundtrip.capture import CaptureError, CaptureSpec, capture_static, validate_camera, validate_public_capture
from robo.roundtrip.spec import load_spec, validate_spec

NATIVE_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")
TRAIN_LOCAL_X_M = (-0.12, 0.0, 0.12)
TEST_LOCAL_TRANSLATION_M = (0.06, 0.18, 0.0)
TRAJECTORY_VERSION = "native_camera_local_translations_v1"


def _save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _identity(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise CaptureError("source must be a canonical regular file")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def check_identity_report(path, config):
    """Bind successful U0/U1 report to its recorded exact native configuration."""
    path = Path(path).resolve(strict=True)
    report = json.loads(path.read_text())
    run_path = path.parent / "run_manifest.json"
    run = json.loads(run_path.read_text())
    if (run.get("code", {}).get("dirty") is not False or run.get("phase") != "identity" or run.get("config") != config
            or run.get("config_sha256") != canonical_hash(config)):
        raise CaptureError("identity source config differs from capture native config")
    if (report.get("passed") is not True or report.get("initial_objects_equal") is not True
            or report.get("initial_observation_diff") != {}
            or report.get("u1_initial_observation_diff") != {}):
        raise CaptureError("native U0/U1 identity prerequisite has not passed")
    for name in ("u0", "u1"):
        rows = report.get(name)
        if not isinstance(rows, list) or len(rows) != 10:
            raise CaptureError("identity control lacks its ten recorded action steps")
        if any(row.get("observation_diff") != {} or row.get("state_max_abs") != 0
               or row.get("predicate_equal") is not True for row in rows):
            raise CaptureError("native identity action/state/predicate mismatch")
    return {"report": _identity(path), "run_manifest": _identity(run_path)}


def load_canonical_reference(directory, config, *, reference_seed=0):
    """Privileged pilot initial-state read, never a constructor input."""
    directory = Path(directory).resolve(strict=True)
    state_path, xml_path = directory/"canonical_state.json", directory/"scene.xml"
    run_path = directory.parent/"run_manifest.json"
    identities = {"state":_identity(state_path), "xml":_identity(xml_path), "run_manifest":_identity(run_path)}
    run = json.loads(run_path.read_text())
    if (directory.name != f"canonical_seed{reference_seed}" or run.get("phase") != "pilot"
            or run.get("config") != config or run.get("config_sha256") != canonical_hash(config)
            or run.get("code", {}).get("dirty") is not False):
        raise CaptureError("canonical reference must bind the exact clean native pilot config/seed")
    state = json.loads(state_path.read_text())
    metadata = state.get("native_metadata", {})
    if (metadata.get("layout_id") != config["instance"]["layout_id"]
            or metadata.get("style_id") != config["instance"]["style_id"]
            or not isinstance(metadata.get("lang"), str) or not metadata["lang"].strip()):
        raise CaptureError("canonical native metadata lacks matching layout/style/instruction")
    for key in ("task_id", "env_name"):
        if key in metadata and metadata[key] != config["instance"]["task_id"]:
            raise CaptureError("canonical metadata task differs from configured native task")
    for key in ("qpos", "qvel", "act", "time"):
        if key not in state or not np.isfinite(np.asarray(state[key], dtype=float)).all():
            raise CaptureError("canonical native integration state is incomplete/nonfinite")
    return {"identities":identities, "state":state, "xml":xml_path.read_text()}


def _check_canonical_restore(adapter, source):
    if adapter.native_success():
        raise CaptureError("canonical capture reference is already terminal")
    if adapter.get_policy_observation().get("annotation.human.task_description") != source["native_metadata"]["lang"]:
        raise CaptureError("restored native instruction differs from canonical metadata")
    observed = adapter.get_state()
    for key in ("qpos", "qvel", "act", "time"):
        if key not in observed or not np.array_equal(np.asarray(observed[key]), np.asarray(source[key])):
            raise CaptureError("restored native integration state differs at " + key)


def native_camera_poses(adapter):
    """Privileged camera configuration only; no native body/object state lookup."""
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    return {name: np.asarray(get_camera_extrinsic_matrix(adapter.native.sim, name), dtype=float)
            for name in NATIVE_CAMERAS}


def smoke_view_plan(camera_poses):
    """Predeclared translations in known OpenCV camera frames; no target fitting.

    TRAIN spans both cameras' horizontal translations. Held-out views lie on a
    distinct lower plane 18 cm away. This is a small engineering split, not an
    out-of-distribution benchmark or the proposed 120/20/40 full acquisition.
    """
    if set(camera_poses) != set(NATIVE_CAMERAS):
        raise CaptureError("smoke trajectory requires the fixed native left/right cameras")
    frames = []
    # Camera validation uses a dummy valid K solely to validate SE(3), not data K.
    for name in NATIVE_CAMERAS:
        validate_camera(np.eye(3), camera_poses[name], width=1, height=1)
    for name in NATIVE_CAMERAS:
        for x in TRAIN_LOCAL_X_M:
            T = np.asarray(camera_poses[name], dtype=float).copy()
            T[:3, 3] += T[:3, :3] @ np.array([x, 0., 0.])
            frames.append({"frame_id": f"f{len(frames):06d}", "split": "train",
                "camera": {"native_name": name, "T_world_from_camera": T.tolist()}})
    for name in NATIVE_CAMERAS:
        T = np.asarray(camera_poses[name], dtype=float).copy()
        T[:3, 3] += T[:3, :3] @ np.array(TEST_LOCAL_TRANSLATION_M)
        frames.append({"frame_id": f"f{len(frames):06d}", "split": "test",
            "camera": {"native_name": name, "T_world_from_camera": T.tolist()}})
    poses = [np.array(frame['camera']['T_world_from_camera']) for frame in frames]
    if any(np.allclose(a, b, rtol=0, atol=1e-10) for i, a in enumerate(poses) for b in poses[i+1:]):
        raise CaptureError("native camera path has duplicate poses; no replacement search")
    return frames


def collect_native_smoke(*, config, identity_report, capture_id, out, public_out, vault,
                         width=1280, height=720, adapter_factory=None, camera_pose_reader=None, canonical_reference=None, reference_seed=0):
    validate_spec(config)
    if (config["instance"]["layout_id"] != 11 or config["instance"]["style_id"] != 11
            or config["reset_seeds"][0] != 0):
        raise CaptureError("this predeclared capture smoke fixes layout11/style11/seed0")
    if reference_seed not in config["reset_seeds"] or isinstance(reference_seed, bool):
        raise CaptureError("reference seed is outside the declared pilot roster")
    if reference_seed != 0 and canonical_reference is None:
        raise CaptureError("nonzero reference seed requires its sealed canonical reference")
    identity = check_identity_report(identity_report, config)
    canonical = load_canonical_reference(canonical_reference, config, reference_seed=reference_seed) if canonical_reference is not None else None
    spec = CaptureSpec(capture_id=capture_id, width=width, height=height,
        counts={"train": 6, "dev": 0, "test": 2}, robot_mode="parked")
    spec.validate()
    code = git_snapshot()
    if code["commit"] == "nogit" or code["dirty"] is not False:
        raise CaptureError("native capture requires a committed clean source")
    out = Path(out).resolve()
    public_out = Path(public_out).resolve()
    vault = Path(vault).resolve()
    for left, right in ((out,public_out),(out,vault),(public_out,vault)):
        if left == right or left in right.parents or right in left.parents:
            raise CaptureError("run provenance, public capture and private vault roots must not overlap")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _save(out/"run_manifest.json", {"schema_version":1,"scope":"native_static_capture_smoke",
        "code":code,"config":config,"config_sha256":canonical_hash(config),"identity_control":identity,
        "canonical_reference": canonical["identities"] if canonical else None,
        "reference_policy_initial_state_bound": canonical is not None,
        "capture_id":capture_id,"width":width,"height":height,"seed":reference_seed,
        "planned_counts":dict(spec.counts),"trajectory":{
            "version":TRAJECTORY_VERSION,"native_cameras":list(NATIVE_CAMERAS),
            "train_local_x_m":list(TRAIN_LOCAL_X_M),"test_local_translation_m":list(TEST_LOCAL_TRANSLATION_M),
            "pose_assistance":"known native camera poses, idealized privileged acquisition aid",
            "native_fixture_bounds_used":False,"native_object_poses_used_for_camera_path":False,
            "camera_collision_check":"NOT_RUN; unvalidated acquisition path, no collision-free claim",
            "heldout_scope":"spatially offset smoke views; no out-of-distribution claim"},
        "robot_scan_state":"native canonical reset, no actuation or hiding",
        "measurement_scope":"sensor/capture integrity; no learned policy or reconstruction measurement"})
    if adapter_factory is None:
        from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
        adapter_factory = RoboCasaAdapter
    adapter = None
    try:
        adapter = adapter_factory(config)
        adapter.reset_from_spec({"seed":reference_seed})
        if canonical is not None:
            adapter.import_xml(canonical["xml"], canonical_state=canonical["state"])
            _check_canonical_restore(adapter, canonical["state"])
        poses = (camera_pose_reader or native_camera_poses)(adapter)
        frames = smoke_view_plan(poses)
        observation = adapter.get_policy_observation()
        instruction = observation["annotation.human.task_description"]
        result = capture_static(adapter, spec=spec, frames=frames, public_out=public_out, vault=vault,
            robot_config={"robot":config["robot"], "action_convention":"native RoboCasa configured controller"},
            task_instruction=instruction)
        private = Path(result["private"])
        adapter.export_reference_for_evaluator(private/"native_reference")
        _save(private/"environment_lock.json", adapter.environment_lock())
        public_manifest = validate_public_capture(result["public"])
        reference_files = {str(path.relative_to(private)): _identity(path)
            for path in sorted(private.rglob("*")) if path.is_file()}
        _save(private/"reference_hashes.json", reference_files)
        report = {"status":"PASS","scope":"native_static_capture_smoke","paper_ready":False,
            "capture_id":capture_id,"view_counts":dict(spec.counts),"width":width,"height":height,
            "capture_public_manifest":_identity(Path(result["public"])/"capture_manifest.json"),
            "private_reference_hashes":_identity(private/"reference_hashes.json"),
            "canonical_state":_identity(private/"canonical_native_state.json"),
            "native_reference_xml":_identity(private/"native_reference/scene.xml"),
            "train_frames":public_manifest["train_frames"],"elapsed_seconds":time.monotonic()-started,
            "constructor_isolation":"NOT_RUN here; constructor must use capture.constructor_command allowlist",
            "source_state":"restored native pilot initial state" if canonical else "native seed reset, not bound to a policy episode",
            "reference_policy_initial_state_bound":canonical is not None,
            "canonical_reference":canonical["identities"] if canonical else None}
        if canonical is not None:
            for item in canonical["identities"].values():
                if _identity(item["path"]) != item:
                    raise CaptureError("canonical reference changed during capture")
        if git_snapshot() != code:
            raise CaptureError("source changed during static capture")
        _save(out/"capture_result.json",report)
        return report
    except Exception as exc:
        _save(out/"capture_failure.json",{"status":"FAIL","error_type":type(exc).__name__,
            "reason":str(exc),"planned_counts":dict(spec.counts),"elapsed_seconds":time.monotonic()-started})
        raise
    finally:
        if adapter is not None:
            adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--identity-report",required=True)
    parser.add_argument("--canonical-reference", help="Private native pilot canonical_seedN directory; binds exact policy initial state")
    parser.add_argument("--reference-seed",type=int,default=0)
    parser.add_argument("--capture-id",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--public-out",required=True)
    parser.add_argument("--vault",required=True)
    parser.add_argument("--width",type=int,default=1280)
    parser.add_argument("--height",type=int,default=720)
    args=parser.parse_args(argv)
    report=collect_native_smoke(config=load_spec(args.config),identity_report=args.identity_report,
        capture_id=args.capture_id,out=args.out,public_out=args.public_out,vault=args.vault,
        width=args.width,height=args.height,canonical_reference=args.canonical_reference,reference_seed=args.reference_seed)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
