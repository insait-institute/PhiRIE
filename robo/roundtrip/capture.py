"""Static native sensor capture and an enforceable constructor mount boundary.

The adapter is privileged. Only TRAIN RGB/(optional metric camera-z depth),
OpenCV cameras, declared robot configuration and language leave this process.
No reference simulator import is required in a constructor process.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import Mapping

import numpy as np


class CaptureError(ValueError):
    pass


@dataclass(frozen=True)
class CaptureSpec:
    capture_id: str
    width: int = 1280
    height: int = 720
    sensor_regime: str = "ideal_rgbd"
    counts: Mapping[str, int] = field(default_factory=lambda: {"train": 120, "dev": 20, "test": 40})
    translation_tolerance_m: float = 1e-7
    rotation_tolerance_rad: float = 1e-6
    robot_mode: str = "parked"

    def validate(self):
        if not re.fullmatch(r"c-[0-9a-f]{16,64}", self.capture_id):
            raise CaptureError("capture_id must be opaque c- plus 16..64 hex digits")
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) <= 0):
            raise CaptureError("positive integer image dimensions required")
        if self.sensor_regime not in {"ideal_rgbd", "posed_rgb"}:
            raise CaptureError("sensor regime must explicitly declare RGB or RGB-D")
        if (set(self.counts) != {"train", "dev", "test"}
                or any(type(n) is not int or n < 0 for n in self.counts.values())
                or self.counts["train"] <= 0 or self.counts["test"] <= 0):
            raise CaptureError("exact nonnegative split counts with TRAIN and TEST required")
        if self.robot_mode not in {"parked", "hidden"}:
            raise CaptureError("declare robot parked or hidden")
        for value in (self.translation_tolerance_m, self.rotation_tolerance_rad):
            if not math.isfinite(value) or value < 0:
                raise CaptureError("finite nonnegative static-state tolerances required")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _json_bytes(value):
    return (json.dumps(_json_value(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlap(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


def validate_camera(K, T_world_from_camera, *, width, height):
    """Check per-frame undistorted pinhole K and right-handed OpenCV c2w."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_world_from_camera, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise CaptureError("camera K must be finite 3x3")
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-10)
            or abs(K[1, 0]) > 1e-10 or abs(K[0, 1]) > 1e-10
            or not 0 <= K[0, 2] < width or not 0 <= K[1, 2] < height):
        raise CaptureError("unsupported or invalid undistorted pinhole K")
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-10):
        raise CaptureError("camera transform must be finite homogeneous 4x4")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise CaptureError("camera rotation must be proper orthonormal SO(3); no scale/reflection")
    return K, T


def backproject_camera_z(depth_m, K, T_world_from_camera):
    """Metric world points; depth is optical-axis z, never Euclidean ray range."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating) or not np.isfinite(depth).all() or (depth < 0).any():
        raise CaptureError("metric z depth must be finite nonnegative floating HxW; 0 means missing")
    height, width = depth.shape
    K, T = validate_camera(K, T_world_from_camera, width=width, height=height)
    v, u = np.indices(depth.shape)
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=-1)
    points = rays * depth[..., None]
    return points @ T[:3, :3].T + T[:3, 3], depth > 0


def _state(adapter):
    result = {}
    raw = adapter.get_named_body_state()
    if not isinstance(raw, dict) or not raw:
        raise CaptureError("adapter must expose nonempty private named-body pose state")
    for name, value in raw.items():
        # Agreed pose convention: [x,y,z,qw,qx,qy,qz], world frame.
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-7):
            raise CaptureError("body state must be finite xyz + unit quaternion wxyz")
        result[str(name)] = pose.copy()
    return result


def _check_static(initial, current, spec):
    if set(initial) != set(current):
        raise CaptureError("mixed-state capture: body roster changed")
    for name, before in initial.items():
        after = current[name]
        distance = float(np.linalg.norm(after[:3]-before[:3]))
        dot = float(np.clip(abs(np.dot(after[3:], before[3:])), 0, 1))
        angle = 0.0 if np.array_equal(after[3:], before[3:]) else 2*math.acos(dot)
        if distance > spec.translation_tolerance_m or angle > spec.rotation_tolerance_rad:
            raise CaptureError("mixed-state capture: body pose exceeds predeclared tolerance")


def _validate_plan(frames, spec):
    ids = []
    counts = {s: 0 for s in spec.counts}
    for frame in frames:
        if set(frame) != {"frame_id", "split", "camera"}:
            raise CaptureError("frame declaration requires only frame_id/split/camera")
        if not re.fullmatch(r"f[0-9]{6}", frame["frame_id"]):
            raise CaptureError("frame_id must be opaque f plus six digits")
        if frame["split"] not in counts or not isinstance(frame["camera"], dict):
            raise CaptureError("invalid split or camera declaration")
        ids.append(frame["frame_id"])
        counts[frame["split"]] += 1
    if len(set(ids)) != len(ids):
        raise CaptureError("duplicate frame IDs across capture/splits")
    if counts != dict(spec.counts):
        raise CaptureError("frame roster differs from predeclared split counts")


def capture_static(adapter, *, spec: CaptureSpec, frames, public_out, vault,
                   robot_config, task_instruction, forbidden_public_tokens=()):
    """Collect one canonical static scan. Failures retain partial images privately/publicly
    but have no successful public manifest and cannot pass constructor validation.

    Adapter.render_capture(camera, width=..., height=...) returns rgb, depth_m,
    K, T_world_from_camera, timestamp, and optionally depth_semantics='camera_z'.
    Native camera selectors occur only in the private capture plan.
    """
    spec.validate()
    frames = list(frames)
    _validate_plan(frames, spec)
    public_root, private_root = Path(public_out).resolve(), Path(vault).resolve()
    if _overlap(public_root, private_root):
        raise CaptureError("private vault and public root must not overlap")
    public = public_root / spec.capture_id
    private = private_root / spec.capture_id
    if public.exists() or private.exists():
        raise CaptureError("immutable capture ID already exists; use a new ID")
    # Known robot config is user-declared configuration, never native scene metadata.
    allowed_robot_fields = {"robot", "urdf", "base_pose", "controller", "cameras", "action_convention", "joint_names"}
    if not isinstance(robot_config, dict) or not set(robot_config) <= allowed_robot_fields:
        raise CaptureError("robot_config contains undeclared public metadata fields")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise CaptureError("public task instruction required")
    public_metadata = _json_bytes({"robot": robot_config, "instruction": task_instruction}).decode()
    if any(token and token in public_metadata for token in forbidden_public_tokens):
        raise CaptureError("private identifier appears in public metadata")
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows = {s: [] for s in spec.counts}
    try:
        initial = _state(adapter)
        _write_json(private / "canonical_native_state.json", adapter.get_state())
        _write_json(private / "body_state_initial.json", initial)
        _write_json(private / "capture_plan.json", {"frames": frames, "spec": spec.__dict__})
        seen_cameras = set()
        for frame in frames:
            _check_static(initial, _state(adapter), spec)
            rendered = adapter.render_capture(frame["camera"], width=spec.width, height=spec.height)
            _check_static(initial, _state(adapter), spec)
            rgb = np.asarray(rendered["rgb"])
            if rgb.dtype != np.uint8 or rgb.shape != (spec.height, spec.width, 3):
                raise CaptureError("RGB must be uint8 HxWx3 at declared size; no resizing")
            K, T = validate_camera(rendered["K"], rendered["T_world_from_camera"], width=spec.width, height=spec.height)
            camera_key = hashlib.sha256(K.tobytes()+T.tobytes()).hexdigest()
            if camera_key in seen_cameras:
                raise CaptureError("duplicate camera view across capture/splits")
            seen_cameras.add(camera_key)
            timestamp = float(rendered["timestamp"])
            if not math.isfinite(timestamp):
                raise CaptureError("native timestamp must be finite")
            # Require explicit semantics in the producer API contract. Missing pixels are 0.
            if rendered.get("depth_semantics", "camera_z") != "camera_z":
                raise CaptureError("adapter supplied ray-range depth, expected camera-z meters")
            depth = rendered.get("depth_m")
            if spec.sensor_regime == "ideal_rgbd":
                if depth is None:
                    raise CaptureError("RGB-D capture is missing metric depth")
                depth = np.asarray(depth)
                if depth.shape != (spec.height, spec.width):
                    raise CaptureError("depth size differs from declared RGB size")
                backproject_camera_z(depth, K, T)
            split, fid = frame["split"], frame["frame_id"]
            destination = public if split == "train" else private
            image_rel = f"{split}/rgb/{fid}.png"
            image_path = destination / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            with image_path.open("xb") as stream:
                Image.fromarray(rgb).save(stream, format="PNG", compress_level=6)
            row = {"frame_id": fid, "rgb": image_rel, "rgb_sha256": _hash(image_path),
                   "K": K.tolist(), "T_world_from_camera": T.tolist(), "timestamp": timestamp}
            if spec.sensor_regime == "ideal_rgbd":
                depth_rel = f"{split}/depth_m/{fid}.npy"
                depth_path = destination / depth_rel
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                with depth_path.open("xb") as stream:
                    np.save(stream, depth, allow_pickle=False)
                row.update(depth_m=depth_rel, depth_sha256=_hash(depth_path), missing_depth_pixels=int((depth == 0).sum()))
            rows[split].append(row)
        _check_static(initial, _state(adapter), spec)
        for split, entries in rows.items():
            dest = public if split == "train" else private
            (dest/split).mkdir(exist_ok=True)
            with (dest/split/"cameras.jsonl").open("x") as stream:
                for row in entries:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        _write_json(public / "robot_config.json", robot_config)
        with (public/"task_instruction.txt").open("x") as stream:
            stream.write(task_instruction+"\n")
        public_manifest = {"schema_version": 1, "status": "PASS", "capture_id": spec.capture_id,
            "sensor_regime": spec.sensor_regime, "metric_camera_pose_assistance": True,
            "camera_convention": "OpenCV_x_right_y_down_z_forward", "transform_convention": "T_world_from_camera",
            "depth_semantics": "camera_z_meters_zero_missing" if spec.sensor_regime == "ideal_rgbd" else None,
            "color_space": "sRGB", "distortion": "none_undistorted", "width": spec.width, "height": spec.height,
            "robot_mode": spec.robot_mode, "train_frames": len(rows["train"]), "split": "train_only",
            "masks": "none_automatic_inference_required", "files": {str(p.relative_to(public)): _hash(p) for p in sorted(public.rglob("*")) if p.is_file()}}
        _write_json(private/"capture_receipt.json", {"status": "PASS", "capture_id": spec.capture_id,
            "counts": {s: len(r) for s, r in rows.items()}, "static_body_roster": sorted(initial),
            "canonical_state_sha256": _hash(private/"canonical_native_state.json"),
            "public_manifest_sha256": hashlib.sha256(_json_bytes(public_manifest)).hexdigest(),
            "private_files": {str(p.relative_to(private)): _hash(p) for p in sorted(private.rglob("*")) if p.is_file()}})
        _write_json(public / "capture_manifest.json", public_manifest)
        return {"public": str(public), "private": str(private), "manifest": public_manifest}
    except Exception as exc:
        _write_json(private/"capture_failure.json", {"status": "FAIL", "error_type": type(exc).__name__, "reason": str(exc), "completed_frames": {s: len(v) for s, v in rows.items()}})
        raise


def validate_public_capture(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads((path/"capture_manifest.json").read_text())
    if manifest.get("status") != "PASS" or manifest.get("split") != "train_only":
        raise CaptureError("constructor requires a successful TRAIN-only capture")
    files = manifest["files"]
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != set(files) | {"capture_manifest.json"}:
        raise CaptureError("public capture file roster differs")
    for name, digest in files.items():
        file = path/name
        if Path(name).is_absolute() or ".." in Path(name).parts or file.is_symlink() or not file.resolve().is_relative_to(path):
            raise CaptureError("unsafe public capture path")
        if _hash(file) != digest:
            raise CaptureError("public capture content hash mismatch")
        if name.startswith(("dev/", "test/")) or name not in {"robot_config.json", "task_instruction.txt"} and not name.startswith("train/"):
            raise CaptureError("non-TRAIN/private file in public capture")
        if manifest["sensor_regime"] == "posed_rgb" and ("depth" in name or name.endswith((".ply", ".obj", ".npz"))):
            raise CaptureError("depth-derived artifact in RGB-only public bundle")
    rows = [json.loads(line) for line in (path/"train/cameras.jsonl").read_text().splitlines()]
    if len(rows) != manifest["train_frames"] or len({r['frame_id'] for r in rows}) != len(rows):
        raise CaptureError("TRAIN camera count/identity mismatch")
    for row in rows:
        validate_camera(row["K"], row["T_world_from_camera"], width=manifest["width"], height=manifest["height"])
    return manifest


def slurm_device_minors(environ, *, query=None):
    """Map scheduler-visible GPU UUIDs to actual /dev minors.

    GPU containers can remap Slurm index, NVML index and device minor three
    different ways. The UUID supplied by Slurm identifies the allocated card;
    querying that card never authorizes any additional visible GPU.
    """
    allocation = environ.get("SLURM_STEP_GPUS") or environ.get("SLURM_JOB_GPUS", "")
    indices = set()
    for part in allocation.split(","):
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part):
            raise CaptureError("Slurm physical GPU indices unavailable; no CUDA device fallback")
        ends = [int(value) for value in part.split("-")]
        if len(ends) == 1:
            indices.add(ends[0])
        elif ends[0] <= ends[1] and ends[1] - ends[0] < 1024:
            indices.update(range(ends[0], ends[1] + 1))
        else:
            raise CaptureError("invalid Slurm GPU allocation range")
    visible = environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-"):
        return indices
    uuids = visible.split(",")
    if len(uuids) != len(indices) or len(set(uuids)) != len(uuids):
        raise CaptureError("visible GPU UUID count differs from Slurm allocation")
    query = subprocess.check_output if query is None else query
    minors = set()
    for uuid in uuids:
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
            raise CaptureError("invalid allocated GPU UUID")
        root = ET.fromstring(query(["nvidia-smi", "-q", "-x", "--id=" + uuid], text=True))
        matches = [gpu for gpu in root.findall("gpu") if gpu.findtext("uuid") == uuid]
        if len(matches) != 1 or not (matches[0].findtext("minor_number") or "").isdigit():
            raise CaptureError("allocated UUID has no unique physical device minor")
        minors.add(int(matches[0].findtext("minor_number")))
    if len(minors) != len(indices):
        raise CaptureError("allocated UUIDs alias the same physical device")
    return minors


def _cuda_device_paths(device_paths):
    """Only NVIDIA character devices belonging to this Slurm GPU allocation.

    Slurm job/step physical GPU indices are authoritative; CUDA_VISIBLE_DEVICES
    may be remapped and is deliberately not used as an allocation proof.
    UUID-only allocation strings need an externally resolved physical-index
    contract; this helper rejects them instead of exposing every GPU.
    """
    paths = [str(path) for path in device_paths]
    if not paths:
        return []
    if len(paths) != len(set(paths)):
        raise CaptureError("duplicate CUDA device allowlist member")
    if not os.environ.get("SLURM_JOB_ID"):
        raise CaptureError("CUDA device exposure requires a Slurm allocation")
    allocated = slurm_device_minors(os.environ)
    selected = set()
    for path in paths:
        match = re.fullmatch(r"/dev/nvidia([0-9]+|ctl|-uvm|-uvm-tools)", path)
        if match is None:
            raise CaptureError("CUDA allowlist accepts only individual NVIDIA device nodes")
        if match[1].isdigit():
            selected.add(int(match[1]))
            if int(match[1]) not in allocated:
                raise CaptureError("CUDA device lies outside the Slurm allocation")
        try:
            mode = os.lstat(path).st_mode
        except OSError as exc:
            raise CaptureError("requested CUDA character device is unavailable") from exc
        if not stat.S_ISCHR(mode):
            raise CaptureError("CUDA allowlist member must be a real character device, not symlink/file/directory")
    if not selected:
        raise CaptureError("CUDA control nodes require at least one allocated GPU device")
    return paths


def constructor_command(*, bubblewrap, public_capture, output, runtime_mounts,
                        forbidden_roots, argv, device_paths=()):
    """Empty-root, read-only allowlist; no network, no host home, no native assets.

    runtime_mounts maps trusted dependency/code/checkpoint source paths to sandbox
    absolute paths. Stage a minimal constructor environment: mounting a native
    simulator environment or a repository/data ancestor is forbidden by caller's
    explicit reference/native roots. No fallback to an unsandboxed launch.
    """
    devices = _cuda_device_paths(device_paths)
    validate_public_capture(public_capture)
    public_capture, output = Path(public_capture).resolve(strict=True), Path(output).resolve(strict=True)
    bwrap = Path(bubblewrap).resolve(strict=True)
    if not output.is_dir() or any(output.iterdir()):
        raise CaptureError("constructor output must be an empty existing directory")
    if not forbidden_roots:
        raise CaptureError("private vault and native asset/API roots must be declared")
    forbidden = [Path(p).resolve(strict=True) for p in forbidden_roots]
    if _overlap(public_capture, output):
        raise CaptureError("public capture and constructor output overlap")
    mounts = [(public_capture, "/capture", True), (output, "/output", False)]
    targets = {"/capture", "/output", "/proc", "/dev", "/tmp"}
    for source, target in runtime_mounts.items():
        source, target = Path(source).resolve(strict=True), str(target)
        if not target.startswith("/") or ".." in Path(target).parts or target == "/":
            raise CaptureError("runtime mount target must be a restricted absolute path")
        if any(Path(target) == Path(existing) or Path(target) in Path(existing).parents
               or Path(existing) in Path(target).parents for existing in targets):
            raise CaptureError("runtime mount targets overlap protected mounts")
        targets.add(target)
        mounts.append((source, target, True))
    for source, _, _ in mounts:
        if any(_overlap(source, private) for private in forbidden):
            raise CaptureError("allowlist mount exposes private/native reference tree")
    if not argv or not Path(argv[0]).is_absolute():
        raise CaptureError("sandbox command requires an absolute executable")
    command = [str(bwrap), "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
               "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for source, target, readonly in mounts:
        command += ["--ro-bind" if readonly else "--bind", str(source), target]
    for device in devices:
        command += ["--dev-bind", device, device]
    return command + ["--chdir", "/output", "--", *map(str, argv)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_public_capture(args.validate)
    report = {"status": "PASS", "capture_id": result["capture_id"], "train_frames": result["train_frames"]}
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
