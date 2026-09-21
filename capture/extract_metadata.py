"""Assemble the non-video parts of a capture bundle: device/stream metadata,
optional sidecar RGB-D/AR-pose/calibration-marker files, capture timestamps,
and a privacy/redaction pass -- everything `capture/README.md` lists in the
bundle spec besides the validation report itself (`validate_video.py`).

Nothing here is required for a plain-RGB capture: every sidecar lookup is
best-effort and absent-by-default. This module only ever reads the video
container, filesystem timestamps, and whatever sidecar files a capture app
chose to drop next to the video -- never task labels or object annotations
(there is nothing here that could even name an object).

Usage:
    python -m capture.extract_metadata VIDEO.mp4 --out outputs/capture_bundle/<name> [--redact]

Writes <out>/metadata.json. With --redact, also writes
<out>/redacted_preview/*.jpg (faces blurred, best-effort) for operator
review before the raw video leaves the capture device's trusted boundary.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from capture.validate_video import probe_stream, sample_frames

# Sidecar filename patterns this looks for next to `<stem>.mp4`, in the order
# a capture app is likely to write them. All optional.
DEVICE_METADATA_SUFFIXES = [".metadata.json", "_metadata.json"]
POSE_SUFFIXES = [".poses.jsonl", "_poses.jsonl", ".poses.json", "_poses.json"]
DEPTH_SUFFIXES = [".depth.npz", "_depth.npz"]
DEPTH_DIR_SUFFIXES = ["_depth", ".depth"]
MARKER_SUFFIXES = [".markers.json", "_markers.json"]

FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def _find_sidecar(video: Path, suffixes: list[str]) -> Path | None:
    for suf in suffixes:
        cand = video.parent / f"{video.stem}{suf}"
        if cand.exists():
            return cand
    return None


def _find_sidecar_dir(video: Path, suffixes: list[str]) -> Path | None:
    for suf in suffixes:
        cand = video.parent / f"{video.stem}{suf}"
        if cand.is_dir():
            return cand
    return None


# ---------------------------------------------------------------------------
# Device / stream metadata (ffprobe tags -- rotation, creation_time, make/model
# on capture apps that write QuickTime/EXIF-style tags)
# ---------------------------------------------------------------------------

def device_metadata(video: Path) -> dict[str, Any]:
    info = probe_stream(video)
    stream, fmt = info["stream"], info["format"]
    tags = {**(fmt.get("tags") or {}), **(stream.get("tags") or {})}
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "rotation": stream.get("rotation"),
        "duration_s": float(fmt.get("duration") or stream.get("duration") or 0.0),
        "container_tags": tags,  # e.g. com.apple.quicktime.make/model, creation_time
    }


def capture_timestamps(video: Path, device_meta: dict[str, Any]) -> dict[str, Any]:
    tags = device_meta.get("container_tags") or {}
    creation_time = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
    stat = video.stat()
    return {
        "container_creation_time": creation_time,
        "filesystem_mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "duration_s": device_meta.get("duration_s"),
    }


# ---------------------------------------------------------------------------
# Optional sidecars: AR poses, LiDAR depth, precomputed marker observations
# ---------------------------------------------------------------------------

def pose_sidecar_summary(video: Path) -> dict[str, Any]:
    """ARKit/ARCore per-frame poses, if the capture app wrote them next to
    the video. Logged when present; never required (plain RGB is the
    baseline capture mode, per the plan)."""
    path = _find_sidecar(video, POSE_SUFFIXES)
    if path is None:
        return {"present": False}
    try:
        if path.suffix == ".jsonl":
            n = sum(1 for line in path.read_text().splitlines() if line.strip())
        else:
            data = json.loads(path.read_text())
            n = len(data) if isinstance(data, list) else len(data.get("frames", []))
        return {"present": True, "path": str(path), "n_poses": n}
    except (json.JSONDecodeError, OSError) as e:
        return {"present": True, "path": str(path), "error": str(e)}


def depth_sidecar_summary(video: Path) -> dict[str, Any]:
    """LiDAR depth, if present as a sidecar .npz or a directory of frames.
    Never required -- monodepth is the fallback the rest of the pipeline
    already uses (see videos/README.md)."""
    npz_path = _find_sidecar(video, DEPTH_SUFFIXES)
    if npz_path is not None:
        return {"present": True, "kind": "npz", "path": str(npz_path)}
    dir_path = _find_sidecar_dir(video, DEPTH_DIR_SUFFIXES)
    if dir_path is not None:
        n = len(list(dir_path.iterdir()))
        return {"present": True, "kind": "dir", "path": str(dir_path), "n_files": n}
    return {"present": False}


def marker_sidecar_summary(video: Path) -> dict[str, Any]:
    """Pre-computed calibration-marker observations from the capture app's own
    AR session (distinct from validate_video.py's own from-pixels ArUco
    detection, which needs no sidecar at all)."""
    path = _find_sidecar(video, MARKER_SUFFIXES)
    if path is None:
        return {"present": False}
    try:
        data = json.loads(path.read_text())
        n = len(data) if isinstance(data, list) else len(data.get("observations", []))
        return {"present": True, "path": str(path), "n_observations": n}
    except (json.JSONDecodeError, OSError) as e:
        return {"present": True, "path": str(path), "error": str(e)}


def declared_device_metadata_sidecar(video: Path) -> dict[str, Any]:
    path = _find_sidecar(video, DEVICE_METADATA_SUFFIXES)
    if path is None:
        return {"present": False}
    try:
        return {"present": True, "path": str(path), "data": json.loads(path.read_text())}
    except (json.JSONDecodeError, OSError) as e:
        return {"present": True, "path": str(path), "error": str(e)}


# ---------------------------------------------------------------------------
# Privacy / redaction pass (faces: wired up via cv2 Haar cascade; screens: TODO)
# ---------------------------------------------------------------------------

_FACE_DETECTOR = None


def _face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        if not FACE_CASCADE_PATH.exists():
            return None
        _FACE_DETECTOR = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    return _FACE_DETECTOR


def detect_faces(gray) -> list[tuple[int, int, int, int]]:
    det = _face_detector()
    if det is None or det.empty():
        return []
    boxes = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [tuple(int(v) for v in b) for b in boxes]


# TODO(redaction): screen/display redaction (laptop/TV/phone screens caught
# incidentally in a room scan) has no cheap, reliable detector wired up here.
# A bright-rectangle heuristic false-positives constantly on windows and
# reflective countertops. Leaving this as a documented gap rather than
# shipping a redaction pass that gives operators false confidence; revisit
# with a proper screen/display detector (e.g. a small trained classifier or
# an off-the-shelf display-detection model) before relying on it for
# anything privacy-sensitive.
def detect_screens(_gray) -> list[tuple[int, int, int, int]]:
    return []


def run_redaction_pass(video: Path, out_dir: Path | None, sample_stride_samples: int = 40,
                        blur_ksize: int = 51) -> dict[str, Any]:
    """Best-effort face redaction preview. Faces only (see TODO above for
    screens). Runs on the same sampled frames validate_video.py scores, at
    full working resolution for the blur preview.
    """
    samples, _fps, _dur, _total = sample_frames(
        video, samples_per_second=3.0, max_samples=sample_stride_samples, work_width=960)
    flagged: list[dict[str, Any]] = []
    if out_dir is not None:
        (out_dir / "redacted_preview").mkdir(parents=True, exist_ok=True)
    for s in samples:
        faces = detect_faces(s.gray)
        screens = detect_screens(s.gray)
        if not faces and not screens:
            continue
        flagged.append({"frame_index": s.index, "t": s.t, "n_faces": len(faces), "n_screens": len(screens)})
        if out_dir is not None:
            img = s.bgr_small.copy()
            k = blur_ksize | 1  # odd kernel required
            for (x, y, w, h) in faces:
                roi = img[y:y + h, x:x + w]
                if roi.size:
                    img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
            cv2.imwrite(str(out_dir / "redacted_preview" / f"frame_{s.index:06d}.jpg"), img)
    return {
        "faces_detector_available": _face_detector() is not None,
        "screens_detector_status": "TODO_not_implemented",
        "n_frames_sampled": len(samples),
        "n_frames_flagged": len(flagged),
        "flagged_frames": flagged,
    }


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------

def extract(video: Path, out_dir: Path | None = None, redact: bool = False) -> dict[str, Any]:
    dev_meta = device_metadata(video)
    bundle = {
        "video_path": str(video),
        "video_bytes": video.stat().st_size,
        "device_metadata": dev_meta,
        "device_metadata_sidecar": declared_device_metadata_sidecar(video),
        "capture_timestamps": capture_timestamps(video, dev_meta),
        "poses": pose_sidecar_summary(video),
        "depth": depth_sidecar_summary(video),
        "calibration_marker_sidecar": marker_sidecar_summary(video),
        "privacy": {
            "redaction_run": redact,
            "raw_video_retained": True,  # this pass never deletes/modifies the original
        },
    }
    if redact:
        bundle["privacy"]["redaction"] = run_redaction_pass(video, out_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(bundle, indent=2))
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--redact", action="store_true",
                     help="run best-effort face-redaction preview (screens: not implemented, see TODO)")
    args = ap.parse_args()
    bundle = extract(args.video, args.out, redact=args.redact)
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
