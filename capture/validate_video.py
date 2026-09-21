"""Validate a phone-capture video BEFORE it enters reconstruction.

Scores a single RGB(+optional depth/pose) video against the coverage and
quality bar a casual scan needs for pose estimation + 3DGS training to work
(`videos/README.md`'s "slow, smooth sweep", "good texture", "show the
floor") and for downstream robot-alignment (a calibration fiducial visible
in several frames). Produces a JSON report with per-check scalar metrics
AND actionable, timestamped recapture messages -- never a bare pass/fail.

Deliberately video-only: this module reads pixels, ffprobe stream metadata,
and (optionally) device/pose sidecar metadata written by
`capture/extract_metadata.py`. It has no code path that can accept task
labels, object annotations, or per-object identity of any kind -- there are
none at capture time, and none should ever be threaded in here (see
`tests/test_capture_validation.py::test_no_task_label_input_path`, which
asserts this from the function signatures and source text, not just by
convention).

Usage:
    python -m capture.validate_video VIDEO.mp4 \
        --config configs/capture/phone_default.yaml \
        --out outputs/capture_validation/<name>

Writes <out>/report.json (machine-readable) and <out>/report.txt
(human-readable, the actionable recapture messages) and returns exit code 0
on PASS/WARN, 1 on FAIL (any critical issue) so callers can gate expensive
reconstruction on it (acceptance criterion: "validation completes before
expensive reconstruction").
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "capture" / "phone_default.yaml"


def load_config(path: Path | str | None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Small data types
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    code: str
    severity: str          # "critical" | "warning" | "info"
    message: str           # actionable, human-facing, timestamped where relevant
    start_s: float | None = None
    end_s: float | None = None
    metric: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mmss(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# ffprobe helpers (duration/fps/resolution; no external python video deps)
# ---------------------------------------------------------------------------

def probe_stream(video: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,"
         "nb_frames,duration,rotation:format=duration,tags",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return {"stream": stream, "format": fmt}


def parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str in ("0/0", "N/A"):
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate_str)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    index: int          # frame index in the source video
    t: float            # timestamp, seconds
    gray: np.ndarray    # resized grayscale, uint8
    bgr_small: np.ndarray  # resized BGR, for downstream color checks


def sample_frames(video: Path, samples_per_second: float, max_samples: int,
                   work_width: int) -> tuple[list[Sample], float, float, int]:
    """Sample frames at a roughly FIXED temporal rate (`samples_per_second`),
    capped at `max_samples` total for cost control on long clips.

    A fixed *rate* (not a fixed total count) matters here: every motion
    threshold below is calibrated on the time interval between samples
    (`Sample.t` deltas), so a 4s fixture and a 2-minute capture must be
    sampled at the same Hz or the same physical camera motion produces
    wildly different displacement-per-sample-pair numbers on the two clips
    (this bit an earlier version of this module, whose fixed-COUNT sampling
    put the 4s synthetic fixtures at ~15 Hz and the real 28s pilot clip at
    ~1.5 Hz). `max_samples` still bounds cost on long clips -- the effective
    rate degrades gracefully below `samples_per_second` there rather than
    growing per-video cost unboundedly.

    Returns (samples, fps, duration_s, total_frames). Resizes every sampled
    frame to `work_width` on the long side so every threshold in the config
    is resolution-independent (a 4K and a 720p phone clip score the same
    scene the same way).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[validate_video] could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total <= 0:
        info = probe_stream(video)
        fps = fps or parse_fps(info["stream"].get("avg_frame_rate")) or parse_fps(info["stream"].get("r_frame_rate"))
        total = total or int(info["stream"].get("nb_frames") or 0)
    duration = total / fps if fps > 0 else float(probe_stream(video)["format"].get("duration") or 0.0)

    step_from_rate = max(1, round(fps / samples_per_second)) if (fps > 0 and samples_per_second > 0) else 1
    step_from_cap = max(1, total // max_samples) if total > 0 else 1
    step = max(step_from_rate, step_from_cap)

    samples: list[Sample] = []
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            h, w = frame.shape[:2]
            scale = work_width / max(w, h)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA) if scale < 1 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            t = idx / fps if fps > 0 else float(idx)
            samples.append(Sample(index=idx, t=t, gray=gray, bgr_small=small))
        idx += 1
        ok, frame = cap.read()
    cap.release()
    if not samples:
        raise SystemExit(f"[validate_video] no frames decoded from {video}")
    return samples, fps, duration, total


# ---------------------------------------------------------------------------
# Per-frame / per-pair metrics
# ---------------------------------------------------------------------------

def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian: low value = flat/blurred, high = sharp/textured."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


_ORB = None
_BF_MATCHER = None


def _orb():
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=500)
    return _ORB


def _bf_matcher():
    global _BF_MATCHER
    if _BF_MATCHER is None:
        _BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)
    return _BF_MATCHER


@dataclass
class OrbFrame:
    keypoints: tuple
    descriptors: np.ndarray | None

    @property
    def count(self) -> int:
        return len(self.keypoints)


def detect_orb(gray: np.ndarray) -> OrbFrame:
    kps, des = _orb().detectAndCompute(gray, None)
    return OrbFrame(keypoints=tuple(kps), descriptors=des)


def baseline_px(frame_a: OrbFrame, frame_b: OrbFrame, min_good_matches: int) -> float | None:
    """Median pixel displacement of well-matched ORB keypoints between two
    sampled frames -- the angular/translation baseline proxy this validator
    uses in place of full SfM. Returns None when there are not enough
    confident correspondences to say anything (too few keypoints in either
    frame, or too few survive Lowe's ratio test): that itself is a signal
    (huge motion / a hard cut / severe blur), surfaced separately as
    NO_FRAME_OVERLAP rather than silently coerced into a flow number.

    Deliberately NOT dense optical flow (e.g. Farneback): sampled frames can
    be a large fraction of a second apart (see `sample_frames`), and dense
    flow's local search window saturates and silently UNDER-reports large
    displacements once they exceed it -- verified empirically against the
    fixtures in tests/data/phone/ (see capture/README.md), where it made a
    fast pan measure LOWER than a moderate one. Sparse feature matching has
    no such window; a correspondence is a correspondence at any offset.
    """
    if frame_a.descriptors is None or frame_b.descriptors is None:
        return None
    if frame_a.count < min_good_matches or frame_b.count < min_good_matches:
        return None
    matches = _bf_matcher().knnMatch(frame_a.descriptors, frame_b.descriptors, k=2)
    good = []
    for m in matches:
        if len(m) < 2:
            continue
        best, second = m
        if best.distance < 0.75 * second.distance:
            good.append(best)
    if len(good) < min_good_matches:
        return None
    disps = [
        float(np.hypot(frame_a.keypoints[m.queryIdx].pt[0] - frame_b.keypoints[m.trainIdx].pt[0],
                        frame_a.keypoints[m.queryIdx].pt[1] - frame_b.keypoints[m.trainIdx].pt[1]))
        for m in good
    ]
    return float(np.median(disps))


def frame_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Normalized cross-correlation-based similarity in [~-1, 1]; ~1.0 means
    near-identical frames (paused phone / duplicated frame)."""
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 1.0  # both frames flat/constant -> trivially identical
    return float((a * b).sum() / denom)


_ARUCO_DICT = None
_ARUCO_DETECTOR = None


def _aruco_detector(dict_name: str):
    global _ARUCO_DICT, _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        _ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, params)
    return _ARUCO_DETECTOR


def marker_ids(gray: np.ndarray, dict_name: str) -> list[int]:
    detector = _aruco_detector(dict_name)
    _corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [int(i) for i in ids.flatten()]


# ---------------------------------------------------------------------------
# Run-finding helper: collapse a per-sample boolean flag into contiguous runs
# ---------------------------------------------------------------------------

def find_runs(flags: list[bool], times: list[float]) -> list[tuple[int, int, float, float]]:
    """Return (start_idx, end_idx, start_t, end_t) for each maximal run of
    consecutive True flags."""
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            runs.append((i, j, times[i], times[j]))
            i = j + 1
        else:
            i += 1
    return runs


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_video(video_path: str | Path, config: dict[str, Any] | None = None,
                    device_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one capture video against `config` (see configs/capture/phone_default.yaml).

    Inputs are strictly: a video file path, a threshold config, and OPTIONAL
    device/pose metadata (camera model, LiDAR/AR pose availability -- see
    `capture/extract_metadata.py`). There is no parameter, and no code path
    below, that accepts task labels or per-object annotations of any kind.
    """
    video_path = Path(video_path)
    cfg = config or load_config(None)
    sc = cfg["sampling"]
    samples, fps, duration, total_frames = sample_frames(
        video_path, samples_per_second=sc["samples_per_second"], max_samples=sc["max_samples"],
        work_width=sc["work_width"])

    issues: list[Issue] = []
    mc = cfg["motion"]
    min_good_matches = mc["min_good_matches"]

    # ---- per-frame metrics -------------------------------------------------
    blur = [blur_score(s.gray) for s in samples]
    bright = [brightness(s.gray) for s in samples]
    orb_frames = [detect_orb(s.gray) for s in samples]
    feats = [of.count for of in orb_frames]
    marker_cfg = cfg["calibration_marker"]
    markers = [marker_ids(s.gray, marker_cfg["dictionary"]) for s in samples]
    times = [s.t for s in samples]

    # ---- per-pair metrics ---------------------------------------------------
    sims = [frame_similarity(samples[i - 1].gray, samples[i].gray) for i in range(1, len(samples))]
    bright_deltas = [abs(bright[i] - bright[i - 1]) for i in range(1, len(samples))]
    pair_times = [(times[i - 1] + times[i]) / 2 for i in range(1, len(samples))]
    pair_dts = [max(1e-3, times[i] - times[i - 1]) for i in range(1, len(samples))]
    baselines = [baseline_px(orb_frames[i - 1], orb_frames[i], min_good_matches) for i in range(1, len(samples))]
    # Motion thresholds are calibrated in px/SECOND, not raw px-per-sample-pair:
    # sample spacing varies (a capped-cost long clip samples sparser than a
    # short one), and a threshold on the raw per-pair displacement would
    # silently mean something different on every clip length.
    baselines_px_s = [(b / dt if b is not None else None) for b, dt in zip(baselines, pair_dts)]

    # ==== BLUR: severe_blur ====================================================
    bc = cfg["blur"]
    sharp_flags = [b >= bc["severe_blur_var_min"] for b in blur]
    sharp_fraction = sum(sharp_flags) / len(sharp_flags)
    for (i0, i1, t0, t1) in find_runs([not f for f in sharp_flags], times):
        run_dur = t1 - t0 if i1 > i0 else (duration / max(1, len(samples)))
        if run_dur >= bc["severe_blur_min_run_s"] or (i1 == i0 and len(samples) <= 3):
            worst = min(blur[i0:i1 + 1])
            issues.append(Issue(
                code="SEVERE_BLUR", severity="critical",
                message=(f"frames from {mmss(t0)}-{mmss(t1)} are severely blurred "
                         f"(sharpness {worst:.0f}, need >= {bc['severe_blur_var_min']:.0f}) "
                         f"- hold the phone steadier or slow the sweep through this section"),
                start_s=t0, end_s=t1, metric={"laplacian_var_min": worst}))
    if sharp_fraction < bc["min_sharp_fraction"] and not any(i.code == "SEVERE_BLUR" for i in issues):
        issues.append(Issue(
            code="SEVERE_BLUR", severity="critical",
            message=(f"only {sharp_fraction:.0%} of sampled frames are acceptably sharp "
                      f"(need >= {bc['min_sharp_fraction']:.0%}) - recapture more slowly"),
            metric={"sharp_fraction": sharp_fraction}))

    # ==== EXPOSURE: brightness jumps ==========================================
    ec = cfg["exposure"]
    for i, d in enumerate(bright_deltas):
        if d >= ec["max_brightness_delta"]:
            issues.append(Issue(
                code="EXPOSURE_JUMP", severity="warning",
                message=(f"exposure/white-balance jumped sharply at {mmss(pair_times[i])} "
                          f"(brightness change {d:.0f}/255) - avoid panning across bright "
                          f"windows/lights mid-scan"),
                start_s=pair_times[i], metric={"brightness_delta": d}))

    # ==== DUPLICATION: paused / repeated frames ===============================
    dc = cfg["duplication"]
    dup_flags = [s >= dc["similarity_min"] for s in sims]
    for (i0, i1, t0, t1) in find_runs(dup_flags, pair_times):
        run_dur = t1 - t0
        if (i1 - i0 + 1) >= dc["min_run_samples"]:
            issues.append(Issue(
                code="FRAME_DUPLICATION", severity="warning",
                message=(f"camera appears paused/frames duplicated for ~{run_dur:.1f}s "
                          f"starting {mmss(t0)} - keep the phone moving continuously, "
                          f"avoid pausing mid-scan"),
                start_s=t0, end_s=t1, metric={"similarity_min": min(sims[i0:i1 + 1])}))

    # ==== MOTION: static camera / fast pan (feature-baseline proxy, px/s) ====
    static_flags = [(b is not None and b <= mc["static_flow_px_s_max"]) for b in baselines_px_s]
    for (i0, i1, t0, t1) in find_runs(static_flags, pair_times):
        run_dur = t1 - t0
        if run_dur >= mc["static_min_run_s"]:
            worst = max(v for v in baselines_px_s[i0:i1 + 1] if v is not None)
            issues.append(Issue(
                code="STATIC_CAMERA", severity="critical",
                message=(f"camera was static for {run_dur:.1f}s starting at {mmss(t0)} "
                          f"- recapture with continuous motion (parallax is what makes "
                          f"reconstruction possible)"),
                start_s=t0, end_s=t1, metric={"baseline_px_s_max": worst}))

    fastpan_flags = [(b is not None and b >= mc["fast_pan_flow_px_s_min"]) for b in baselines_px_s]
    for (i0, i1, t0, t1) in find_runs(fastpan_flags, pair_times):
        worst = max(v for v in baselines_px_s[i0:i1 + 1] if v is not None)
        issues.append(Issue(
            code="FAST_PAN", severity="critical",
            message=(f"camera panned too fast around {mmss(t0)}-{mmss(t1)} "
                      f"(baseline {worst:.0f}px/s, threshold "
                      f"{mc['fast_pan_flow_px_s_min']:.0f}px/s) - frames may fail to "
                      f"register; slow down the sweep through this section"),
            start_s=t0, end_s=t1, metric={"baseline_px_s_max": worst}))

    # A pair with too few confident feature matches to say anything: usually
    # an even more extreme version of "too fast" (or a hard scene cut, or
    # blur bad enough to also break ORB) -- reported distinctly rather than
    # silently dropped or folded into FAST_PAN, since the cause isn't
    # necessarily speed.
    unmatched_flags = [b is None for b in baselines_px_s]
    for (i0, i1, t0, t1) in find_runs(unmatched_flags, pair_times):
        issues.append(Issue(
            code="NO_FRAME_OVERLAP", severity="critical",
            message=(f"could not match enough visual features between sampled frames "
                      f"around {mmss(t0)}-{mmss(t1)} (usually a very fast pan, a hard cut, "
                      f"or blur severe enough to lose all texture) - slow down and make "
                      f"sure consecutive views overlap"),
            start_s=t0, end_s=t1, metric={"n_pairs": i1 - i0 + 1}))

    # ==== FEATURES: low-texture coverage ======================================
    fc = cfg["features"]
    low_feat_flags = [k < fc["min_keypoints"] for k in feats]
    low_feat_fraction = sum(low_feat_flags) / len(low_feat_flags)
    if low_feat_fraction >= fc["max_low_feature_fraction"]:
        runs = find_runs(low_feat_flags, times)
        worst_run = max(runs, key=lambda r: r[3] - r[2]) if runs else None
        loc = f" (worst around {mmss(worst_run[2])}-{mmss(worst_run[3])})" if worst_run else ""
        issues.append(Issue(
            code="LOW_FEATURE_COVERAGE", severity="warning",
            message=(f"{low_feat_fraction:.0%} of sampled frames have very few visual "
                      f"features (< {fc['min_keypoints']} keypoints){loc} - reframe to "
                      f"include textured surfaces, avoid blank walls/ceiling as the only content"),
            metric={"low_feature_fraction": low_feat_fraction}))

    # ==== CALIBRATION MARKER =====================================================
    n_marker_frames = sum(1 for m in markers if m)
    if n_marker_frames < marker_cfg["min_marker_frames"]:
        issues.append(Issue(
            code="NO_CALIBRATION_MARKER", severity="warning",
            message=(f"calibration fiducial ({marker_cfg['dictionary']}) detected in only "
                      f"{n_marker_frames}/{len(samples)} sampled frames (need >= "
                      f"{marker_cfg['min_marker_frames']}) - place a printed ArUco marker "
                      f"near the robot base/workspace and make sure it stays visible across "
                      f"several frames of the interaction-workspace pass"),
            metric={"marker_frames": n_marker_frames, "total_samples": len(samples)}))

    # ==== DURATION ================================================================
    dur_cfg = cfg["duration"]
    if duration < dur_cfg["min_duration_s"]:
        issues.append(Issue(
            code="SHORT_DURATION", severity="warning",
            message=(f"clip is {duration:.1f}s, shorter than the recommended "
                      f"{dur_cfg['min_duration_s']:.0f}s two-pass minimum (room/context loop "
                      f"+ object-height interaction-workspace loop) - consider a longer scan"),
            metric={"duration_s": duration}))
    elif duration > dur_cfg["max_duration_s"]:
        issues.append(Issue(
            code="LONG_DURATION", severity="info",
            message=(f"clip is {duration:.1f}s, longer than the recommended "
                      f"{dur_cfg['max_duration_s']:.0f}s ceiling - fine, just costs more "
                      f"reconstruction compute; consider trimming dead time"),
            metric={"duration_s": duration}))

    # ---- overall verdict ------------------------------------------------------
    severities = {i.severity for i in issues}
    if "critical" in severities:
        verdict = "FAIL"
    elif "warning" in severities:
        verdict = "WARN"
    else:
        verdict = "PASS"

    valid_baselines = [b for b in baselines_px_s if b is not None]
    report = {
        "video": str(video_path),
        "verdict": verdict,
        "duration_s": duration,
        "fps": fps,
        "total_frames": total_frames,
        "n_samples": len(samples),
        "device_metadata_present": device_metadata is not None,
        "device_metadata": device_metadata or {},
        "metrics": {
            "blur_laplacian_var": {"min": min(blur), "mean": float(np.mean(blur)), "max": max(blur)},
            "brightness": {"min": min(bright), "mean": float(np.mean(bright)), "max": max(bright)},
            "feature_count_orb": {"min": min(feats), "mean": float(np.mean(feats)), "max": max(feats)},
            "baseline_px_per_s": (
                {"min": min(valid_baselines), "mean": float(np.mean(valid_baselines)), "max": max(valid_baselines)}
                if valid_baselines else {"min": None, "mean": None, "max": None}),
            "n_unmatched_pairs": sum(unmatched_flags),
            "n_pairs": len(baselines_px_s),
            "frame_similarity": (
                {"min": min(sims), "mean": float(np.mean(sims)), "max": max(sims)}
                if sims else {"min": 1.0, "mean": 1.0, "max": 1.0}),
            "calibration_marker_frames": n_marker_frames,
            "sharp_fraction": sharp_fraction,
            "low_feature_fraction": low_feat_fraction,
        },
        "issues": [i.to_dict() for i in issues],
    }
    return report


# ---------------------------------------------------------------------------
# Report writing / CLI
# ---------------------------------------------------------------------------

def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    lines = [
        f"capture validation: {report['video']}",
        f"verdict: {report['verdict']}  "
        f"(duration {report['duration_s']:.1f}s, {report['n_samples']} samples scored, "
        f"{report['total_frames']} total frames @ {report['fps']:.1f}fps)",
        "",
    ]
    if not report["issues"]:
        lines.append("no issues found.")
    for issue in report["issues"]:
        lines.append(f"[{issue['severity'].upper():>8}] {issue['code']}: {issue['message']}")
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None,
                     help="optional device/pose metadata JSON from extract_metadata.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device_metadata = json.loads(args.metadata.read_text()) if args.metadata else None
    report = validate_video(args.video, cfg, device_metadata=device_metadata)
    write_report(report, args.out)
    print((args.out / "report.txt").read_text())
    sys.exit(0 if report["verdict"] != "FAIL" else 1)


if __name__ == "__main__":
    main()
