"""Tests for capture/validate_video.py (+ a light smoke test of
capture/extract_metadata.py).

Run: .venv/bin/python -m pytest -q tests/test_capture_validation.py

Fixtures (tiny, ~4s synthetic clips + the one real pilot clip) are
regenerated on demand by tests/data/phone/make_fixtures.py -- see that
module's docstring for how each one is built and why (band-limited noise
canvas, not mandelbrot/checkerboard; single-pass ffmpeg encode, not a
two-pass crop-then-overlay -- both were tried first and rejected for
corrupting the exact signal the fixture needs to isolate).
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import extract_metadata, validate_video  # noqa: E402
from tests.data.phone import make_fixtures  # noqa: E402

PILOT_CLIP = ROOT / "videos" / "pilot_a29cccc784.mp4"
CONFIG = validate_video.load_config(None)


def _codes(report: dict) -> set[str]:
    return {i["code"] for i in report["issues"]}


@pytest.fixture(scope="module")
def fixture_paths() -> dict[str, Path]:
    return make_fixtures.make_all()


# ---------------------------------------------------------------------------
# FAST_VALIDATION_MATRIX: each synthetic fixture trips its OWN distinct code
# ---------------------------------------------------------------------------

def test_static_camera_triggers_static_camera_only(fixture_paths):
    report = validate_video.validate_video(fixture_paths["static_camera"], CONFIG)
    codes = _codes(report)
    assert "STATIC_CAMERA" in codes, codes
    # distinct from the other three failure modes
    assert "FAST_PAN" not in codes
    assert "SEVERE_BLUR" not in codes
    assert "NO_CALIBRATION_MARKER" not in codes  # this fixture carries the marker


def test_fast_pan_triggers_fast_pan_only(fixture_paths):
    report = validate_video.validate_video(fixture_paths["fast_pan"], CONFIG)
    codes = _codes(report)
    assert "FAST_PAN" in codes, codes
    assert "STATIC_CAMERA" not in codes
    assert "SEVERE_BLUR" not in codes
    assert "NO_CALIBRATION_MARKER" not in codes  # this fixture carries the marker


def test_severe_blur_triggers_severe_blur(fixture_paths):
    report = validate_video.validate_video(fixture_paths["severe_blur"], CONFIG)
    codes = _codes(report)
    assert "SEVERE_BLUR" in codes, codes
    assert "STATIC_CAMERA" not in codes
    assert "FAST_PAN" not in codes
    # NOTE: this fixture has no marker (a sharp marker composited on a
    # heavily-blurred frame breaks the whole-frame blur statistic -- see
    # make_severe_blur()'s docstring), so NO_CALIBRATION_MARKER is an
    # expected secondary warning here, not asserted absent.


def test_no_marker_triggers_no_calibration_marker_only(fixture_paths):
    report = validate_video.validate_video(fixture_paths["no_marker"], CONFIG)
    codes = _codes(report)
    assert "NO_CALIBRATION_MARKER" in codes, codes
    assert "STATIC_CAMERA" not in codes
    assert "FAST_PAN" not in codes
    assert "SEVERE_BLUR" not in codes


def test_four_failure_modes_are_pairwise_distinguishable(fixture_paths):
    """The FAST_VALIDATION_MATRIX negative check: four different synthetic
    problems must not all collapse onto one generic "bad video" flag."""
    primary_code = {
        "static_camera": "STATIC_CAMERA",
        "fast_pan": "FAST_PAN",
        "severe_blur": "SEVERE_BLUR",
        "no_marker": "NO_CALIBRATION_MARKER",
    }
    # severe_blur is deliberately built WITHOUT the calibration marker (a sharp
    # marker composited on a heavily-blurred frame breaks the whole-frame blur
    # statistic the SEVERE_BLUR check needs -- see make_severe_blur()'s
    # docstring), so it also legitimately trips NO_CALIBRATION_MARKER. That
    # single documented overlap is allowed; every other cross-pairing must not
    # collapse onto another fixture's primary code.
    allowed_overlaps = {("no_marker", "severe_blur")}

    reports = {name: validate_video.validate_video(path, CONFIG) for name, path in fixture_paths.items()}
    fired = {name: _codes(r) for name, r in reports.items()}
    for name, code in primary_code.items():
        assert code in fired[name], f"{name} did not trigger {code}: {fired[name]}"
    # each fixture's primary code is absent from every OTHER fixture's issues,
    # except the one documented overlap above
    for name, code in primary_code.items():
        for other in primary_code:
            if other == name or (name, other) in allowed_overlaps:
                continue
            assert code not in fired[other], (
                f"{code} (primary for {name}) also fired on {other}: {fired[other]}")


# ---------------------------------------------------------------------------
# Real pilot clip: report exactly what it triggers, don't force a pass
# ---------------------------------------------------------------------------

def test_pilot_clip_validates_without_crashing():
    assert PILOT_CLIP.exists(), f"expected real fixture at {PILOT_CLIP}"
    report = validate_video.validate_video(PILOT_CLIP, CONFIG)
    # structural sanity, not a pass/fail assertion -- see module docstring
    assert report["verdict"] in ("PASS", "WARN", "FAIL")
    assert report["n_samples"] > 0
    assert 27.0 < report["duration_s"] < 29.0   # known: 28.2s
    assert 14.0 < report["fps"] < 16.0           # known: 15fps
    for issue in report["issues"]:
        assert issue["code"] and issue["severity"] in ("critical", "warning", "info")
        assert issue["message"]  # actionable text, never a bare code


def test_validate_video_cli_runs_before_expensive_reconstruction(tmp_path):
    """Acceptance criterion: validation completes fast, standalone, and its
    exit code is usable as a gate (0 unless a critical issue fired)."""
    import subprocess
    import time
    out_dir = tmp_path / "report"
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "capture.validate_video", str(PILOT_CLIP),
         "--config", str(ROOT / "configs" / "capture" / "phone_default.yaml"),
         "--out", str(out_dir)],
        cwd=ROOT, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.txt").exists()
    assert elapsed < 60, f"validation took {elapsed:.1f}s -- too slow to gate reconstruction with"
    assert proc.returncode in (0, 1)  # 1 iff verdict == FAIL, still a clean run


# ---------------------------------------------------------------------------
# Validation never reads task labels or per-object annotations
# ---------------------------------------------------------------------------

BANNED_IDENTIFIER_SUBSTRINGS = ("label", "annotation", "task_", "object_id", "object_class",
                                 "instance_id", "ground_truth", "gt_")


def test_validate_video_signature_has_no_task_label_input():
    sig = inspect.signature(validate_video.validate_video)
    assert set(sig.parameters) == {"video_path", "config", "device_metadata"}, sig.parameters
    # `config` is thresholds (see configs/capture/phone_default.yaml), `device_metadata`
    # is camera/pose/depth info from extract_metadata.py -- neither can carry a task
    # label or a per-object annotation, since capture time has none to give it.


def test_capture_module_source_has_no_task_label_identifiers():
    """Static check over every identifier (function args, assigned names) in
    both capture modules: none of them could be a task-label / annotation
    channel. Walks the AST rather than grepping raw text so it does not
    false-positive on this file's own prose docstrings explaining the
    guarantee (e.g. "no task labels or per-object annotations")."""
    for mod_path in (ROOT / "capture" / "validate_video.py", ROOT / "capture" / "extract_metadata.py"):
        tree = ast.parse(mod_path.read_text())
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
        offenders = {
            ident for ident in identifiers
            if any(bad in ident.lower() for bad in BANNED_IDENTIFIER_SUBSTRINGS)
        }
        assert not offenders, f"{mod_path.name} has suspicious identifiers: {offenders}"


# ---------------------------------------------------------------------------
# extract_metadata.py: light smoke test (device metadata + optional redaction)
# ---------------------------------------------------------------------------

def test_extract_metadata_basic_on_pilot_clip(tmp_path):
    bundle = extract_metadata.extract(PILOT_CLIP, out_dir=tmp_path, redact=False)
    assert bundle["device_metadata"]["width"] == 1168
    assert bundle["device_metadata"]["height"] == 778
    assert bundle["poses"] == {"present": False}     # no ARKit/ARCore sidecar for this clip
    assert bundle["depth"] == {"present": False}      # no LiDAR sidecar for this clip
    assert bundle["privacy"]["raw_video_retained"] is True
    assert (tmp_path / "metadata.json").exists()


def test_extract_metadata_redaction_pass_runs(fixture_paths, tmp_path):
    # Tiny fixture, no faces in it -- this only exercises that the
    # (best-effort) redaction pass runs end-to-end without crashing.
    bundle = extract_metadata.extract(fixture_paths["no_marker"], out_dir=tmp_path, redact=True)
    redaction = bundle["privacy"]["redaction"]
    assert redaction["screens_detector_status"] == "TODO_not_implemented"
    assert redaction["n_frames_sampled"] > 0
    assert redaction["n_frames_flagged"] == 0  # no faces in synthetic noise
