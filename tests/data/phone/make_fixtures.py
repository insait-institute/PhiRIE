"""Deterministically synthesize the tiny (~4s, 960x540, 15fps) fixture clips
`tests/test_capture_validation.py` scores against: a static camera, a fast
pan, severe motion blur, and a clean-but-markerless pass. Regenerated on
first test-session use (idempotent -- skips a clip whose file already
exists) so re-running the suite never re-pays ffmpeg cost.

Each fixture is a crop-pan across the SAME band-limited random-noise canvas
(`_canvas()`), so ORB feature count stays high everywhere except where a
fixture deliberately destroys it (blur), and it is the pan SPEED / blur
sigma / marker presence -- not the content -- that drives the metric under
test. `static_camera` and `fast_pan` get the same generated ArUco marker
composited into a fixed screen corner so their motion issue is the only
thing flagged; `severe_blur` and `no_marker` both go without it (see
`make_severe_blur`'s docstring for why the marker and heavy blur don't mix),
so NO_CALIBRATION_MARKER is expected on both of THOSE, and isolates cleanly
against the two motion fixtures.

Crop/blur and the marker overlay are done in a SINGLE ffmpeg encode
(`_build`) rather than a two-pass crop-then-overlay: an earlier version of
this file re-encoded a heavily-blurred base clip through a second libx264
pass for the overlay, and the low-detail blurred content picked up gradient
banding/blocking from THAT second compression pass sharp enough to erase
the intended blur signal (variance-of-Laplacian recovered from ~11 back up
to ~360, indistinguishable from an unblurred clip). One encode avoids it.

Run directly to (re)build all fixtures:
    .venv/bin/python tests/data/phone/make_fixtures.py [--force]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "/usr/bin/ffmpeg"
HERE = Path(__file__).resolve().parent
CANVAS_W, CANVAS_H = 6000, 540   # wide strip so even the fast pan never saturates (clips) mid-fixture
CROP_W, CROP_H = 960, 540
FPS = 15
DUR_S = 4
MARKER_PNG = HERE / "_marker.png"
MARKER_POS = "20:20"
MARKER_SIZE = 220
CANVAS_PNG = HERE / "_canvas.png"
CRF = "14"  # near-lossless: these clips are seconds long, size is not a concern,
            # and a higher (lossier) CRF reintroduces the banding artifact above


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _make_marker() -> Path:
    if not MARKER_PNG.exists():
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(d, 0, MARKER_SIZE)
        # pad with a white border so the marker's own black border is not
        # touching the frame edge (helps detection at an angle / low light)
        padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(MARKER_PNG), padded)
    return MARKER_PNG


def _canvas() -> Path:
    """Wide strip of band-limited random texture, shared by every panning
    fixture. Deliberately NOT `mandelbrot`/checkerboard: those have large
    flat or perfectly-repetitive regions where a crop-pan produces near-zero
    optical flow and near-zero ORB features (the aperture problem / no
    gradient at all) -- confounding exactly the motion/feature signals this
    fixture set exists to isolate. Smoothed full-spectrum noise has a
    gradient everywhere and no repeating period, so a pure horizontal
    translation is trackable and it is the pan SPEED (not the content) that
    drives the flow/blur measurements below.
    """
    if not CANVAS_PNG.exists():
        rng = np.random.default_rng(1234)
        noise = rng.integers(0, 256, size=(CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.2)
        canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(CANVAS_PNG), canvas)
    return CANVAS_PNG


def _build(out: Path, pre_filter: str, with_marker: bool) -> None:
    """One ffmpeg encode: crop/blur/etc (`pre_filter`, applied to the looped
    canvas) then, if requested, an ArUco marker composited on top."""
    canvas = _canvas()
    inputs = ["-loop", "1", "-i", str(canvas)]
    if with_marker:
        inputs += ["-loop", "1", "-i", str(_make_marker())]
        filter_complex = f"[0:v]{pre_filter}[base];[base][1:v]overlay={MARKER_POS}[vout]"
    else:
        filter_complex = f"[0:v]{pre_filter}[vout]"
    _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
          *inputs, "-t", str(DUR_S), "-r", str(FPS),
          "-filter_complex", filter_complex, "-map", "[vout]",
          "-c:v", "libx264", "-crf", CRF, "-pix_fmt", "yuv420p", str(out)])


# Pan speeds (canvas px/s, BEFORE the validator's work_width resize) and blur
# sigma chosen empirically against this exact canvas (see capture/README.md
# "How the fixtures were calibrated" for the measured numbers):
#   fast: well above the fast_pan_flow_min gate in configs/capture/phone_default.yaml
#   moderate (blur, no-marker): comfortably between the static and fast-pan gates
#   BLUR_SIGMA: crushes variance-of-Laplacian ~40x below the sharp baseline
#     while leaving just enough coarse structure for optical flow to still
#     see the (moderate, not-static) pan -- a much larger sigma smears away
#     ALL trackable structure and the fixture degenerates into looking like
#     a static camera to the flow proxy, which is not the failure mode this
#     fixture is for.
FAST_PAN_PX_S = 900
MODERATE_PAN_PX_S = 180
BLUR_SIGMA = 4


def make_static(force: bool = False) -> Path:
    """Static crop of the canvas, held for the whole clip: zero camera
    motion, full texture/sharpness (so blur/feature checks stay clean and
    STATIC_CAMERA is the one thing this fixture triggers)."""
    out = HERE / "static_camera.mp4"
    if out.exists() and not force:
        return out
    _build(out, f"crop={CROP_W}:{CROP_H}:x=0:y=0", with_marker=True)
    return out


def make_fast_pan(force: bool = False) -> Path:
    """Rapid translation across the wide texture canvas: large frame-to-frame
    optical flow, well above any legitimate handheld sweep."""
    out = HERE / "fast_pan.mp4"
    if out.exists() and not force:
        return out
    max_x = CANVAS_W - CROP_W
    _build(out, f"crop={CROP_W}:{CROP_H}:x='min(t*{FAST_PAN_PX_S}\\,{max_x})':y=0", with_marker=True)
    return out


def make_severe_blur(force: bool = False) -> Path:
    """Moderate translation (not fast-pan by itself) with heavy Gaussian
    blur, no marker. (A sharp marker composited on TOP of a heavily blurred
    frame was tried and rejected: variance-of-Laplacian is a whole-frame
    statistic, and one small crisp high-contrast patch dominates it enough
    to make the whole frame register as "sharp" -- the same reason a real
    severely-blurred clip legitimately would not have a crisply-readable
    marker either, so this fixture also (correctly) triggers
    NO_CALIBRATION_MARKER as a secondary warning; SEVERE_BLUR is what it is
    asserted on.)"""
    out = HERE / "severe_blur.mp4"
    if out.exists() and not force:
        return out
    max_x = CANVAS_W - CROP_W
    pre = (f"crop={CROP_W}:{CROP_H}:x='min(t*{MODERATE_PAN_PX_S}\\,{max_x})':y=0,"
           f"gblur=sigma={BLUR_SIGMA}")
    _build(out, pre, with_marker=False)
    return out


def make_no_marker(force: bool = False) -> Path:
    """Clean moderate pan, good texture, no blur -- but no calibration
    marker anywhere in frame."""
    out = HERE / "no_marker.mp4"
    if out.exists() and not force:
        return out
    max_x = CANVAS_W - CROP_W
    _build(out, f"crop={CROP_W}:{CROP_H}:x='min(t*{MODERATE_PAN_PX_S}\\,{max_x})':y=0", with_marker=False)
    return out


FIXTURES = {
    "static_camera": make_static,
    "fast_pan": make_fast_pan,
    "severe_blur": make_severe_blur,
    "no_marker": make_no_marker,
}


def make_all(force: bool = False) -> dict[str, Path]:
    return {name: fn(force=force) for name, fn in FIXTURES.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if the file already exists")
    args = ap.parse_args()
    paths = make_all(force=args.force)
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
