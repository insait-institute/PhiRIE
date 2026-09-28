# capture/ — phone-scan capture protocol and pre-reconstruction validation

This is the one page a new operator follows to turn a phone into a usable
capture bundle for `run/run_video2sim.sh`, and the tool that checks the
bundle BEFORE the (expensive) reconstruction pipeline runs. It builds on
`videos/README.md`'s existing drop-folder contract (same limits: rigid
objects only, monodepth-scale RGB, floor must be visible) and adds what
that doc doesn't cover: a repeatable two-pass filming procedure, and a
validator that scores blur/motion/coverage before you spend GPU-hours
reconstructing a clip that was never going to register.

## 1. What to film: the two-pass route

Film in **two passes, back to back, in one continuous recording** (don't
stop the recording between them — a single clip is what the rest of the
pipeline consumes):

**Pass 1 — room/context loop.** Walk a slow arc or full loop around the
room at roughly standing height, keeping the whole workspace in view.
Purpose: give the reconstruction wide-baseline context — walls, the floor
(mandatory: metric scale RANSACs the floor plane), and where the workspace
sits in the room.

**Pass 2 — object-height interaction-workspace loop.** Lower the phone to
roughly the height a robot's camera/gripper would work at, and slowly orbit
the specific objects, supports, and obstacles the task cares about. For
each task-relevant object:
- get **both sides** where the geometry allows (walk around it, don't just
  pan across the front),
- keep the **full support surface** (the table/shelf/floor patch under it)
  in frame at some point, not just the object floating mid-frame,
- if there's a robot base or a printed fiducial marking robot-alignment,
  make sure it's visible in **several frames** of this pass, not just one.

This validator does not (and cannot, from pixels alone) verify you actually
did two passes or hit every object twice — that's still on the operator.
What it CAN check is the coverage proxies that a two-pass scan should
produce: no long static/frozen segments, no pans too fast to register, a
sharp majority of frames, and (if you used one) the calibration marker
showing up in enough sampled frames. See §3.

## 2. The capture bundle

A capture bundle is one video plus whatever of the following exists next to
it — **only the video is required**, everything else is optional and
logged, never blocking:

| piece | how it's supplied | read by |
|---|---|---|
| original video | `<name>.mp4` | both modules |
| device metadata | ffprobe container tags (rotation, creation_time, make/model if the capture app wrote them), or an optional `<name>.metadata.json` / `<name>_metadata.json` sidecar | `extract_metadata.py` |
| RGB-D / LiDAR depth | optional `<name>.depth.npz` or `<name>_depth/` directory | `extract_metadata.py` |
| ARKit/ARCore poses | optional `<name>.poses.jsonl` or `<name>_poses.json` | `extract_metadata.py` |
| calibration-marker observations | detected directly from pixels (ArUco, no sidecar needed) by `validate_video.py`; OR a precomputed `<name>.markers.json` / `<name>_markers.json` from the capture app's own AR session, logged by `extract_metadata.py` | both |
| capture timestamps | container `creation_time` tag + filesystem mtime | `extract_metadata.py` |
| privacy/redaction status | `extract_metadata.py --redact` (best-effort face blur preview; screens: not implemented, see TODO in that file) | `extract_metadata.py` |
| validation report | `report.json` / `report.txt` | `validate_video.py` |

Plain RGB with none of the optional sidecars is the fully-supported
baseline — this is what `videos/pilot_a29cccc784.mp4` is, and what
`run/run_video2sim.sh` already consumes unchanged.

## 3. Running the validator

```bash
.venv/bin/python -m capture.validate_video path/to/clip.mp4 \
    --config configs/capture/phone_default.yaml \
    --out outputs/capture_validation/<name>

# device metadata + capture timestamps + privacy status alongside it:
.venv/bin/python -m capture.extract_metadata path/to/clip.mp4 \
    --out outputs/capture_validation/<name> [--redact]
```

Exit code is `0` unless a `critical` issue fired (usable as a CI/pipeline
gate before `run/run_video2sim.sh`). `report.json` is the machine-readable
form (every scalar metric plus the issue list); `report.txt` is the
human-readable one (what an operator reads to decide whether to reshoot).

### What it checks (from pixels + container metadata only)

- **blur** — variance-of-Laplacian per sampled frame -> `SEVERE_BLUR`
- **exposure jumps** — frame-to-frame mean-brightness delta -> `EXPOSURE_JUMP`
- **frame duplication** — normalized cross-correlation between consecutive
  samples -> `FRAME_DUPLICATION` (a paused/frozen phone)
- **motion baseline** — median pixel displacement of confidently-matched
  ORB keypoints between sampled frames, normalized to px/second ->
  `STATIC_CAMERA` (too little) / `FAST_PAN` (too much) / `NO_FRAME_OVERLAP`
  (too few confident matches to say anything at all — usually an even more
  extreme pan, a hard cut, or blur bad enough to lose all texture)
- **feature coverage** — ORB keypoint count per frame -> `LOW_FEATURE_COVERAGE`
  (blank walls/ceiling dominating the clip)
- **calibration marker** — ArUco (`DICT_4X4_50` by default) detections
  across sampled frames -> `NO_CALIBRATION_MARKER`
- **duration** — against the two-pass recommendation -> `SHORT_DURATION` /
  `LONG_DURATION`

Every issue carries a **severity** (`critical` / `warning` / `info`), a
**timestamp** (or range), and an **actionable message** — never a bare
score. Example: `"camera was static for 3.2s starting at 00:14 - recapture
with continuous motion"`, not `motion_score: 0.02`.

It is **strictly video + container-metadata + optional device/pose
metadata**. There is no parameter and no code path anywhere in
`validate_video.py` that can accept a task label or a per-object
annotation — there are none at capture time, by construction (see
`tests/test_capture_validation.py`'s static checks over the function
signature and the module's identifiers).

### Why px/second, and why ORB matching instead of dense optical flow

Samples are taken at a **fixed rate** (`sampling.samples_per_second`, capped
in count for long clips via `sampling.max_samples`), not a fixed count —
a 4-second fixture and a 2-minute capture have to be sampled at the same Hz
or the same physical camera motion looks like wildly different numbers on
the two clips. And the motion baseline itself is **median ORB-keypoint-match
displacement**, not dense optical flow (e.g. Farneback): dense flow's local
search window silently *underestimates* large displacements once they
exceed it, which in early testing made a deliberately-fast synthetic pan
measure LOWER than a moderate one. Sparse feature matching has no such
window — a correspondence is a correspondence at any pixel offset — and
"too few confident matches to say anything" becomes its own honest signal
(`NO_FRAME_OVERLAP`) instead of a silently wrong number.

## 4. How the fixtures were calibrated

`tests/data/phone/make_fixtures.py` builds four ~4s/960x540/15fps synthetic
clips, all crop-pans across the same band-limited random-noise canvas (full
gradient everywhere, no repeating period, no flat regions — chosen
specifically because `mandelbrot`/checkerboard sources have both, which
confounds exactly the flow/feature signals below):

| fixture | what's different | primary code it must trigger |
|---|---|---|
| `static_camera.mp4` | zero pan (fixed crop, held) | `STATIC_CAMERA` |
| `fast_pan.mp4` | 900 canvas-px/s pan | `FAST_PAN` |
| `severe_blur.mp4` | 180 px/s pan + `gblur=sigma=4` | `SEVERE_BLUR` |
| `no_marker.mp4` | 180 px/s pan, no ArUco marker composited | `NO_CALIBRATION_MARKER` |

`static_camera` and `fast_pan` also get a generated ArUco marker composited
into a fixed corner so their motion issue is the only thing flagged.
`severe_blur` and `no_marker` both go without a marker — a marker overlay
on top of a heavily-blurred frame was tried and rejected: variance-of-
Laplacian is a whole-frame statistic, and one small sharp high-contrast
patch dominates it enough to make the whole frame register as "sharp" (a
real severely-blurred phone clip legitimately would not have a crisply
readable marker either, so `severe_blur` intentionally also trips
`NO_CALIBRATION_MARKER` as a secondary, documented warning).

Measured against `configs/capture/phone_default.yaml`'s thresholds (px/s
baseline; blur is variance-of-Laplacian on a 640px-wide resize):

| fixture | baseline px/s (min/mean/max) | blur var (mean) |
|---|---|---|
| `static_camera` | 0 / 0 / 0 | 441 |
| `fast_pan` | 432 / 582 / 648 | 447 |
| `severe_blur` | 86 / 117 / 130 | 10 |
| `no_marker` | 86 / 117 / 130 | 102 |

`static_flow_px_s_max=5`, `fast_pan_flow_px_s_min=220` sit cleanly between
the static/moderate/fast bands above.

## 5. Interpreting a report: the real pilot clip

`videos/pilot_a29cccc784.mp4` (28.2s, 1168x778, 15fps h264 — the only real
phone clip in the repo right now, per `docs/ICRA_RESEARCH_CONTRACT.md`'s
`phone_deploy` track, n=1) is **not** force-tuned to pass. Run it and read
what it actually says:

```bash
.venv/bin/python -m capture.validate_video videos/pilot_a29cccc784.mp4 \
    --config configs/capture/phone_default.yaml --out /tmp/capture_validation_pilot
```

It comes back `FAIL`, honestly: one brief `SEVERE_BLUR` window (~00:24-25),
three `EXPOSURE_JUMP` warnings, extensive `FAST_PAN` / `NO_FRAME_OVERLAP`
findings scattered through most of the clip (this is a single continuous
handheld sweep, not the slower two-pass procedure above — a real
data point that the "casual scan" framing in the internal phone-capture protocol notes
is more aspirational than this one clip achieves), no `NO_CALIBRATION_MARKER`
(it never carried one), and `SHORT_DURATION` (28.2s vs. the 60s two-pass
recommendation). None of that was reason to loosen a threshold — the point
of this validator is to say this BEFORE reconstruction spends GPU-hours on
it, and to give the next capture operator (Task 17, ≥3-room target, still
open per the research contract) concrete, actionable notes rather than one
scalar score.

## 6. Redaction (optional, best-effort)

`capture/extract_metadata.py --redact` runs a Haar-cascade face detector
(`cv2.data.haarcascades`, bundled with opencv) over the same sampled frames
`validate_video.py` scores, and writes Gaussian-blurred previews to
`<out>/redacted_preview/`. The **original video is never modified** — this
is a review preview for an operator to check before raw footage leaves a
trusted boundary, not a guarantee. **Screen/display redaction is not
implemented** (see the `TODO(redaction)` in `extract_metadata.py`): a cheap
bright-rectangle heuristic false-positives constantly on windows and
countertops, and shipping it would give false confidence rather than real
privacy protection. This is a documented gap, not a silent one.
