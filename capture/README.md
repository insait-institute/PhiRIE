# capture/ — phone-scan capture protocol and pre-reconstruction validation

This is the page a new operator follows to turn a phone into a usable
capture bundle for `run/run_video2sim.sh`, and the tool that checks the
bundle BEFORE the (expensive) reconstruction pipeline runs. It builds on
`videos/README.md`'s drop-folder contract (same limits: rigid objects only,
monodepth-scale RGB, floor must be visible) and adds what that page doesn't
cover: a repeatable two-pass filming procedure, and a validator that scores
blur/motion/coverage before you spend GPU-hours reconstructing a clip that
was never going to register.

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

The validator does not (and cannot, from pixels alone) verify you actually
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
baseline — it is exactly what `run/run_video2sim.sh` consumes.

## 3. Running the validator

```bash
.venv/bin/python -m capture.validate_video path/to/clip.mp4 \
    --config configs/capture/phone_default.yaml \
    --out outputs/capture_validation/<name>

# device metadata + capture timestamps + privacy status alongside it:
.venv/bin/python -m capture.extract_metadata path/to/clip.mp4 \
    --out outputs/capture_validation/<name> [--redact]
```

Both commands are also exposed as the `capture` module of the `phiroom` CLI
(`bash run/phiroom.sh plan capture validate -- --help`), which runs them in
the CPU control environment.

Exit code is `0` unless a `critical` issue fired (usable as a pipeline gate
before `run/run_video2sim.sh`). `report.json` is the machine-readable form
(every scalar metric plus the issue list); `report.txt` is the
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
metadata**. There is no parameter and no code path in `validate_video.py`
that can accept a task label or a per-object annotation — there are none at
capture time, by construction.

### Why px/second, and why ORB matching instead of dense optical flow

Samples are taken at a **fixed rate** (`sampling.samples_per_second`, capped
in count for long clips via `sampling.max_samples`), not a fixed count —
a 4-second clip and a 2-minute capture have to be sampled at the same Hz
or the same physical camera motion looks like wildly different numbers on
the two clips. And the motion baseline itself is **median ORB-keypoint-match
displacement**, not dense optical flow (e.g. Farneback): dense flow's local
search window silently *underestimates* large displacements once they
exceed it, which in early testing made a deliberately-fast synthetic pan
measure LOWER than a moderate one. Sparse feature matching has no such
window — a correspondence is a correspondence at any pixel offset — and
"too few confident matches to say anything" becomes its own honest signal
(`NO_FRAME_OVERLAP`) instead of a silently wrong number.

## 4. Thresholds and interpreting a report

The thresholds in `configs/capture/phone_default.yaml` are expressed in
resolution-independent units (metrics are computed on frames resized to
`sampling.work_width` on the long side; motion in px/s of median ORB-match
displacement; blur as variance-of-Laplacian). `static_flow_px_s_max=5` and
`fast_pan_flow_px_s_min=220` separate a held phone, a moderate sweep and a
fast pan; `severe_blur_var_min=40` separates sharp frames from motion blur.
Copy the file and adjust it for a different phone or scene if needed.

A typical single continuous handheld sweep (30 s, no marker) comes back as
`FAIL`: a brief `SEVERE_BLUR` window, a few `EXPOSURE_JUMP` warnings,
scattered `FAST_PAN` / `NO_FRAME_OVERLAP` findings, `NO_CALIBRATION_MARKER`
and `SHORT_DURATION` against the 60 s two-pass recommendation. That is the
intended behaviour: the point of the validator is to say this BEFORE
reconstruction spends GPU-hours on the clip, and to give the operator
concrete, timestamped notes rather than one scalar score. Loosening a
threshold to make a clip pass defeats the purpose.

## 5. Redaction (optional, best-effort)

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
