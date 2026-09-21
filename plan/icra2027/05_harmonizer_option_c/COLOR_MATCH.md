# Fixed color matching baseline

Implementation: `robo.rendering.color_match`, connected to the existing
`ObservationPipeline` as `variant="color_match"`. This is a deterministic
non-generative baseline for the five-way visual experiment; the canonical
policy treatment/reset ledger and frozen A0/A4 experiments are unchanged.

Fit one calibration per camera on a predeclared aligned construction TRAIN
view and its valid background mask. The E0-bound sequence producer must verify
the source view and reference provenance and exclude robot/inserted object
pixels from the calibration mask. The fitting API rejects a declared overlap
between construction and evaluation view IDs, but arbitrary caller-provided
arrays are not independently certified by a numerical utility.

For each RGB channel, with population mean/std on valid background pixels:

```
gain = reference_std / source_std if source_std > 0 else 1
offset = reference_mean - gain * source_mean
output = uint8(rint(clip(input * gain + offset, 0, 255)))
```

`rint` uses ties-to-even rounding. There are no scene-specific manually tuned
gains, thresholds or LUTs. A constant source channel uses the declared unit gain
and mean offset. Inputs must be uint8 RGB, aligned with a nonempty boolean mask.
The serialized calibration binds algorithm, camera, TRAIN/evaluation IDs,
source/reference/mask pixel hashes, all fitted statistics and a canonical digest.
Application validates the digest/camera and never fits parameters from the
current observation or evaluation reference.

Pass one frozen calibration for each `image_keys` entry through
`color_match_calibrations`. Pipeline initialization copies the records to prevent
external mutation; every application checks its identity. Images are transformed
without changing the source arrays or non-RGB observation fields. Per-camera
`transform_latency_ms` covers only the affine operation and must not be reported
as E5's end-to-end observation latency; the existing whole-observation timer is
reported separately. It is never labeled as Harmonizer or robot restoration.

Smoke command after the repository's existing headless environment setup:

```bash
python -m pytest -q tests/test_color_match.py tests/test_observation_pipeline.py \
  tests/test_harmony_planned_frames.py tests/test_harmony_visual_metrics.py \
  tests/test_demo_final.py --basetemp=outputs/e5-color-tests/pt1
```

Result: 48 passed in 9.79 seconds. Tests recover a known affine transform,
exclude foreground pixels, cover constant channels, reject split overlap,
empty masks, changed parameters/camera, and verify deterministic RGB-only
application across episodes. This is synthetic smoke, not a real visual result.
Pilot/full: NOT_RUN pending authenticated aligned sequences. No checkpoint,
model loading, GPU allocation or Slurm job is required for this numerical stage.
