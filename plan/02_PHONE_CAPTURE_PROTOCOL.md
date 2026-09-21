# Task 02 — Phone Capture Protocol and Capture-Side Tooling

**Priority:** P0  
**Suggested owner:** mobile capture / data engineer  
**Depends on:** Task 00  
**Blocks:** Tasks 03, 04, 17

## Objective
Turn the central intuition into a repeatable user procedure: a casual 60–120 second phone scan must capture sufficient parallax, task objects, supports, obstacles, and robot-alignment evidence without task-specific object clicks.

## Existing code
- `run/run_video2sim.sh`: current video-to-simulator launcher.
- `agents/recon/frames.py`: frame extraction.
- `videos/`: current drop-folder interface.

## Outputs
- `capture/README.md`
- `capture/validate_video.py`
- `capture/extract_metadata.py`
- `configs/capture/phone_default.yaml`
- `tests/test_capture_validation.py`

The capture bundle must contain the original video, device metadata, optional RGB-D/AR poses, calibration-marker observations, capture timestamps, privacy/redaction status, and a validation report.

## Implementation steps
1. Specify a two-pass route: room/context loop, then object-height interaction-workspace loop.
2. Define minimum coverage: task objects visible in multiple views, both sides where possible, complete support surfaces, robot base or fiducial in several frames.
3. Compute blur, exposure jumps, frame duplication, angular/translation baseline, feature count, and provisional coverage warnings.
4. Support plain RGB as the baseline; log LiDAR depth and ARKit/ARCore poses when available without making them mandatory.
5. Provide actionable recapture messages, not only scalar scores.
6. Add optional automatic face/screen redaction while retaining raw encrypted input for authorized processing.

## Tests
- Clean test video passes.
- Fast pan, static camera, severe blur, and missing calibration marker each trigger the expected warning.
- Validation never reads task labels or per-object annotations.

## Acceptance criteria
- [ ] A new operator can follow one page of instructions without developer intervention.
- [ ] Validation completes before expensive reconstruction.
- [ ] Capture statistics populate the experiment manifest.
- [ ] User-owned phone-room captures can be ingested unchanged by Task 03.

## Paper artifact unlocked
The first arrow in Fig. 1, capture ablation table, and defensible “casual phone scan” wording.
