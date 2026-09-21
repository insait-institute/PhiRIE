# Task 03 — Robust Phone Video to Metric Reconstruction Front End

**Priority:** P0  
**Suggested owner:** 3D reconstruction engineer  
**Depends on:** Tasks 01–02  
**Blocks:** Tasks 04–06, 17

## Objective
Harden `run/run_video2sim.sh` into a reproducible front end producing posed frames, metric scale evidence, a Gaussian representation, a collision-capable mesh, and diagnostics from arbitrary accepted phone captures.

## Existing code
- `run/run_video2sim.sh`
- `agents/recon/frames.py`, `colmap_poses.py`, `metricize.py`, `make_scene_dir.py`, `gsplat_train.py`
- `models/vggt_scene.py`
- `agents/discover/derive_mesh_from_splat.py`

## Outputs
- `agents/recon/phone_pipeline.py`
- `configs/recon/phone_default.yaml`
- `run/run_phone2sim.sh` as a thin, documented wrapper
- `tests/test_phone_recon_contract.py`

## Implementation steps
1. Replace hidden environment defaults with a typed config while preserving current shell compatibility.
2. Make COLMAP/HLoc the primary pose path and record fallback to VGGT/Omega explicitly.
3. Separate arbitrary-scale reconstruction from metricization; retain uncertainty and source of scale.
4. Export train/validation views and held-out novel-view metrics.
5. Produce TSDF/mesh quality diagnostics: watertightness is not required, but floor/support coverage and collision decimation must be measured.
6. Add resume-safe stage hashes so changed inputs invalidate only dependent stages.
7. Emit one build manifest conforming to Task 01.

## Tests
- Existing phone pilot reproduces the prior scene outputs within stated tolerance.
- Interrupted run resumes without reusing stale outputs after a config change.
- COLMAP failure invokes the declared fallback and marks the build, rather than silently continuing.
- Unit scale is preserved through scene-dir emulation and splat-derived mesh extraction.

## Acceptance criteria
- [ ] At least three heterogeneous phone videos reach automatic instance discovery.
- [ ] Every reconstruction has train/held-out reprojection and scale diagnostics.
- [ ] No ScanNet++ annotation or path is logically required despite layout compatibility.

## Paper artifact unlocked
End-to-end phone build time/success and capture degradation ablations.
