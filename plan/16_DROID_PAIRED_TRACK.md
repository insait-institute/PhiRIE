# Task 16 — DROID Metric Reconstruction and Paired-Data Track

**Priority:** P1  
**Suggested owner:** DROID data engineer  
**Depends on:** Tasks 01, 03–10  
**Blocks:** broader real-data validation

## Objective
Use raw DROID episodes to reconstruct task workspaces in the robot-base frame, align policy videos/actions, and derive paired evidence wherever corresponding real success labels/checkpoints are trustworthy.

## Existing code
- `run/fetch_droid_raw.py`, `run/gcs_fetch.py`
- `run/run_droid_recon.sh`
- `agents/recon/droid_extract.py`
- `agents/recon/align_to_traj.py`

## Implementation steps
1. Freeze a stratified episode list by lab, task, success/failure, camera quality, and static-view availability before reconstruction.
2. Improve static-frame selection and robot/manipulated-object masking for splat training.
3. Validate FK/camera calibration and Umeyama alignment on held-out trajectory points.
4. Associate language, robot states, actions, and success metadata with the scene manifest.
5. Distinguish episode replay agreement from autonomous policy correlation.
6. Use DROID-Sim/RoboSnap assets only under license and release-compatible protocols.

## Outputs
- `configs/droid/frozen_episodes.yaml`
- `agents/recon/droid_static_select.py`
- `agents/eval/droid_alignment_eval.py`
- `docs/DROID_PROTOCOL.md`

## Acceptance criteria
- [ ] At least three labs and ten usable workspaces reconstruct.
- [ ] Alignment residual and dynamic-scene contamination are quantified.
- [ ] Any paired policy claim identifies exact checkpoint and real-trial source.
- [ ] Otherwise the track is reported as replay/coverage evidence only.

## Paper artifact unlocked
Cross-site real-data generalization and a secondary paired-evidence figure.
