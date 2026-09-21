# Project history

Chronological log of how this repository was built, compiled from the git
history and the coding agent's memory notes (snapshot 2026-08-04). Numbers
quoted are the audited ones unless marked otherwise.

## 2026-07 first half — reproduction to system

- **07-06 → 07-12** — Work begins as a reproduction of *SimFoundry*
  (arXiv:2606.28276, NVIDIA/Stanford et al.): a zero-shot single-frame
  image→sim pipeline (s0 frame selection → s1 SAM3 segmentation → s2 metric
  depth → s3 lifting → s4 TRELLIS image-to-3D → s5 Sim(3) registration →
  s6 CoACD+physics/URDF → s7 PyBullet settle → s8 gsplat render) on
  ScanNet++ scenes with prebuilt Gaussian splats. First commit 07-12.
- **07-13** — GT-driven *factory* mode and the 50-scene validation fleet.
  Gaussian-native **inpainting** pipeline: object removal with a multi-view
  mask vote (essential for transparent bottles) and support-plane hole fill
  refined against per-frame composited 2D inpaints (Qwen-Image-Edit-2509
  beats LaMa on flat structured surfaces). Multi-scene viser viewer.
- **07-14** — GT-free discovery (`automated_image2sim`): SAM3 on strided
  frames × 16 prompts, raycast onto mesh, voxel-IoU union-find merge. MJCF/
  MuJoCo export with Isaac manifests; MuJoCo rollout videos and a live
  browser viewer.
- **07-15** — Baselines re-implemented on our inputs: FlashSplat (exact
  gsplat port) and MaskClustering; ReconViaGen (VGGT-conditioned TRELLIS)
  integrated into the V_mesh slot. Removal verification battery. Furniture
  vocabulary tier added (later descoped to opt-in).
- **07-16 → 07-17** — ICRA paper first draft (double-blind). **Hybrid
  V_mesh**: generate both TRELLIS and ReconViaGen assets per object,
  register both, keep the lower symmetric-chamfer one. Fleet result:
  F1@20mm 0.708→0.783, all 18 ReconViaGen collapses caught by the gate.
  Demo-film pipeline (scan reveal / discover / interact beats).
- **07-19 → 07-20** — Demo v3 with curation gates (asset F1 + hole-fill
  PSNR). Ablation ladder: GT-seg+GT-mesh / SAM3+GT-mesh / SAM3+derived-mesh
  / single-image zero-shot. BEHAVIOR-1K Tier-0 static coverage analysis.

## 2026-07 second half — audit, robot loop, single image

- **07-21 → 07-22** — **Pre-submission audit** (159 review agents → 70
  confirmed issues) followed by systematic fixes: pooled-F1 semantics,
  matched-density MaskClustering comparison (honest edge 1.5× at IoU 0.5),
  official-split render metrics, corrected timing semantics
  (GT 9.7 A6000 h, auto 14.1 A6000 h per 50 scenes). Registration stress
  suite (`tests/test_align_synthetic.py`).
- **07-22** — Two new directions the same day:
  - **Single-image variant**: SHARP feed-forward splat from one photo, then
    per-object SAM3 → lift → TRELLIS+ReconViaGen hybrid. Key fix: a fixed
    camera-frame→z-up rotation (R_ZUP) that recovered 8 wrongly rejected
    objects; 12 objects reached full URDF status from one desk photo.
  - **Robot closed loop**: pi0.5 (openpi, DROID joint-position policy)
    deployed inside exported MuJoCo scenes, RoboLab-style task suites,
    photoreal composite observations (clean background splat + posed asset
    gaussians + segmentation-masked robot). First full runs scored 0 —
    diagnosed as near-field splat mush in the wrist camera plus scan-frame
    exterior cameras leaving the robot small at the frame edge.
- **07-23** — MuJoCo physics sanity video for the single-image scene: valid
  sim, no explosions; drift explained by per-object floating support islands
  and the R_ZUP up-axis approximation, not asset quality.
- **07-27** — Repo renamed **SimFoundry → SimAny** ("SimFoundry" now refers
  only to the cited prior system) and refactored from flat scripts into a
  package; the rename was verified with an import-smoke baseline diff that
  caught two real regressions. The pi0.5 loop turned out to have been dead
  for 5 days on two infra bugs: orbax checkpoint restore silently hanging
  against CephFS (fixed with a node-local /scratch copy: 63 min → 15 s) and
  a missing `#SBATCH --mem` (2 GB cgroup default OOM-killing the 12 GB
  restore).

## 2026-08 — honest numbers and the PhiRoom layout

- **08-01** — All 13 slurm scripts unified on `env.sh`; docs set written
  (CONTRIBUTIONS, BASELINES, ENVIRONMENTS, PAPER_REVISIONS).
- **08-03** — Presentable demo achieved (c50 mug grasp+lift, score 0.50)
  and three demo traps documented: the exterior camera materially changes
  the score (0.50 vs 0.00 across neighbouring scan frames), decluttering
  exposes inpaint scars that clutter hides, and static framing does not
  predict where the grasp happens.
- **08-04** — Inpainted-scene pool widened to 6; systematic scene hunt
  concluded a **dead end**: pre-validated selection criteria (jaw fit,
  bearing, inpaint PSNR) were falsified by 0/36 on the new scenes, while
  c50d2d1d42 composite (4/10 nonzero episodes, 2 lifts) remains a
  favourable outlier. Honest conclusion recorded: visual policy success
  tracks target apparent size and contrast, which oppose good composition.
- **08-04** — Repository reorganized into the **PhiRoom layout** (agents/
  models/ robo/ interface/ run/ docs/ data/ checkpoints/ coding_agents/)
  and pushed to github.com/RunyiYang/PhiRoom with history preserved via
  rename tracking.

## Naming

*SimFoundry* (repo name until 07-27) → *SimAny* (system/paper name) →
*PhiRoom* (GitHub repository name). In prose, "SimFoundry" always means the
cited prior system, never this codebase.
