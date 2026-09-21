# Memory index (snapshot 2026-08-04)

One line per note, as kept in the agent's live index. Only notes relevant to
this repository are included in the snapshot.

- [SimFoundry repro status](simfoundry-repro-status.md) — s0-s8 pipeline, run
  recipe, val-fleet + audit-corrected results, TRELLIS/pybullet/gsplat gotchas
- [SimFoundry→SimAny rename](simany-rename-2026-07-27.md) — package refactor,
  SIMF_*→SIMANY_*, what it silently broke and how it was verified
- [SimAny baseline positioning](simany-baseline-positioning.md) — verified
  SimFoundry/WANDA/Lumera facts that shape the claims, plus citation fixes
- [SimAny paper state](simany-paper-state.md) — main+supplementary structure,
  applied revisions, open page-budget issue
- [pi0.5 DROID sim eval](pi05-droid-sim-eval.md) — closed loop in exported
  MuJoCo scenes: serve recipe, action contract, camera traps, honest scores
- [pi0.5 zero-score root cause](pi05-zero-score-root-cause.md) — base-placement
  geometry bug; realism/visibility/gripper/scorer all ruled out
- [SHARP single-image feedforward](sharp-single-image-feedforward.md) — the
  feed-forward splat behind the single-image variant, verified recipe
- [ScanNet++ gsplat data layout](scannetpp-gsplat-data-layout.md) — splat
  paths, mesh-frame poses, render recipe (PSNR 33+)
- [Cluster GPU + envs](cluster-gpu-and-envs.md) — srun debug/a6000 usage,
  mini-viewer & sam3 envs, SAM3 needs bf16 autocast
