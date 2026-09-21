# P03: RGB-video and persistent Gaussian identity

## Mission
Use TRAIN-only static capture. Estimate cameras via the real COLMAP/3DGS adapter or an explicitly versioned existing frontend. Output SfM gauge honestly, then attach observation-derived metric calibration before physics. Do not use sensor/GT depth in the RGB-video row.

## Existing/new implementation to read
robo/campaign/video.py; existing Gaussian training, removal and composite renderer

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
Real video reconstruction with source Gaussian IDs, camera/scale receipts, heldout eval-only views and move-and-reveal/robot-occlusion checks. Neither scaled input nor same-pose rendering proves dynamic correctness.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
