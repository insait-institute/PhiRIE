# P07: state-conditioned and 3D-cached appearance

## Mission
First fix rawGS visibility and removal. Run official Harmonizer, existing Option C and our state-conditioned corrections on identical raw states. Use deployed depth/ID/robot/contact/normal buffers, causal per-camera history and disocclusion rejection. Optionally train StateResidual16 on scene-disjoint same-geometry/state pairs. Compare frozen and learned corrections separately.

## Existing/new implementation to read
robo/campaign/harmonizer.py; visual.py; learned.py; color_cache.py; existing robot_restore and observation hooks

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
Real model smoke, actual buffer correspondence, protected pixel checks, per-frame p95/coverage and same-physics policy ablation. Color-cache code fits bounded SH-DC residuals from ACTUAL compositing responsibilities; extracting these weights from the admitted renderer is a required integration step. Keep mean/scale/rotation/opacity fixed; no global geometry guarantee.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
