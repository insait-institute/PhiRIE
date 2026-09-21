# P06: masked background completion model comparison

## Mission
Compare Telea, SDXL, Qwen Edit2511 and approved explicit Gemini model IDs. Save raw provider output as well as outside-mask-preserving composite. Same prompt semantics/masks/canvas/call count. Use clean same-state synthetic targets or valid observed holdouts, never invented real-background GT.

## Existing/new implementation to read
robo/campaign/models.py: inpaint, reserve_remote; existing TRAIN-view Gaussian background refit

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
24DEV+96TEST cases per admitted model; actual API usage/cost reservations; hole/crop and raw outside-mask metrics; post-GS heldout/multiview results. An API timeout/refusal stays missing/failed, not another model silently substituted.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
