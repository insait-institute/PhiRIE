# P04: Chorus open-vocabulary instances

## Mission
Freeze Chorus and the matching SigLIP2 text encoder. Disable/filter-map raw-Ply outliers explicitly. Compute 3D candidate grouping and compare with current SAM3 lifting at unchanged downstream generator budgets. Preserve source Gaussian identity through splitting, removal and body binding.

## Existing/new implementation to read
robo/campaign/semantics.py: encode, text_query, group; existing discover/factorization interfaces

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
Input-aligned features/query receipts, instance Gaussian-ID sets, projected-mask checks, standard mIoU/instance AP/query accuracy, discovery/build coverage. Unknown queries remain failures; new-category policy competence is not inferred from segmentation.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
