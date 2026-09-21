# P05: TRELLIS / TRELLIS.2 / ReconViaGen

## Mission
Run the actual three generator adapters on common object jobs. RVG-v0.2 and RVG-v0.5 are distinct. Compare matched anchor and declared native multi-view conditions. Keep PBR visual, Gaussian output when supported, canonical mesh and shared registration/physics separate.

## Existing/new implementation to read
robo/campaign/models.py: generate; assets.py; existing E3 registration and native geometry evaluator

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
DEV48x3x3 proposals; TEST240x3x1 after DEV recipe freeze. No hidden evaluation alignment/best-seed selection. Same candidate pool reused across B0/B1/B2/B3. Publish quality, failures, coverage and cost.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
