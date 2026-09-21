# P00: close the old follow-up, without rewriting history

## Mission
Read results/simulation-submission-20260908 at 8e7ae22 and 12_simulation_submission. Resolve the 40 F1 missing units. Audit interpreter-symlink fix and F2 state-cache timing. Preserve native success and distinguish supplementary geometry checks.

## Existing/new implementation to read
robo/roundtrip/mechanism_followup.py; scorer_sensitivity.py; object_fidelity.py; original native table pipeline

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
A fresh valid paired block for each unresolved instance, complete canonical component contrasts, F2 measured/unavailable counts and F3 source-matched region metrics. Do not manufacture zeros or tune TEST.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
