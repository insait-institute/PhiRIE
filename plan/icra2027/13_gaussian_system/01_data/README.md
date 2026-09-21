# P01: data inventory, family splits and beautiful source gallery

## Mission
Read DATASETS.md. Enumerate genuine upstream indexes, permissions and existing local assets. Freeze source-derived scene families and task identities. Curate DEMO separately, before outcomes. Acquire one admitted scene per dataset before expanding.

## Existing/new implementation to read
robo/campaign/data.py; matrix.py; native capture adapters already in robo/roundtrip

## Execution
Use RUNBOOK.md and its actual `robo.campaign` commands. Start with one genuine source-bound DEV unit. Exercise a missing-input or invalid-state negative test. Keep one worktree and owner for changed shared files. Commit tested changes before sealing a full task. Never treat missing local model/native assets as a completed experiment.

## Acceptance and handoff
inventory.jsonl; source-gallery/index.html; scene-family/pretraining-overlap report; native readiness per dataset; exact installed asset releases. Test family leakage and reject insufficient quotas.

Create STATUS.md with source/config/checkpoint identities, exact command, Slurm job IDs, planned/executed/measured counts, output paths, failed gates, actual hardware/time and next executable action. States: IMPLEMENTED_CPU_TESTED, NATIVE_SMOKE_PASS, FULL_RUNNING, COMPLETE_MEASURED, PARTIAL, BLOCKED_EXTERNAL, METHOD_FAILED. Scientific null/negative is not an engineering failure. No fabricated results, test-set recipe tuning or implicit fallback.
