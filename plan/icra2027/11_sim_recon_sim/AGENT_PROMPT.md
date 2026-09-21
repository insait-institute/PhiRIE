# Lead agent entry point: continue with native scale-up

The first same-engine native one-object milestone is already measured. The former prompt targeted that milestone and an earlier unexecuted budget. Do not dispatch it again or restart completed experiments.

**Read and execute the active prompt:**

[09_scale_up/AGENT_PROMPT.md](09_scale_up/AGENT_PROMPT.md)

Supporting files:

- [Active TODO and N0-N7 task ownership](09_scale_up/README.md)
- [Cohorts, controls, metrics and conclusion boundaries](09_scale_up/EXPERIMENT_MATRIX.md)
- [Prospective configuration template, not executable yet](09_scale_up/cohort.template.yaml)
- [Scale-up execution board](09_scale_up/EXECUTION_STATUS.md)
- [Actual first-milestone measurements and preserved failures](STATUS.md)

Minimal dispatch instruction:

```text
Work in RunyiYang/PhiRoom. Read plan/icra2027/11_sim_recon_sim/STATUS.md and
09_scale_up/AGENT_PROMPT.md, README.md, EXPERIMENT_MATRIX.md and all N0-N7 task
READMEs. Continue from the existing native-policy/import/capture/one-object
milestone; do not rebuild it. First complete legitimate missing DEV pairings,
then 8 independent DEV builds and 40 paired resets per arm (REF/B0 first).
Parallelize cohort, candidate, actual-context verification/import, policy,
statistics and GS tasks by worktree. Freeze the separate 48-instance TEST
matrix only after validity-critical DEV controls pass. Use ordinary independent
Slurm jobs, no arrays. Preserve old results, native task semantics and all
failures. Return measured episodes, tables, continuous videos, job receipts and
supported conclusions, not another plan or a status-only closure.
```

The previous prompt remains in Git history at `e40dbdbf0922d357cf12c4b7c93289d955fb79c0`. This update adds no experiment result and does not alter old frozen cohorts.
