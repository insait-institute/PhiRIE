# SimAnyRoom ICRA 2027 Experiment TODO

Paper title: **SimAnyRoom: Making Rooms Simulatable through Agentic Real-to-Sim**

This directory is the authoritative execution plan for the current paper. The older flat files in `plan/` are design history and should be used only when a task README below links to them. One coding agent should own one task directory. Do not give a single agent the whole plan.

## Scientific contract

The paper makes one central claim: room-scale real-to-sim should be treated as **evidence-driven agentic artifact construction**, not as an unconditional cascade of one-shot model calls.

Every experiment must therefore obey these rules:

1. **No hand-edited paper numbers.** JSON/CSV is authoritative. LaTeX is generated from the frozen outputs.
2. **No ground truth in construction decisions.** GT is visible only to evaluation code after a build is frozen.
3. **Coverage precedes conditional quality.** Rejected jobs, failed builds, timeouts, crashes, and enhancer failures remain in the denominator.
4. **One comparison changes one declared axis.** Policy, robot, cameras, control, resets, horizon, instruction, and rubric stay frozen for paired manipulation.
5. **No silent fallback.** In particular, a failed Harmonizer call may not become a raw-RGB episode under the Harmonizer label.
6. **A retry must invoke a new action.** Re-reading or re-scoring the same artifact is not an agentic retry.
7. **No new headline metric without a written reason.** Use the metrics already fixed in the paper: yield, PSNR, SSIM, LPIPS, CD, F1@20, stability, success/stages, AUROC, AUPRC, Brier, Risk@80/60, alignment error, runtime, and coverage.
8. **Every result is traceable.** Store the code commit, config hash, input hashes, checkpoint hashes, hardware, start/end time, and exact output paths.

## Freeze naming and artifact layout

Use one immutable freeze ID:

```text
YYYYMMDD-<git-short-sha>-v<integer>
```

All tasks write under the same root:

```text
outputs/icra2027/<freeze_id>/
  contract/
  construction/
  fidelity/
  agentic/
  harness/
  harmony/
  audit/
  real_world/
  demo/
  paper_tables/
```

Never overwrite a freeze. A rerun creates a new freeze ID. Symlinks such as `outputs/icra2027/latest` are allowed, but tables must record the resolved immutable path.

## Task board

| ID | Priority | Task | Unlocks | Depends on |
|---|---|---|---|---|
| E0 | P0 | [`00_shared_contract/README.md`](00_shared_contract/README.md) | valid provenance and shared freeze | none |
| E1 | P0 | [`01_construction_scale/README.md`](01_construction_scale/README.md) | Table I, scale/input claims | E0 |
| E2 | P0 | [`02_fidelity_metrics/README.md`](02_fidelity_metrics/README.md) | Table II, PSNR/SSIM/LPIPS/CD/F1 | E0, frozen builds |
| E3 | P0 | [`03_agentic_ablation/README.md`](03_agentic_ablation/README.md) | title claim and agentic ablation | E0, object proposals |
| E4 | P0 | [`04_manipulation_harness/README.md`](04_manipulation_harness/README.md) | manipulation table | E0, E3, full-room collision |
| E5 | P1 | [`05_harmonizer_option_c/README.md`](05_harmonizer_option_c/README.md) | visual diagnostics and observation intervention | E0, E2, E4 |
| E6 | P1 | [`06_task_local_support/README.md`](06_task_local_support/README.md) | task-support table and risk-coverage | E0, E1, E2 |
| E7 | P1 | [`07_real_world_capture/README.md`](07_real_world_capture/README.md) | DROID/phone construction table | E0, stable constructor |
| E8 | P2/P0 if hardware exists | [`08_real_robot_pairs/README.md`](08_real_robot_pairs/README.md) | matched sim-real evidence | E0, E4, E7 |
| E9 | P0, final | [`09_paper_freeze/README.md`](09_paper_freeze/README.md) | reproducible paper tables | E1–E8 as available |
| D0 | P0 | [`10_demo/README.md`](10_demo/README.md) | 90 s video, 30 s teaser, live presentation | hero builds from E3–E5 |

## Dispatch waves

### Wave A: contract and cheap evidence

Run E0 first. After its freeze tests pass, E1 and E2 can run in parallel. D0 may begin implementing the video framework, but may not freeze shots yet.

### Wave B: central claim

Run E3. This is the highest-value unfinished experiment because it directly tests the word **agentic** in the title. Do not start a broad manipulation sweep until the construction policies and their output paths are frozen.

### Wave C: downstream evaluation

Run E4, E5, and E6 in parallel after their prerequisites exist. E4 owns all rollout ledgers. E5 owns only the observation service, preservation diagnostics, and the Harmonizer treatment. E6 owns hidden-GT task-query labels and LOSO analysis. They must not create competing rollout formats.

### Wave D: real inputs and physical trials

Run E7 on DROID and phone captures. Run E8 only when the real robot, policy checkpoint, cameras, and reset protocol can genuinely be matched. If not, E8 must leave the physical-trial table empty and trigger claim removal rather than invent a proxy.

### Wave E: freeze and presentation

E9 produces the final paper artifact tree. D0 then re-renders its final result cards from the same frozen JSON/CSV files. The video may not display numbers from a different freeze.

## Definition of done for every coding agent

A task is complete only when all of the following exist:

- implementation and/or config changes;
- unit tests for the new data contract;
- one fast smoke test from a clean checkout;
- one machine-readable example output;
- the full-run command or Slurm launcher;
- a `STATUS.md` inside the task directory containing:
  - owner and branch;
  - final commit;
  - commands executed;
  - hardware and runtime;
  - output paths and hashes;
  - passed and failed acceptance criteria;
  - known limitations;
- no uncommitted changes in the working tree;
- no manual value copied into a paper table.

## Branch and merge policy

Use branches named:

```text
agent/icra-e<id>-<short-task-name>
```

Examples: `agent/icra-e2-fidelity`, `agent/icra-e4-manip-harness`.

Agents may merge only after the task's smoke test and schema tests pass. Expensive full experiments can run after merge from an immutable main-branch commit. The final handoff must include the commit SHA, not merely a branch name.

## Global preflight

Before dispatching any task:

```bash
git status --short
bash run/smoke_imports.sh
pytest -q tests
python -m robo.eval.harness_smoke
```

If a command is unavailable in the local environment, record the exact missing dependency in `STATUS.md`; do not replace it with a different test and call the gate passed.

## Paper-to-task map

| Paper result | Authoritative producer |
|---|---|
| automatic construction | `robo.eval.construction_metrics` |
| room/object fidelity | `robo.eval.fidelity_metrics` |
| agentic ablation | E3 output contract, then E9 |
| closed-loop manipulation | `robo.eval.harness_runner` + `robo.eval.main_table` |
| task-local support | `robo.eval.audit_loso` + `robo.eval.audit_metrics` |
| Harmonizer diagnostics | `robo.eval.harmony_visual_metrics` |
| real-world construction/trials | `robo.eval.real_world_table` |
| all final table provenance | `robo.eval.paper_pipeline` |

When an existing producer is insufficient, extend it. Do not add a second script that computes the same table differently.