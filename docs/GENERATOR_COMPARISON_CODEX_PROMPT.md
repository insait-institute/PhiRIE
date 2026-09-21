# Execute the missing phi-RIE generator comparisons

## How to use

This is the repository copy of the missing-generator execution prompt. Give Codex this instruction from the implementation checkout:

```text
Read docs/GENERATOR_COMPARISON_CODEX_PROMPT.md and execute the bounded missing-generator comparison. Reuse existing outputs, submit only unresolved work within the stated limits, and return actual Slurm job IDs and result paths. Do not modify the paper.
```

Storing or reading this document does not launch jobs or authorize spending. The execution instructions below apply when the operator explicitly starts this task. The eight-GPU-hour allowance is cumulative across the task and its resumptions, not a fresh allowance for every chat turn or agent restart.

Related documentation: [final experiment entry point](../FINAL_EXPERIMENTS.md), [bounded component protocol](../plan/icra2027/14_final_experiments/03_components/README.md), [existing runbook](../plan/icra2027/14_final_experiments/RUNBOOK.md), and [module/runtime interface](MODULES.md). This is a narrowly scoped generator follow-up, not a replacement for historical protocols or a second policy campaign.

---

Work in the existing RunyiYang/PhiRoom implementation checkout on the INSAIT server. The paper repository is RunyiYang/SimAnyRoom. Do not modify the paper in this task.

This is an execution task, not another experiment-planning task. Recover existing outputs, implement only missing integration, validate it, and actually submit the missing Slurm jobs. Return real job IDs and exact result locations. Do not stop after writing scripts or showing a dry run unless submission is genuinely blocked.

## 1. Scope and budget

Complete the missing SAM 3D Objects and TRELLIS.2 backend comparisons, alongside TRELLIS and ReconViaGen as references. This explicitly authorizes adding SAM 3D Objects to the bounded generator comparison, despite older documents limiting the generator list. It does not authorize expanding other experiments.

Use the existing 32-object, single-seed component protocol where its roster has actually been frozen. Adding SAM 3D Objects gives 32 objects x 4 backends x 1 seed = 128 planned proposal slots. These are NOT 128 Slurm jobs or necessarily 128 new generations. Reuse compatible outputs and submit only unresolved work.

Resource limits for this invocation:
- At most 2 active GPU jobs submitted by this task, counting pending and running jobs and all allocations started for testing.
- At most 1 GPU per job. Respect the existing project-wide cap and stricter account/QOS limits, including other agents' jobs. Do not cancel or preempt unrelated work.
- Authorize at most 8 additional allocated GPU-hours across smoke tests, inference, export and evaluation. Count failures too. Reserve requested GPU count x wall-time for pending/running jobs so total authorized cost cannot be exceeded.
- Smoke jobs: at most 30 minutes each. Production jobs: at most 2 hours each, right-sized from measured smoke timing.
- Ordinary Slurm jobs only, no arrays, nested GPU allocations, unbounded daemons, multi-node runs, or separate batch schedulers.
- Use one coding agent for implementation and submission. Do not spawn a multi-agent campaign or keep an LLM session alive polling jobs for hours.

Choose the existing compatible partition and runtime from repository configuration and scheduler availability. Do not invent a GPU type, partition, account or memory requirement. If the budget cannot finish the cohort, submit the feasible wave and report the remaining work and required additional budget. Do not shrink the evaluation denominator.

Do NOT launch the previous 432-proposal expansion, the 2,040-unit policy campaign, new 50-scene construction sweeps, Harmonizer, Hunyuan3D, new scene reconstruction, policy training, or paid APIs.

## 2. Inspect the actual checkout and existing outputs

Read AGENTS.md and applicable local instructions. Inspect git status and preserve uncommitted work. Identify the correct checkout from its remote, rather than assuming the local directory is named PhiRoom. Historical paths under /group/worldcept/code/SimAny are search hints, not proof that an artifact exists.

Read these existing entry points before changing anything:
- FINAL_EXPERIMENTS.md
- plan/icra2027/14_final_experiments/03_components/README.md
- plan/icra2027/14_final_experiments/RUNBOOK.md
- models/s4_sam3d.py
- models/s4_trellis2.py
- models/s4_trellis.py
- models/s4_reconviagen.py
- agents/assets/factory_hybrid.py
- agents/assets/s5_align.py
- robo/campaign/models.py
- docs/MODULES.md and the relevant runtime configuration

Inspect current result branches, manifests, candidate directories, evaluation records, squeue and sacct. Search for sam3d_mesh.ply, sam3d_gs.ply, sam3d_meta.json, trellis2_mesh.ply, trellis2_pbr.npz and their associated source/config records. Do not treat an empty GitHub search or an absent summary table as proof that model outputs do not exist.

Classify each required backend/object slot:
- Complete compatible generation and evaluation: reuse.
- Complete compatible generation, missing evaluation: evaluate only.
- Interrupted export with reusable inference artifacts: recover/export only.
- Missing generation: generate.
- Previously failed attempt: preserve its reason and distinguish it from never-started work.
- Incompatible inputs/protocol or insufficient provenance: keep separately, do not silently reuse.

Check input, mask, checkpoint, seed, preprocessing and source identity, not just file existence or modification time. Hashing an old file now does not prove which model or input produced it.

## 3. Freeze one comparable object roster

Locate the actual predeclared 32-object roster, not just the plan describing it. Preserve its dataset and prior-exposure labels. RoboCasa observations must not be described as real ScanNet++ data.

If no actual roster exists, create one new 32-object ScanNet++ component cohort from the existing fixed real-scene pool. Select deterministically across scenes and categories using source-side metadata before inspecting generator quality or downstream success. Record the selection rule and seed. Do not create both a simulated cohort and an additional real-scene cohort under this authorization.

Never select the intersection of successful generators as the planned roster. Keep all planned slots, source failures and missing observations visible. A new cohort on previously inspected scenes is a follow-up, not an untouched test set.

Freeze the object roster, source observations, conditioning images, masks, seed, model revisions, registration configuration, mesh budget, collision settings and metric definitions before inference. Reuse the predeclared seed. If none exists, use seed 42 for every backend and record it before generation. No seed sweep or best-sample selection.

Use one new immutable component-run directory and register it with the existing project dispatcher/accounting. Do not change frozen historical manifests or relabel old results.

## 4. Complete only the missing adapters

SAM 3D Objects already has a factory adapter. Determine whether it can be safely reused by the component runner. Add a thin adapter where necessary rather than rebuilding the model integration. Use SAM 3D Objects, not SAM3 image segmentation or the older SAM3D mask-lifting method.

TRELLIS.2 already has a mesh/material producer. Reuse it. Its lack of native Gaussian output must not prevent geometry, collision and export evaluation. Mark Gaussian-specific appearance as not applicable unless a separately validated conversion already exists. Do not launch a new mesh-to-Gaussian fitting project.

Use the existing uv-managed control entry point and backend-specific interpreters. Do not upgrade or replace environments used by running jobs. Use an isolated clean worktree for code changes. Pin the exact code used by new workers. Do not modify shared source files while submitted jobs depend on them.

Reuse existing checkpoints. Do not accept licenses, download new large checkpoints, or install system-wide dependencies without separate authorization. An inaccessible dependency blocks only its affected backend.

Use real CLI --help and repository scripts to determine commands. Do not invent flags. The full final-policy dispatcher is not automatically a generator worker: reuse the component runner and shared accounting, and do not trigger the full policy suite accidentally.

## 5. Keep the comparison controlled

Use identical source object/frame/mask information for the single-image methods, while retaining and recording their required preprocessing. ReconViaGen's additional views must be recorded and distinguished from single-image conditioning.

Use the same observation-based registration procedure, transform family, retry allowance, triangle budget and collision decomposition for all four backends. Do not give our preferred backend extra candidates or retry opportunities. For this backend comparison, evaluate one declared proposal per method/object. Do not expand the main paper's candidate-selection pool or change its existing results.

Use observed construction geometry for registration and reserve reference geometry for evaluation. Trace the actual contents and origin of files named gt_points.ply rather than relying on their filename. Legacy annotation-assisted artifacts must remain labelled and must not be mixed with an annotation-free comparison.

Do not refine predictions against evaluator ground truth. Use the common registration route for SAM 3D too; record its predicted pose separately rather than giving it a different alignment protocol. Check coordinate conventions and ensure scale is applied once.

Use the same physical priors and simulator settings across backends. Separate generation, registration, collision export and stability validity. A complete mesh or texture file is not automatically a successful physical or appearance result.

## 6. Test, then submit actual missing work

Run focused CPU tests for adapter outputs, provenance, input validation and metric joins. Avoid running unrelated long test suites.

Run one representative end-to-end engineering smoke per unresolved backend on a designated development object, outside the confirmation roster when possible. Reuse an existing compatible smoke only when the code, environment and protocol match. Inspect real files and rendered views, not just a successful process exit. Check finite mesh geometry, canonical/world frames, registration, available appearance and collision import. Record latency, peak memory and expected total cost.

Low reconstruction quality is a scientific outcome, not grounds to tune on confirmation objects. Resolve implementation faults without repeatedly regenerating until an attractive sample appears. Preserve every attempt. At most one infrastructure retry after a diagnosed fix, within budget. A change in scientific recipe requires a new declared study, not an overwritten failure.

After smoke validation, submit the missing production work within the approved limits. Prefer bounded per-backend batches with persistent model loading and per-object output checks, instead of loading a model in one Slurm job per object. Recheck the registry and scheduler immediately before submission. Capture the real sbatch receipt, job ID, allocation request and output log path. Reconcile ambiguous submissions before retrying.

GPU work must run only inside allocated compute jobs, not on the login node. Use existing CPU jobs for CPU-heavy geometry/export stages where appropriate.

If smokes are queued or still running, return their real IDs and an exact continuation command. Do not bypass the smoke gate to claim that production is ready. Do not spend hours repeatedly polling from this chat session.

## 7. Aggregate measured outputs

Use the existing metric producers. Keep per-object records and a full-roster status summary. Export:
- planned, generated, registered, exported and measured counts by backend;
- independent CD and F1 at the existing thresholds;
- collision import validity and the existing standardized stability result;
- comparable appearance metrics only where actually defined and measured;
- generation, registration/export and total elapsed time, plus peak GPU memory.

Report quality on both available support per backend and explicitly identified common support, with denominators. Do not turn missing measurements into zeros, pool incompatible datasets, or count multiple seeds as independent objects. Do not count inference-only timing as total asset-conversion timing. Keep original failed attempts and empty outputs visible.

Use descriptive display names: "phi-RIE + TRELLIS", "phi-RIE + TRELLIS.2", "phi-RIE + SAM 3D Objects", and "phi-RIE + ReconViaGen (multi-view)". Preserve internal historical IDs only where required by the data contracts. Do not show experiment codes in paper-facing tables.

Produce CSV/JSON and a LaTeX table fragment, but do not insert new claims or incomplete numbers into the manuscript yet.

## 8. Required handoff

Commit tested integration changes to a dedicated branch without force-pushing. Publish only compact logs, manifests and measurements, not dataset images or model weights, following the repository's results policy.

Before ending, report:
1. Exact checkout, code commit and run directory.
2. Frozen cohort, dataset, seed and planned slot count.
3. Reused generation outputs and evaluation-only work.
4. Actual submitted job IDs, commands, states, resources and log paths.
5. Validated results versus pending, failed, blocked and never-started slots.
6. GPU-hours spent and reserved against the 8-hour allowance.
7. Exact commands to inspect jobs, dispatch the remaining eligible wave, and collect results, derived from the real implementation.

Completion of this invocation means either real validated results or verified submission of the permitted work with a precise continuation. A YAML file, dry run, CPU test or model import alone is not a submitted experiment. Clearly report PARTIAL or SUBMITTED when work is still outstanding. Do not claim ongoing agent monitoring after ending the session.
