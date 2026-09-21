# SimAnyRoom: reference simulation -> reconstruction -> simulation

## Current execution entry point

**The first native one-object milestone is complete; the next task is scale-up.** Read [STATUS.md](STATUS.md) for actual measured DEV evidence, then start at:

- [Native scale-up TODO and task board](09_scale_up/README.md)
- [Lead-agent execution prompt](09_scale_up/AGENT_PROMPT.md)
- [Scale-up experimental matrix and scientific contract](09_scale_up/EXPERIMENT_MATRIX.md)
- [Unresolved cohort/budget template](09_scale_up/cohort.template.yaml)
- [Scale-up execution status](09_scale_up/EXECUTION_STATUS.md)

The scale-up plan was written after reading code `e40dbdbf0922d357cf12c4b7c93289d955fb79c0`. It supersedes the original **unexecuted** 16-instance/1,600-episode RoboCasa budget and initial-milestone dispatch order. It does not overwrite any recorded source, freeze, failure or published result. Existing SR0-SR8 READMEs below remain implementation references; for new scale-up budgets and dispatch, `09_scale_up` takes precedence. Data isolation and experiment validity are not relaxed.

The old [initial lead prompt](AGENT_PROMPT.md) now directs readers to the active prompt. Do not repeat the original milestone simply because an earlier README still says it is required.

## Recorded milestone scope

At the recorded baseline: five native DEV reference episodes, four successes; one reconstructed target-only pair succeeded under the same learned policy. The original room/destination were retained as oracle context. Inputs were ideal RGB-D, construction fixed TRELLIS, and rendering native uniform-color, NOT Gaussian. The unequal-length fixed-action replay and closed-loop results do not yet establish feedback compensation. All exact counts, state/control identities, action traces and limitations live in [STATUS.md](STATUS.md); scale-up planning documents are not additional measurements.

## Scientific question

Starting with a working native simulator, expose only a declared sensor capture to SimAnyRoom, reconstruct movable assets and eventually their room context, and measure how much visual fidelity, physical response and native task performance survive. This is **simulation-grounded reconstruction evaluation**, not real-world transfer, an official complete leaderboard submission, or a claim to outperform a physics engine.

**Primary:** RoboCasa in native MuJoCo/robosuite. **Extension:** BEHAVIOR in native OmniGibson/PhysX. Same-engine comparisons first; do not pool platform success rates. User-facing names are **Agentic Construction** and **Task-Conditioned System Verification**.

```text
Native reference + compatible robot/policy/task
 -> static posed RGB-D capture (RGB-only separately declared)
 -> automatic discovery and reconstructed assets
 -> evidence selection and bounded registration retry
 -> actual-context system verification / bounded repair
 -> import into the SAME engine with unchanged robot/task semantics
 -> matched robot-action replay and closed-loop policy
 -> standard metrics, coverage, uncertainty and real continuous video
```

## Current prospective budget summary

All numbers here are design budgets, not results or power guarantees:

| Stage | Independent instances | Resets/instance | Units per arm | Scope |
|---|---:|---:|---:|---|
| DEV expansion | 2 layouts x 2 tasks x 2 instances = 8 | 5 | 40 | Target-only, REF/B0 first |
| TEST native core | 8 layouts x 3 tasks x 2 instances = 48 | 10 | 480 | Target-only oracle context |
| Task-context extension | 4 fixed TEST layouts x 3 tasks x 2 instances = 24 | 10 | 240 | Target+destination and task workspace |

Core arms REF/B0/B3/B4/BM total 2,400 primary learned-policy episodes. B3/B4 at the two new scopes add 960, reusing compatible L0/reference records. Full-horizon replay, optional GS/V1/Harmonizer, two full-room cases and BEHAVIOR have separate budgets in the active matrix. The primary improvement claim requires B3/B4 outcomes and compute/coverage controls, not merely a large count of B0 target replacements.

## Original SR implementation references

Read [PROTOCOL.md](PROTOCOL.md), [TABLES_AND_METRICS.md](TABLES_AND_METRICS.md) and [SOURCES.md](SOURCES.md) alongside the new matrix. Proposed interfaces must be checked against actual current code and `--help`; a markdown command is not proof of implementation.

| ID | Original task | Still-relevant contract |
|---|---|---|
| SR0 | [Native reference and identity](00_native_reference/README.md) | Native policy and wrapper/import controls |
| SR1 | [Capture and data isolation](01_capture_and_splits/README.md) | Static observations and inaccessible private GT |
| SR2 | [Construction/native import](02_reconstruct_and_import/README.md) | Shared reconstruction artifacts and same-engine import |
| SR3 | [System verification/repair](03_system_verification/README.md) | Actual context and bounded, evidence-driven actions |
| SR4 | [Gaussian observations](04_gaussian_observations/README.md) | Shared live pose and correct occlusion |
| SR5 | [Physical replay](05_action_replay/README.md) | Execute actions, never play back object poses |
| SR6 | [Paired policy](06_paired_policy/README.md) | Native task semantics and frozen procedure |
| SR7 | [BEHAVIOR extension](07_behavior_extension/README.md) | Native OmniGibson and role/state support |
| SR8 | [Tables/demo](08_tables_and_demo/README.md) | Generated standard metrics and source-bound presentation |

## Non-negotiable boundaries

- Construction uses public TRAIN observations, declared sensor access and known robot/cameras. Hidden original object meshes/poses/parameters and TEST outcomes cannot guide candidate choice or repair.
- Start ideal RGB-D, not mislabeled RGB-only; old BEHAVIOR depth initialization is not removed just by setting GT_MESH=0.
- Scene identity is canonical XML/assets/state/metadata, not seed alone. Pair perturbations while retaining reconstruction placement error.
- Native success rules remain unchanged; scorer origin/site semantics require explicit validation after replacement.
- Replay and visual closed-loop policies are separate evidence. Relative motion diagnostics are not absolute pose errors.
- Actual native collision/context, not a different isolated convex hull, defines deployment-matched verification.
- Finite imperfect baseline environments may produce measured failures. Conservative verification must not globally suppress every baseline rollout. Load-invalid/nonfinite states still terminate and retain coverage loss.
- L0 is oracle-context asset replacement. L1/L2/L3 claims require those actual reconstructed scopes and explicit retained-context inventories.
- Use ordinary independent Slurm jobs, **no job arrays**. Reuse hash-compatible outputs and warm models; do not block native experiments on Harmonizer, BEHAVIOR or a learned classifier.
- Old ScanNet++ E1-E9 results and the native DEV milestone remain immutable. No scientific conclusion is fabricated to fill a table.

## Outputs and release

Reuse the existing stage root:

```text
outputs/icra2027/<new-stage>/sim_recon_sim/scale_up/
  contract/ cohorts/ capture_public/ builds/ imports/ verification/
  replay/ episodes/ metrics/ tables/ demo/
```

Private native references are in an access-controlled evaluator vault, not a constructor-readable sibling directory. Workers write per-unit terminal shards and a single merger validates the full roster. A final release binds compatible stage commits and hashes; there is no requirement to regenerate correct stages to obtain a cosmetic identical Git commit.

Done means real multi-instance measurements, independent method/scope contrasts, standard metrics, complete denominators, supported conclusions and continuous demo footage. Documentation, valid schemas and a visually attractive diagram alone do not complete scale-up. The active N0-N7 READMEs specify exact handoffs and acceptance criteria.
