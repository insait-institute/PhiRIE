# Implementation and execution status

2026-09-07. New follow-up implementation is committed for review. Engineering CI run `34154091094` completed successfully on Python 3.11 with MuJoCo 3.3.1, including new unit/integration tests and the existing native table regressions. The exact-source integration commit is `e760c721ba3aafe28a07cbf3612a526b71256353`. Final documentation commit may be newer.

Implemented: B1/B2 outcome reuse; versioned native method admission; fresh same-process five-method block; ordinary-job preparation/launch/collection; canonical component contrasts; read-only final-state scorer diagnostics; reference-mask/crop appearance metrics; CLI wrapper/config/task instructions.

Not executed here: any Slurm policy rollout, actual RoboCasa asset restoration, native segmentation, checkpoint-backed LPIPS on project images, or new scientific measurement. Those require the existing cluster data/model/native environment and the real smoke described in README. No CPU test is presented as GPU/native experiment completion.

Preserved development failures: initial CI runner lacked EGL runtime, then pydantic; fixed CI environment without modifying scientific thresholds. An initial source snapshot upload failed account artifact-storage quota; no user artifacts were deleted. Original source-snapshot workflow restored. No physical-robot requirement or new experiment outcome was invented.

| Track | Code/tests | Cluster smoke | Full experiment | Next |
|---|---|---|---|---|
| F1 mechanism controls | implemented; CPU integration PASS | NOT_RUN | NOT_RUN | Resolve config, prepare, dry-run, one ordinary instance |
| F2 scorer sensitivity | implemented; pure geometry/trace checks PASS | NOT_RUN | NOT_RUN | Native DEV final-state restore and original-rubric match |
| F3 object-region fidelity | implemented; fake-scene metric integration PASS | NOT_RUN | NOT_RUN | Genuine reference segmentation/held-out identity check |

Operators append source/config/checkpoint identities, job IDs, commands and exact output manifests below. Preserve this distinction between implementation and scientific results.
