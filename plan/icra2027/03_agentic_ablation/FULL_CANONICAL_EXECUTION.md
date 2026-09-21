# Full automatic canonical construction execution

Owner: `trellis2_backend`; branch: `agent/icra-e3-full-canonical`.
Preparation base: `73abc74c80491915f86c1c815ee75cd4bd1e5f51`.
Scope: complete automatic TRAIN-only construction, 50 scenes / 1,871 objects /
9,355 A0–A4 rows. This is engineering evidence; evaluation and paper promotion
remain separate tasks. Only the evaluation manifest contract is extended to accept the E0-bound
predeclared pilot; metric, matcher, surface-selection and paper code are unchanged.

## Reused inputs and fixed semantics

Use exactly the command and explicit terminal-audit roots in
`POOL_OVERRIDES.md`. Discovery is `20260905-76c15d5-v1`; initial TRELLIS is
`20260905-02b54da-v2`; ordinary RVG is `20260905-d1e8e21-v1`. Only the two
predeclared RVG replacements use `20260905-58cabb1-v1`. The last recovery
`831700` completed in 22m43s and its independent audit `831701` in 66s; the
3f15a9266d audit authenticates 108/108 proposals and 216 finite/nonempty files.
Original failed process histories remain bound and charged once. Failed,
invalid and unprepared jobs retain all five policy rows.

The existing `agentic_automatic_policies.yaml` is unchanged. A1–A4 share their
initial proposal pool. Only the existing bounded registration retry creates
an additional proposal/artifact; no new generator run is requested. Selection
and retry read sealed construction evidence without evaluation GT. There is no
threshold or acceptance tuning on these 50 scenes.

## Frozen pilot rule

Before new controller outcomes, choose the smallest strictly positive complete
discovery population, breaking ties lexicographically by scene ID. This selects
`13c3e046d7` (one object; the tied scenes are `ac48a9b736` and `f3685d06a9`).
The legacy GT-assisted scene `5748ce6f01` is not the current automatic population;
the new rule uses only complete TRAIN discovery counts, before controller results.
The pilot consumes the same full-cohort config and sealed inventory that the
remaining scenes will use. It is not a separate success-selected population.

Pilot launch requires the clean committed source to be merged to main, focused
schema/negative tests, exact E0, canonical preflight and a sealed inventory with
the complete 50 / 1,871 / 9,355 counts. Run an ordinary one-GPU observation job,
then a dependent CPU control job for the selected scene. Before releasing the
other 49 scenes, authenticate observation/control seals and the five terminal
policy rows using the existing canonical loaders. Require a genuine finite
observation and at least one actual construction proposal so the pilot tests
the real renderer/registration/probe path. Acceptance, fidelity improvement and
retry improvement are not pilot release criteria. Preserve any failed pilot;
do not substitute a different scene after inspecting its outcome.

Before the full release, also run the existing independent evaluation matcher
and E3 evaluation producer for this sealed pilot, under a separate evaluation
freeze. Its manifest pins this construction execution YAML using
`pilot_execution_config: {path, size_bytes, sha256}`. The construction execution
YAML freezes `pilot` with selection rule, scene ID and one planned job before E0.
The matcher validates the construction E0, full roster/object IDs and control
seal before opening GT. Require the exact final evaluation shard schema and
five policy rows; unmatched geometry remains null, and metric positivity is
not a gate. Legacy pilot manifests retain their fixed `38d58a7a31` / 15 rule.

## Execution recipe

Use the existing `run/icra2027/e3_fresh_canonical.sbatch` for every phase.
Set absolute `E3_CANONICAL_CODE`, `E3_CANONICAL_FREEZE`,
`E3_CANONICAL_CONFIG`; select `E3_CANONICAL_PHASE` from `preflight`, `inventory`,
`observe`, `control`, with `E3_CANONICAL_SCENE` only for the last two.
Fresh full configs live in
`configs/experiments/icra2027/e3_full_canonical/` after authenticated preparation.

Observation: one GPU with at least 44 GiB usable / 30 GiB free, 4 CPUs, 32 GiB
RAM, 10 minutes, interpreter
`/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python`. Prefer idle
Hala / gcp* / sof1* GPUs. CPU control: 4 CPUs, 32 GiB RAM, four-hour limit,
zero GPU, interpreter `/group/worldcept/code/SimAny/.venv/bin/python`.
The long limit covers large scenes; no CPU job reserves a GPU.

After the new pilot passes, submit ordinary independent observation/control
pairs for all 49 remaining scenes. No arrays. Keep the pilot output as its
one scene in the full denominator; never submit it twice. Record exact commands,
verified Slurm resource snapshots, dependencies, job IDs and logs beneath the
freeze. Resume only missing units in a new admissible attempt, preserving all
failed/partial logs and original source/config identities.

Expected outputs are canonical `agentic/input_inventory`,
`agentic/observations/<scene>`, `agentic/control/<scene>` and
`execution_receipts/<scene>`. No root evaluation aggregate is needed for the
construction-only handoff. The independent single-scene pilot evaluation may
run only after that scene is sealed. Full scientific aggregation requires the
complete inventory and all 9,355 policy rows.

## Readiness and smoke evidence

Read-only source closure replay is recorded under
`outputs/full_canonical_preparation/readiness.{json,stderr}`.
A completed readiness receipt can be published with `--publish-readiness` and
its exact SHA256, rechecking metadata/config/audit anchors and retaining all
original/recovery runtime histories through the existing config writer. This
only avoids a repeated preparation scan; canonical inventory after E0 still
authenticates every artifact, and drift fails the freeze.
Use the existing canonical schema, terminal-pool, recovery and runtime tests:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e3_canonical_pool_overrides.py tests/test_e3_canonical_cohort_config.py \
  tests/test_e3_canonical_cohort_phase.py tests/test_e3_fresh_canonical.py \
  tests/test_agentic_runtime_accounting.py tests/test_agentic_automatic_population.py \
  --basetemp outputs/full_canonical_preparation/pytest
```

Then run the exact-source `run/icra2027/preflight.sh --smoke`, create the new
contract from `e3_full_canonical/agentic_fresh_freeze.yaml`, and execute canonical
preflight/inventory. Record their actual commands, counts and hashes before
submitting the pilot. Current scientific claim gate remains NOT_RUN.

## Published source closure (2026-09-05)

Reserved construction freeze: `20260905-859f51d-v1`. The read-only readiness
command completed with exit 0 in approximately 31 minutes; its 153,978-byte JSON
has SHA256 `3d01d5d618ed8115159c13a969ba8bd9fd0510011c0683a402a5373871c3c6ca`.
It authenticates 50 scenes / 1,871 planned jobs / 9,355 policy rows, including
1,801 prepared inputs. Initial availability is TRELLIS 1,640 and ReconViaGen
1,764; the remaining 107 ReconViaGen jobs have typed unavailable outcomes.
These are input coverage counts, not canonical controller or paper results.
Both original/recovery process histories are retained exactly.

Publication rechecked metadata anchors and used the existing config writer;
`outputs/full_canonical_preparation/publication.json` is its receipt. The
execution YAML adds the predeclared one-object pilot before source commitment
and E0. `agentic_fresh_jobs.yaml`, `agentic_fresh_execution.yaml` and
`agentic_fresh_freeze.yaml` are all tracked in the new config directory.
Canonical inventory must independently revalidate the entire closure after E0.
A changed artifact fails that freeze; publication is not an inventory cache.

Implementation source `2f22aa5` passed 116 targeted schema/negative tests and
129 exact E0 tests. An earlier E0 invocation incorrectly exported the production
`SIMANY_EVIDENCE_ROOT` into checkout-local fixtures (11 path-bound failures);
its logs are retained. Unsetting that variable for E0 tests passed all 129.
Production canonical phases still explicitly use the evidence root. Repeat E0
on the final config commit and retain its exact log before publishing the
construction contract or scheduling the pilot. Scientific claim gate: NOT_RUN.
