# E0 — Shared Experiment Contract and Freeze Infrastructure

**Priority:** P0  
**Owner:** one infrastructure agent  
**Blocks:** every quantitative experiment

## Goal

Create one fail-closed experiment contract used by construction, fidelity, agentic ablations, manipulation, Harmonizer, task-local support, real captures, and paper-table generation. This task does not run expensive experiments. It makes later outputs comparable and reproducible.

## Paper claim unlocked

Every reported number can be traced to an immutable code/config/input/checkpoint state, and every paired manipulation comparison changes only the declared treatment axis.

## Read first

- `configs/experiments/icra_contract_v1.yaml`
- `configs/experiments/frozen_fields.yaml`
- `configs/experiments/.frozen_fields.lock.json`
- `configs/experiments/harness_paper.final.template.yaml`
- `configs/experiments/paper_pipeline.template.yaml`
- `robo/manifest/`
- `robo/eval/harness_spec.py`
- `robo/eval/harness_validation.py`
- `robo/eval/paper_pipeline.py`
- `plan/00_RESEARCH_CONTRACT.md`
- `plan/01_EXPERIMENT_MANIFEST.md`

Reuse these implementations. Do not introduce a second manifest library.

## Required implementation

### 1. Create the ICRA freeze config

Add:

```text
configs/experiments/icra2027/
  freeze.yaml
  construction_regimes.yaml
  fidelity_manifest.json
  harness.yaml
  audit.yaml
  harmony_visual_manifest.json
  real_world.yaml
```

`freeze.yaml` is the root index. It must contain:

```yaml
freeze_id:
code_commit:
git_dirty: false
created_utc:
paper_repository:
paper_commit_before_update:
input_roots:
checkpoint_roots:
hardware:
construction_config:
fidelity_manifest:
harness_config:
audit_config:
harmony_manifest:
real_world_config:
```

Paths may be relative to the repository or absolute cluster paths, but the resolved path and SHA256 must be written into the frozen output.

### 2. Add a freeze command

Implement one canonical CLI, preferably:

```bash
python -m robo.eval.freeze \
  --config configs/experiments/icra2027/freeze.yaml \
  --out outputs/icra2027/<freeze_id>/contract
```

To allocate a new canonical ID without editing the tracked config, use:

```bash
python -m robo.eval.freeze \
  --config configs/experiments/icra2027/freeze.yaml \
  --auto-freeze-id --out-root outputs/icra2027
```

The allocator uses the UTC date and local `main` commit prefix, while the
manifest records the actual checkout commit separately. Atomic reservations
in `outputs/icra2027/.freeze_ids/` consume an ID even if that attempt fails;
retain these directories. Existing output IDs are skipped. `--dry-run` only
previews the next available ID and does not reserve it. Explicit legacy IDs
remain accepted in smoke mode; paper mode requires canonical ID syntax.
`--allow-dirty-for-smoke` remains necessary for a dirty development checkout.
The copied root config preserves its source bytes; `freeze_manifest.json`
contains the allocated ID and the explicit `paper_ready` flag.

The bare harness command in the historical global checklist requires two
arguments in the current implementation. The accepted configured smoke is:

```bash
python -m robo.eval.harness_smoke \
  --config configs/experiments/icra2027/harness.yaml \
  --out outputs/icra2027/<new-smoke-id>/harness
```

Record a failed bare invocation as a CLI/documentation failure; do not report
it as passing when this separate configured command succeeds.

It must:

- reject a dirty worktree unless `--allow-dirty-for-smoke` is explicitly supplied;
- resolve every referenced config;
- hash all configs and small metadata files;
- fingerprint large checkpoints/directories using the existing manifest hash utilities;
- record Python, CUDA, driver, MuJoCo, PyTorch, and hostname information;
- write `freeze_manifest.json` atomically;
- write `resolved_configs/` copies;
- refuse to overwrite an existing freeze directory;
- emit a concise human-readable `freeze_report.txt`.

### 3. Validate the treatment contract

Extend existing validation rather than adding a parallel checker. Required checks:

- all paired treatments share policy/checkpoint, robot, cameras, action convention, controller, control rate, horizon, task instruction, rubric, and reset IDs;
- only the declared `scene`, `collision`, or `observation` fields differ within a comparison block;
- every real policy has a non-placeholder checkpoint hash;
- `harmonizer_c` has a real checkpoint hash and cannot use the identity backend in a paper run;
- all expected reset IDs are present exactly once per treatment;
- failures are represented as rows, not missing records.

### 4. Add one global preflight

Add:

```text
run/icra2027/preflight.sh
```

It should run imports, manifest schema tests, the harness smoke test, and a dry-run freeze. It must exit non-zero on any failure.

## Output contract

```text
outputs/icra2027/<freeze_id>/contract/
  freeze_manifest.json
  freeze_report.txt
  resolved_configs/
  hashes.csv
  preflight.log
```

`freeze_manifest.json` is the root provenance object referenced by all later task outputs.

## Tests

Add or extend tests covering:

1. dirty worktree rejection;
2. overwrite rejection;
3. deterministic hash for equivalent YAML key order;
4. detection of undeclared treatment drift;
5. placeholder checkpoint rejection in paper mode;
6. missing treatment rows counted as validation failure;
7. atomic output behavior after an injected exception.

## Fast smoke test

```bash
bash run/icra2027/preflight.sh --smoke
```

Expected runtime: under 2 minutes, no GPU.

## Full acceptance criteria

- [x] A clean checkout can create a freeze manifest from only the documented command.
- [x] Re-running into the same directory fails rather than overwriting.
- [x] Modifying one frozen field causes validation to fail with the exact field name.
- [x] All referenced config copies and hashes are present.
- [x] The smoke harness produces complete treatment/reset rows.
- [x] `pytest -q` passes.

## Handoff

Create `plan/icra2027/00_shared_contract/STATUS.md` with the final commit, exact commands, output example, runtime, and failed gates. Other agents may not begin full runs until this task records `READY_FOR_EXPERIMENTS=true`.

Acceptance evidence (2026-09-06): clean sourcec50cb65 exact documented
`env -u SIMANY_EVIDENCE_ROOT SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python bash run/icra2027/preflight.sh --smoke`
with the documented local OSMesa/OpenPI environment passed130tests, imports,
complete synthetic harness, dry-run and immutable publication. Contract
036ae7825cc5623d4998866aeb33547ab1d9ba491603d5cd852fc2a2e6feb338;
source worktree `SimAny-wt/e9-full-qualification/outputs/review/e0.log`.
The final literal whole-suite checkbox remains open: source860f3fc has2929
distinct passing tests/4skips only after preserving834656 and environment-only
98-unit recovery834781, not one all-green whole-suite invocation. Scientific
readiness remains stage-specific; this does not admit missing experiments.

Final whole-suite checkbox closed by a later clean-source invocation: job834881
on `c3ba498cc1be2c481618962f9d0e1392ced9d7ba`, 3006passed/4skipped in336.34s,
imports/pytest/configured harness all exit0. Ordinary SOF3 CPU4/32G, no GPU.
Exact command, JUnit and hashes:
`SimAny-wt/g2/outputs/global-integration/completion_gate.json`. Earlier failed
invocations and missing-unit recovery remain preserved, not overwritten.
