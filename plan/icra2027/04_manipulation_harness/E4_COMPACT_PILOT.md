# Compact automatic E4 prerequisite pilot

Owner: `trellis2_backend`; branch: `agent/icra-e4-compact-canonical`.
Implementation base: `ab79101`; execution commit is the clean commit recorded by E0.
Reserved freeze: `20260905-61d8ba4-v1`. State: `SMOKE_PASSED`; E0 and
canonical execution have not run. Slurm IDs: none. Claim gate: `NOT_RUN`.
This engineering pilot does not replace the declared full E4 matrix.

## Selection and denominator fixed before qualification

`configs/experiments/icra2027/e4_compact_canonical/protocol.yaml` authenticates
the complete 50-scene automatic-discovery population. Select the two smallest
discovered-object populations with both semantic task families, breaking ties
by scene ID. Labels use the existing `pi05_tasks` role vocabulary. Neither
construction quality nor robot-policy outcomes enter selection.

| Scene | All objects | Semantic queries | Fixed region task | Fixed receptacle task |
|---|---:|---:|---|---|
| `27dd4da69e` | 7 | 18 | `obj_1001_to_region` | `obj_1001_to_obj_1000` |
| `40aec5fffa` | 10 | 10 | `obj_1001_to_region` | `obj_1001_to_obj_1005` |

Task IDs have the scene prefix and `__` separator. Choose exactly the
lexicographically first task ID per family per scene **before** qualification.
Keep all 17 objects and 85 A0–A4 terminal rows. All 28 prospective queries may
undergo qualification (2 arms × 5 resets = 280 cells). The four fixed learned
tasks retain 40 planned episodes (2 arms × 5 resets), including unavailable or
failed selected tasks. Never replace a selected task after qualification.
Policy is `pi05_droid_jointpos`; reset seed 0 and XY jitter 0.01 m. The actual
reset bank, paired rig, camera and scorer remain to be sealed after canonical
construction. No learned-policy launch is authorized by this prerequisite.

## Authenticated reuse

The existing canonical source loader, terminal-pool adapter and RVG recovery
accounting authenticate the inputs; no generation or metric producer is copied.

- Discovery: `20260905-76c15d5-v1/auto_discovery_pilot`, fresh official TRAIN-only.
- TRELLIS: `20260905-02b54da-v2`, 14 valid initial proposals. The three invalid
  TRELLIS Gaussian artifacts in `40aec5fffa` retain their typed terminal status.
- RVG: `20260905-d1e8e21-v1` for `27dd4da69e`; explicit recovery
  `20260905-58cabb1-v1` for `40aec5fffa`. All 17 proposals are available.
  The original failed RVG process history remains in runtime accounting.
- Scene Gaussians: `20260905-9ef4ab1-v1/gaussian_train_only/<scene>/train-full/scene.ply`.

Existing preliminary observation caches are not reusable. Their Gaussians are
`/data/ScanNetppv2_gsplat/splats/<scene>.ply`, hashes `f93ca3bf...` and
`18295caa...`; the new TRAIN-only inputs are `234df91c...` and `043d1327...`.
The old slots (`obj_01`, etc.) also differ from the automatic population.
The canonical observation producer explicitly requires one CUDA device; its
render cache is in memory only. No authentic CPU path exists in that producer.

## Reproducible commands and resources

Real-input validation (already passed; output in
`outputs/e4_compact_smoke/readiness.json`):

```bash
env SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
  PYTHONDONTWRITEBYTECODE=1 /group/worldcept/code/SimAny/.venv/bin/python \
  -m run.icra2027.e4_compact_canonical
```

Targeted smoke: **82 passed in 0.88 s**, including task-outcome independence,
unprepared-object retention, duplicate IDs, protocol mutation and forbidden
array/scene/phase failures:

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_compact_canonical.py tests/test_e3_canonical_cohort_config.py \
  tests/test_e3_canonical_cohort_phase.py tests/test_e3_canonical_pool_overrides.py \
  tests/test_e3_fresh_canonical.py tests/test_agentic_runtime_accounting.py \
  --basetemp outputs/e4_compact_smoke/final_pytest
```

After merge/review, run E0 from this exact clean source checkout. No dirty-smoke
exception is permitted for execution:

```bash
env SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
  /group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.freeze \
  --config configs/experiments/icra2027/e4_compact_canonical/agentic_fresh_freeze.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-61d8ba4-v1/contract
```

Set `E4_COMPACT_CODE` to the absolute frozen worktree and `E4_COMPACT_FREEZE`
to the absolute output freeze. Run the existing canonical phase through
`bash run/icra2027/e4_compact_canonical.sh`, with `E4_COMPACT_PHASE` set to
`preflight`, then `inventory`, then independent `observe` and dependent
`control` jobs per `E4_COMPACT_SCENE`. The wrapper does not submit jobs.

Observation interpreter:
`/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python`.
Request 1 GPU, 4 CPUs, 32 GiB RAM, 10 minutes, on Hala / gcp* / sof1*;
the existing guard requires ≥44 GiB usable GPU memory and ≥30 GiB free.
Prior 15-object observation job `830593` used one A6000 and 32 GiB RAM and
completed in 9 s; this is precedent, not a measured duration for these scenes.
Control uses `/group/worldcept/code/SimAny/.venv/bin/python`, 8 CPUs,
32 GiB RAM, 60 minutes and **no GPU**. Prior 15-object control took 1628 s.
Submit ordinary independent jobs only, never arrays.

Expected outputs under the new freeze are `contract/freeze_manifest.json`,
`agentic/input_inventory/`, `agentic/observations/<scene>/`, canonical controller
shards and `execution_receipts/<scene>/`. No output exists yet beyond the ID
reservation. Source/config changes after execution require a new freeze.

## Gates and next handoff

Config/source authentication and targeted smoke PASS. Canonical observation,
control, task materialization, rig/camera validation, fixed-task qualification,
scripted scorer smoke and learned rollouts are NOT_RUN. The next step after
canonical outputs is to construct the actual E4 endpoint and task manifests,
then diagnose their measured gates. The existing E4 materializer's one-scene
inventory assumption and qualification-to-real-policy ingestion boundary may
need narrow typed adapters; no producer changes are included here. No task is
declared unavailable merely because its canonical result is not yet produced.
All paper and manipulation-improvement gates remain NOT_RUN.
