# Compact construction-only handoff

Owner: `trellis2_backend`; branch: `agent/icra-e4-compact-materialization`.
Base commit: `a8733053e85acfda425f16b19cbc26b15d148856`.
Status: SMOKE_PASSED (57 tests in 9.59 s), pending source/commit review. No materialization jobs
or output freeze have been submitted or created by this task.

The existing `robo.eval.e3_factory_materializer` already supports the sealed
multi-scene automatic inventory and scene control shards. Its explicit
`automatic_scene_contract` path does not read an E3 root aggregate or evaluation
outputs. The old `e4_automatic_materialization.validate_inputs` singleton guard
is a launcher limitation; this wrapper uses the existing public producer
without changing it. Do not run E3 evaluation/aggregation to unblock E4.

## Fixed input contract

Source freeze `20260905-61d8ba4-v1`, source commit
`e2185a8809e9bd3462b3c0449a5367cdf95d18b4`, E0 digest
`f80aabb0ee4326b5be667dbd82802917263a5f14f574bd70908ec4559aedfb2a`.
The complete source has 2 scenes, 17 jobs and 85 A0–A4 rows. Scenes are
`27dd4da69e` (7 jobs) and `40aec5fffa` (10 jobs). Preserve all rejected,
abstained and unavailable object records. Each A0/A4 materialization retains
the complete per-scene population; acceptance is copied from the controller.

The four prequalification task IDs remain:

- `27dd4da69e__obj_1001_to_region`
- `27dd4da69e__obj_1001_to_obj_1000`
- `40aec5fffa__obj_1001_to_region`
- `40aec5fffa__obj_1001_to_obj_1005`

Five resets per arm retain 40 planned learned-policy episodes. The source
protocol byte hash binds selection, task families, policy, seed and jitter
declarations. Observation seals bind the common TRAIN frames, source Gaussian,
camera metadata and point files. This stage does not create a qualified reset
bank, change a camera, choose tasks from success outcomes or launch a policy.

## Read-only readiness and new-stage config

```bash
export SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
/group/worldcept/code/SimAny/.venv/bin/python \
  -m run.icra2027.e4_compact_materialization --inspect
```

Missing unpublished control/observation dependencies return `status=WAITING`
with expected paths and unchanged planned denominators. Published malformed
artifacts raise an error. No output tree or synthetic failure result is written
for an unfinished dependency. Source E0, protocol/config resources, complete
inventory, both per-scene audits, observation/control seals and original phase
receipts are authenticated. A READY result contains `source` for a new config:

```yaml
schema_version: 1
scope: e4_compact_materialization_engineering
paper_ready: false
freeze_id: <new reserved immutable stage ID>
python: <absolute resolved path of the configured Python interpreter>
e3_root: /group/worldcept/code/SimAny/outputs/icra2027/20260905-61d8ba4-v1/agentic
source: <exact source mapping emitted by READY inspection>
```

The new E0 must hash this config as input resource
`e4_compact_materialization_config` and the resolved interpreter binary as
`materialization_python`; both use content SHA-256. Stage E0 must bind the clean
executing source commit. It must use a new freeze, never the E3 source freeze.
Freeze the config and source bundle before materializing. No dirty exception.

## Post-control execution

After a READY source review, targeted/E0 gates, and new stage freeze, set
`stage_config` and `stage_root` to the absolute E0-bound config/output paths.
Run each scene once with the same config in ordinary CPU jobs (no GPU/array):

```bash
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/group/worldcept/code/SimAny/.venv/bin/python \
  -m run.icra2027.e4_compact_materialization \
  --config "$stage_config" --freeze-root "$stage_root" --scene-id 27dd4da69e
/group/worldcept/code/SimAny/.venv/bin/python \
  -m run.icra2027.e4_compact_materialization \
  --config "$stage_config" --freeze-root "$stage_root" --scene-id 40aec5fffa
/group/worldcept/code/SimAny/.venv/bin/python \
  -m run.icra2027.e4_compact_materialization \
  --config "$stage_config" --freeze-root "$stage_root" --handoff
```

Outputs: `materialization/<scene>/automatic_scene_descriptor.json`,
`materialization/<scene>/{A0,A4}/materialization_manifest.json`, sealed factory
trees and `materialization/<scene>/result.json`. Final
`materialization/handoff.json` binds actual endpoint manifests, descriptors and
receipts to the fixed 40-episode declaration. The core materializer validates
copied artifacts, exact object rosters and all selected controller decisions.
Already valid partial variants may be authenticated and reused when resuming
the same scene; completed scene/final reports are never overwritten.

The next E4 wrapper consumes these descriptors and factory endpoints to prepare
the actual task/rig/reset qualification. This stage makes no manipulation,
physical-stability or agentic-improvement claim; `paper_ready=false` throughout.

## Tests and current evidence

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_compact_materialization.py \
  tests/test_e4_automatic_multiscene_materializer.py \
  tests/test_e4_automatic_materializer.py \
  --basetemp outputs/e4_compact_materialization_smoke/receipt_pytest
```

Tests cover missing dependencies, incomplete published shards, changed source
commit/config/protocol/inventory/scene ordering, task substitution, altered
stage E0/config/source bundle, arrays, ambiguous phases, existing-producer
delegation and no-overwrite. Existing core tests forbid GT loaders and verify
multi-scene exports with rejected and empty populations without an aggregate.
Real-input review is retained at
`outputs/e4_compact_materialization_smoke/source_review.json`; it recorded both
control shards WAITING while their original jobs were still running. That is
dependency status, not a construction failure or a scientific result.
The final exact implementation review in `final_source_review.json`
authenticated the completed `27dd4da69e` shard and retained only `40aec5fffa`
as an unfinished control dependency.

## Prepared execution stage

Both source controls subsequently completed: `832124` in 8m54s and `832126` in
14m50s, exit `0:0`. `ready_source_review.json` authenticates both complete
shards and their phase receipts. New reserved stage freeze:
`20260905-e8277d2-v1`. Tracked config and E0 recipe:
`configs/experiments/icra2027/e4_compact_materialization/{execution,freeze}.yaml`.
The stage config contains the exact READY source bundle. Execution tests now
pass **58 tests in 5.62 s**, including rejection of a GPU allocation; scene
receipts record stage-config identity, Slurm job ID, host, interpreter and
wall-clock runtime. Exact E0 and root merge precede the two ordinary CPU jobs,
each requesting 4 CPUs, 32 GiB RAM and 30 minutes, with zero GPUs. Source E3
freeze remains unchanged. No policy rollout or qualification result is implied.
