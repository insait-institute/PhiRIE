# Task 01 — Immutable Experiment Manifests and Provenance

**Priority:** P0  
**Suggested owner:** infrastructure / reproducibility engineer  
**Depends on:** Task 00  
**Blocks:** Tasks 09, 10, 18, 20, 21

## Objective
Implement a single manifest and output contract for scene construction, policy rollouts, metrics, and paper artifacts so every number is reproducible from committed configuration and code.

## Existing code
- `agents/core/common.py`: path and scene conventions.
- `robo/eval/pi05_eval.py`: current rollout outputs.
- `agents/eval/aggregate_results.py`: aggregation entry point.
- `run/env.sh`: environment-variable configuration that must be captured rather than remain implicit.

## Output schema
Create:
- `robo/manifest/schema.py`
- `robo/manifest/io.py`
- `robo/manifest/hash.py`
- `configs/experiments/example_manifest.yaml`
- `tests/test_manifest_roundtrip.py`

Every output must carry `scene_build_commit`, `scene_manifest_hash`, `policy_checkpoint_hash`, `controller_config_hash`, `camera_config_hash`, `task_id`, `initial_state_id`, and `rollout_seed`. Never overwrite a completed output directory.

A build manifest must additionally record capture/reconstruction inputs, scale/alignment residuals, instance inventory, asset hashes, full-room collision hash, simulator versions, and verification results. A rollout manifest must record language, rubric version, horizon, observation preprocessing, action convention, video/state/contact paths, success, staged progress, and failure label.

## Implementation steps
1. Define typed Pydantic/dataclass schemas with explicit version numbers.
2. Canonicalize and hash manifests; reject NaN, absolute cluster-specific paths, and unordered dictionaries.
3. Make output directories content-addressed or fail if an existing manifest differs.
4. Add helpers to snapshot git commit, dirty status, package versions, CUDA, simulator, and checkpoint hashes.
5. Retrofit `pi05_eval.py`, factory reports, and table aggregation to emit/read the schema.
6. Add a `manifest diff` CLI highlighting only declared independent variables between methods.

## Tests
- Round-trip YAML/JSON equality.
- Same semantic manifest in different key order yields same hash.
- Changed camera extrinsic or checkpoint byte changes hash.
- Runner refuses to append to a directory with a different manifest.

## Acceptance criteria
- [ ] A smoke rollout can be reproduced from the saved manifest alone.
- [ ] Aggregation rejects comparisons differing on undeclared frozen fields.
- [ ] Table cells link back to run IDs and source manifests.

## Paper artifact unlocked
“Immutable manifests and traceable table provenance” contribution and all confidence intervals.
