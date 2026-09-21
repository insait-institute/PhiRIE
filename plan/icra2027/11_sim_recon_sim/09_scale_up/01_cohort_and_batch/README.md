# N1 — Independent cohorts, capture reuse and ordinary-job batch execution

**Priority P0. Owner: cohort/dispatch agent.** Read the common matrix. This task turns an example-specific driver into an instance-based experiment, not a second rollout implementation.

## Existing entry points

Read `robo/roundtrip/capture.py`, `capture_native.py`, `paired.py`, `spec.py`, native adapter, the successful stage dispatch scripts recorded in the parent STATUS, and the existing Slurm/environment helpers. `milestone_report.py` contains seed0/example-specific paths; do not extend that special case into the new result authority.

## First measurable outputs

1. Inventory all five original native reference episodes and their canonical identities. Complete missing reconstructed pairings INCLUDING reference failures. Reuse the existing reconstruction only when its captured asset identities and canonical context match. A changed asset/context requires a new capture/build, not a transform guess. This is DEV continuation, separate from all TEST counts.
2. Resolve 8 independent DEV instances: 2 layouts x 2 tasks x 2 object/context instances; freeze 5 paired perturbation resets for each. Run REF/B0 first, 40 units per arm. A valid negative reconstructed outcome still completes an episode.
3. After DEV fixes are frozen, resolve the 48 TEST instances and 480-unit-per-arm roster. Preserve unavailable planned combinations with explicit reasons, not replacement by easier tasks after outcomes.

## TODO

- [ ] Create a resolver for proposed `configs/experiments/sim_recon_sim/scale_up/{dev,test,scope_subset}.yaml`. Names are implementation targets, not existing configs. Validate task IDs and native fixed/open fixture semantics against the pinned checkout before construction.
- [ ] Write immutable instance and reset rosters before any TEST outcome. Store native policy training split/asset overlap separately from reconstruction DEV/TEST split.
- [ ] Use an allowlisted public capture bundle. For each canonical state, freeze object state while taking TRAIN and held-out camera views. Verify transforms, camera-z vs range depth, units, static-body state and RGB orientation. Known robot masking is allowed, oracle object masks are not constructor inputs.
- [ ] Choose capture density/trajectories on DEV for each replacement scope. The six-view target smoke is not sufficient evidence for a whole-room scan. One coherent capture is reused across B0/B3/B4/BM. Workspace/room capture extensions have their own immutable IDs and treatment comparisons.
- [ ] Cache discovery, generator inputs, initial TRELLIS/RVG outputs and metric render pairs at canonical-instance granularity. Never rerun models per reset or method when inputs are identical.
- [ ] Add an orchestrating matrix driver around existing capture/build/import and `run_native_episode`, with explicit phases for resolve, dry-run, dispatch, collect and status. Prefer `robo.roundtrip.matrix`; this is a new interface to implement, not an available command yet.
- [ ] Ordinary independent Slurm jobs ONLY. No `--array`, no hidden shell loops launching unbounded jobs. One job per instance/stage or a documented small resource-safe shard. Inspect current quota, memory and permitted GPUs. Leave unrelated jobs alone.
- [ ] Use persistent policy services and isolated rendering/generation environments. Measure a small concurrency pilot rather than assuming all processes fit an H200. CPU metrics, import checks and video assembly run concurrently on appropriate compute allocations, not on a busy login node.
- [ ] Workers own unique directories and write atomic terminal records. Never concurrently append to one shared JSONL. One merger checks identity, duplicates, full planned coverage and outcome status.
- [ ] Resume only missing units. A scheduler cancellation before execution is not a failed policy episode. Code/environment failure, method build failure, timeout, task failure and success remain different statuses.
- [ ] Scope-subset selection is a pre-metric deterministic rule over 4 TEST layouts. Reuse L0 results only when instance/reset/contract hashes match, not by similar task names.

## Proposed commands to implement

```bash
python -m robo.roundtrip.matrix --config configs/experiments/sim_recon_sim/scale_up/dev.yaml --phase dry-run --out "$OUT"
python -m robo.roundtrip.matrix --config configs/experiments/sim_recon_sim/scale_up/dev.yaml --phase dispatch --out "$OUT"
python -m robo.roundtrip.matrix --config configs/experiments/sim_recon_sim/scale_up/dev.yaml --phase collect --out "$OUT"
```

Document the actual implemented CLI and `--help`; do not present planned commands as already verified. The matrix driver delegates all episode execution to the canonical harness.

## Outputs

`cohort_manifest.json`, `instances.jsonl`, `reset_bank.json`, `capture_manifest.json` per instance, `planned_units.jsonl`, `dispatch/jobs.jsonl`, `resource_inventory.json`, worker terminal shards, merged `episode_ledger.jsonl`, `coverage.csv`, `failed_units.jsonl`, `status.json`. Every job receipt includes command, stage commit, source/config hashes, allocation and actual output paths. Derive ETA only from measured throughput and state it as an estimate.

## Tests / acceptance

- [ ] Synthetic dry-run enumerates exactly 40 DEV units per arm and 480 TEST units per arm, without launching models.
- [ ] Seed collision with different XML/object IDs is rejected; five resets of one build are not five independent builds.
- [ ] Resume after an injected worker failure does not duplicate completed units or resample scenes.
- [ ] GT/test paths are inaccessible to construction; capture outputs are static and units are verified.
- [ ] First DEV pairings have real native success, coverage and continuous video outputs.
- [ ] 40-unit DEV block completes before TEST release; method thresholds are then sealed.
- [ ] No job array is submitted. A complete TEST block has one terminal record per predeclared unit.

`STATUS.md` must distinguish planned builds, completed builds, planned resets, executed episodes and successes. Include a bounded blocker list and continue independent work rather than repeatedly generating status-only commits.
