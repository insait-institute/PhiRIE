# SimAny Paired Robot Evaluation Harness

This document is the executable contract behind the paper's main manipulation
table. The harness is not a convenience wrapper around independent rollout
scripts. It admits a comparison only when the same persisted reset record and
the same procedural contract are reused, while exactly one declared treatment
axis changes.

## Treatment axes

| Axis | Supported paper treatments |
|---|---|
| Scene construction | `box_proxy`, `simany` |
| Collision | `private_shims`, `full_room` |
| Policy observation | `raster`, `composite_raw`, `harmonizer_c` |

`robo.eval.harness_spec` rejects a comparison block when two or more of these
axes change. The primitive box proxy is a controlled stand-in and must never be
called ground truth. The private-shim treatment requires a genuine alternate
MJCF/task-suite path. The runner fails rather than relabeling the full-room MJCF
as a shim condition.

## Frozen contract

The following remain fixed inside a comparison block:

- robot asset and initial robot state
- policy ID, checkpoint hash, and preprocessing
- camera intrinsics and extrinsics
- action convention, controller, and control frequency
- task ID, language variant, horizon, and staged rubric
- persisted reset-state ID and derived reset seed

Every episode receives one append-only ledger row and a manifest. Successful
rollouts and task failures store per-tick action, joint state, object pose,
contacts, task stages, policy latency, and observation latency. Build failure,
policy timeout, safety termination, environment crash, and enhancer failure are
distinct terminal outcomes.

## Configuration

Copy and fill the paper template:

```bash
cp configs/experiments/harness_paper.template.yaml \
   configs/experiments/harness_paper.yaml
```

Required manual substitutions are limited to real asset locations and the
frozen policy service/checkpoint. They are data configuration, not hidden
per-episode intervention.

## Run

```bash
python -m robo.eval.harness_runner \
  --config configs/experiments/harness_paper.yaml
```

Safe resume uses the same output directory. `reset_states.json` is never
regenerated once present, and completed treatment/reset pairs are not rerun.
Use a new output directory for a new experimental contract.

## Outputs

```text
outputs/harness_runs/<run>/
  resolved_harness_config.json
  reset_states.json
  harness_ledger.jsonl
  harness_validation.json
  harness_summary.json
  episodes/<treatment>__<reset>/
    manifest.json
    timeseries.json.gz
    video.mp4                  # optional
  paper_tables/
    main_table.json
    main_table.csv
    main_table.tex
```

A run is paper-eligible only when `harness_validation.json` has `"ok": true`.
The validator checks duplicate rows, unknown treatments, unplanned resets,
complete planned coverage for every treatment, and equal reset sets within each
comparison.

## Harmonizer Option C

Start the model in its own environment/container:

```bash
python -m integrations.harmonizer.server \
  --socket /tmp/simany_harmonizer.sock \
  --backend nvidia \
  --source-dir /path/to/NVIDIA/harmonizer/src \
  --model-path /path/to/checkpoint.pkl
```

The service maintains independent temporal histories per
`episode/image_key`. Histories are reset at every episode. The client is
fail-closed: timeout, malformed output, or sequence mismatch becomes
`enhancer_failure`; it never silently substitutes raw RGB.

`robo.rendering.robot_restore.restore_robot_core` implements Option C. Pixels in
an eroded robot core are copied byte-for-byte from the raw simulator image. A
narrow boundary band is feathered, and all pixels outside the dilated robot mask
come from the enhanced image. The exact-core invariant is tested in CI and
recorded per camera.

## Main table

The runner automatically generates the manipulation table. It can also be
regenerated without rerunning episodes:

```bash
python -m robo.eval.main_table \
  --config configs/experiments/harness_paper.yaml \
  --ledger outputs/harness_runs/paper_main/harness_ledger.jsonl \
  --out outputs/harness_runs/paper_main/paper_tables
```

All success and stage rates use the number of planned episodes as denominator.
A failed build or enhancer request therefore lowers coverage and cannot improve
an average by removing a difficult case.
