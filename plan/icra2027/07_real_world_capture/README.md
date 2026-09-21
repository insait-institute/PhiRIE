# E7 — Real-World Capture-to-Simulator Evaluation

**Priority:** P1  
**Paper output:** real-world table, part (a)  
**Depends on:** E0 and a stable constructor

## Research question

Can SimAnyRoom convert uncontrolled robot-workspace and casual phone captures into metric, simulation-ready room models without manual per-object placement?

This task evaluates construction and robot-frame alignment. It does not claim real-world policy success.

## Read first

- `run/run_droid_recon.sh`
- `run/run_video2sim.sh`
- `run/fetch_droid_raw.py`
- `capture/`
- `agents/eval/droid_alignment_eval.py`
- `robo/eval/real_world_metrics.py`
- `robo/eval/real_world_table.py`
- `configs/experiments/real_world_construction.template.csv`
- `plan/02_PHONE_CAPTURE_PROTOCOL.md`
- `plan/03_PHONE_RECONSTRUCTION_FRONTEND.md`
- `plan/04_METRIC_SCALE_ROBOT_ALIGNMENT.md`
- `plan/16_DROID_PAIRED_TRACK.md`
- `plan/17_PHONE_ROOM_COLLECTION.md`

## Evaluation sets

### DROID workspaces

Freeze the current 9-workspace set. Record episode/video IDs and selection criteria before running the final evaluation. Do not replace hard sequences after seeing reconstruction quality.

The set should span:

- different tables/counters;
- clutter levels;
- lighting conditions;
- camera motion and viewpoint ranges;
- object counts.

### Casual phone rooms

Target at least 3 rooms, preferably 5:

- one office/workroom;
- one kitchen/dining area;
- one living/bedroom-like space;
- at least one challenging reflective/transparent-object scene.

Capture with the documented protocol, not ad hoc footage selected after reconstruction.

## Frozen capture protocol

For every phone room record:

```text
phone model and camera mode
resolution / frame rate / codec
capture duration
number of extracted frames
exposure/focus lock state
trajectory description
scale/alignment aid, if any
room dimensions or reference measurements
capture start/end UTC
operator interventions
```

The preferred trajectory is slow and metric-friendly:

1. one wide loop around the task workspace;
2. one lower-angle loop covering object sides and support contacts;
3. short close passes for small objects;
4. no fast pans, digital zoom, abrupt exposure changes, or repeated textureless wall-only segments.

Any fiducial or known-length object used for scale must be declared. Do not call a marker-assisted run marker-free.

## Required pipeline

### DROID

Use the existing entry point:

```bash
bash run/run_droid_recon.sh <declared arguments>
```

The agent must add a frozen batch config rather than hard-coding sequence IDs in a shell loop.

### Phone/video

Use:

```bash
bash run/run_video2sim.sh <video_or_frame_root> <output_root>
```

If the current interface differs, update this README and provide one wrapper:

```text
run/icra2027/run_real_world_construction.sh
```

The wrapper must read only `configs/experiments/icra2027/real_world.yaml` and write to the current freeze root.

## Full-build definition

A workspace counts as a full build only when all required artifacts exist and pass schema checks:

- metric camera trajectory and scene reconstruction;
- object discovery output;
- at least one accepted movable object;
- registered object assets;
- collision and physical parameters;
- clean/background-editing result or an explicitly declared fallback status;
- full-room collision export;
- robot-frame alignment when the capture includes a robot;
- simulator load/settle smoke test;
- immutable build manifest.

Report successful reconstruction and full simulation build separately.

## Robot-frame alignment

For DROID, evaluate alignment using held-out robot kinematics or markers not used by the estimator.

Report:

- translation error in centimeters;
- rotation error in degrees;
- alignment pass rate under a predeclared threshold;
- number of held-out frames/poses;
- failure reason when alignment is unavailable.

Do not evaluate on the same robot observations used to estimate the transform.

For phone rooms without a robot, translation/rotation fields remain null unless an independent alignment reference exists.

## Required output rows

Create:

```text
outputs/icra2027/<freeze_id>/real_world/workspaces.csv
```

Required columns:

```text
freeze_id
source
workspace_id
capture_id
reconstruction_success
full_build_success
accepted_objects
input_frames
translation_cm
rotation_deg
alignment_pass
runtime_minutes
build_manifest_path
capture_manifest_path
failure_stage
failure_reason
```

Then run:

```bash
python -m robo.eval.real_world_table \
  --construction outputs/icra2027/<freeze_id>/real_world/workspaces.csv \
  --out outputs/icra2027/<freeze_id>/real_world/table
```

Part (b) remains empty until E8 provides real trials.

## Additional qualitative artifacts

For each workspace save a fixed set:

```text
input contact sheet
reconstructed-room fly-through
object inventory sheet
physics/collision view
photoreal simulator view
alignment overlay, when applicable
failure card, when unsuccessful
```

Use the same layout for every workspace. Failed examples must be shown in the supplementary material, not omitted.

## Tests

- capture manifest schema validation;
- fixed sequence/room list cannot change without a config hash change;
- full-build status requires every declared artifact;
- failed stage/reason is mandatory when status is false;
- held-out alignment frames do not overlap estimator frames;
- units are cm/degrees in final CSV;
- duplicate workspace IDs fail;
- table aggregation includes failed workspaces;
- phone rooms without independent alignment remain null rather than zero.

## Fast smoke

Run one short DROID clip and one 10–20 second phone clip through reconstruction-only mode, produce manifests, and execute the simulator load smoke. Expected output may be a failed full build, but the failure must be typed and aggregated correctly.

## Full-run acceptance criteria

- [ ] All 9 frozen DROID workspaces have terminal rows.
- [ ] At least 3 frozen phone rooms have terminal rows.
- [ ] Successful reconstruction and full-build coverage are both reported.
- [ ] DROID alignment uses held-out evidence.
- [ ] Every workspace has a capture/build manifest and runtime.
- [ ] Failures are included and visualized.
- [ ] No autonomous real-success claim appears from this task alone.

## Demo handoff

Export:

```text
outputs/icra2027/<freeze_id>/real_world/demo_candidates.csv
```

Select three diverse workspaces using predeclared rules:

1. median-quality successful DROID workspace;
2. median-quality successful phone room;
3. informative failure or difficult success.

The demo agent may not replace them with prettier examples without changing the selection config and documenting the change.

## Handoff

`STATUS.md` must list frozen captures, success/full-build counts, alignment statistics, runtime, failure stages, output manifests, and `REAL_CAPTURE_GATE=PASS|FAIL`.