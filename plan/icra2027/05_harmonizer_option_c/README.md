# E5 — Harmonizer Option C: Full-Frame Enhancement with Robot Restoration

**Priority:** P1, promoted to P0 only if it improves or preserves manipulation  
**Paper output:** observation block in manipulation table; visual diagnostics in extended paper  
**Depends on:** E0, E2 aligned views, E4 harness

## Research question

Can full-frame harmonization reduce neural-rendering and insertion artifacts while preserving the robot, target, temporal consistency, and frozen-policy behavior?

Harmonizer is an external observation tool. It does not improve geometry or physics and must not be presented as if it does.

## Read first

- `integrations/harmonizer/server.py`
- `robo/rendering/harmonizer_client.py`
- `robo/rendering/robot_restore.py`
- `robo/rendering/harness_composite_obs.py`
- `robo/rendering/mujoco_masks.py`
- `robo/eval/observation_pipeline.py`
- `robo/eval/harmony_visual_metrics.py`
- `configs/experiments/harmony_visual_manifest.template.json`
- `configs/experiments/harness_paper.final.template.yaml`
- `run/harness/serve_harmonizer.sh`
- `plan/23_NVIDIA_HARMONIZER_INTEGRATION.md`

## Fixed conditions

1. `raw_composite`
2. `color_match`
3. `harmonizer_non_temporal`
4. `harmonizer_temporal`
5. `harmonizer_robot_restore_c`

The main paper manipulation block may use only raw, raster, and Option C. The full five-way visual comparison belongs in the extended paper/supplement.

## Option C definition

For raw observation `I_raw`, full-frame enhanced image `J`, and robot blend mask `A_R`:

```text
I_C = A_R * I_raw + (1 - A_R) * J
```

`A_R` must be:

- exactly 1 on an eroded robot core;
- exactly 0 outside a dilated robot mask;
- smoothly feathered only in the boundary band.

The robot core must be byte-exact to the raw input after color-space conversion and before policy resize. A visually similar robot is insufficient.

## Service contract

Use the existing Unix-socket service:

```bash
export HARMONIZER_SOURCE=/path/to/NVIDIA/harmonizer/src
export HARMONIZER_CHECKPOINT=/path/to/checkpoint
export HARMONIZER_SOCKET=/tmp/simany_harmonizer.sock
bash run/harness/serve_harmonizer.sh
```

Every response must include:

```text
stream_id
frame_index
model/checkpoint hash
mode
input/output shape
round-trip latency
inference latency
history offsets
server generation
error status
```

Streams are isolated by:

```text
freeze / treatment / scene / task / reset_state / camera
```

Exterior and wrist cameras may never share temporal history. History resets on every episode, camera/config change, server restart, or treatment switch.

## Required implementation work

### 1. Paper-mode server validation

Add a health/preflight command that verifies:

- official source commit and checkpoint hash;
- model loads and returns expected resolution;
- temporal and non-temporal modes are distinguishable;
- stream reset works;
- two camera streams remain isolated;
- no previous-episode frame influences a new episode;
- identity/smoke backend is rejected in paper mode.

### 2. Color-match baseline

Implement one deterministic, non-generative baseline using a fixed method such as per-channel affine color transfer estimated from valid background pixels. Freeze the exact formula. Do not use a manually tuned LUT per scene.

### 3. Export aligned evaluation sequences

For the same frozen camera frames, save:

```text
raw composite
color match
non-temporal Harmonizer
temporal Harmonizer
Option C
real/held-out reference
robot reference mask
robot prediction mask
target reference mask
target prediction mask
explicit temporal warped/reference pairs
latency records
```

Use the same filenames across conditions.

### 4. Connect Option C to E4

The `simany_harmony_c` treatment must run through `robo.eval.harness_runner`. The enhancer may change only RGB. Add an assertion that object poses, contacts, robot state, and task scorer inputs are identical before policy actions diverge.

## Visual metric command

Complete:

```text
configs/experiments/icra2027/harmony_visual_manifest.json
```

Run:

```bash
python -m robo.eval.harmony_visual_metrics \
  --manifest configs/experiments/icra2027/harmony_visual_manifest.json \
  --out outputs/icra2027/<freeze_id>/harmony/visual_table \
  --lpips-device cuda
```

Use the already implemented metrics:

- PSNR/SSIM/LPIPS on aligned held-out frames;
- target-mask IoU;
- robot-mask IoU;
- temporal LPIPS on explicitly warped frame pairs;
- p95 latency per camera;
- deterministic robot-core equality.

Do not compute temporal LPIPS from unrelated consecutive frames without motion compensation.

## Required outputs

```text
outputs/icra2027/<freeze_id>/harmony/
  service_info.json
  health_report.json
  sequences/<condition>/<scene>/<camera>/<frame>.png
  masks/robot_reference/
  masks/robot_prediction/<condition>/
  masks/target_reference/
  masks/target_prediction/<condition>/
  temporal_pairs/
  latency.jsonl
  visual_table/
  failure_cases.csv
  matched_strips/
```

## Preservation gates

Option C reaches the policy experiment only if:

1. robot core equality passes for every processed frame;
2. no episode/history leakage is detected;
3. target-mask IoU does not show a practically meaningful degradation relative to raw;
4. robot silhouette IoU is at least as good as unconstrained Harmonizer and the exact core is preserved;
5. temporal consistency is not materially worse than raw/non-temporal;
6. failure/timeout rate is included in coverage;
7. latency is measured on the actual rollout hardware.

Do not set arbitrary favorable thresholds after seeing the test set. Freeze acceptable degradation on a development set.

## Manipulation gate

Option C remains a main-paper treatment only when E4 shows one of:

- improved overall or stage-level performance with a compatible confidence interval;
- statistically preserved manipulation while aligned visual quality improves;
- a clear improvement for photoreal policies without unacceptable coverage or latency loss.

If visual metrics improve but policy behavior degrades, report the negative result and remove Option C from the main system claim.

## Tests

- exact core pixels remain identical;
- feather mask is bounded in `[0,1]` and monotonic across the boundary;
- two camera histories are isolated;
- reset clears all histories;
- frame index mismatch fails;
- server timeout becomes `enhancer_failure`;
- no raw fallback under the Harmonizer label;
- color-match output is deterministic;
- visual metric manifest rejects mismatched filenames;
- temporal metric requires explicit aligned pairs.

## Fast smoke

1. Start the service with a real checkpoint on one GPU.
2. Process 10 frames from two cameras and two episode IDs.
3. Intentionally reset one stream and verify the other is unchanged.
4. Run the visual metrics on the tiny output.
5. Run one scripted-policy harness episode under raw and Option C.

## Full-run target

- at least 3 rooms and both policy cameras for visual diagnostics;
- at least 300 aligned frames per condition, scene-balanced;
- all E4 observation-treatment episodes under the same reset bank;
- automatically selected median, best, and worst examples using frozen LPIPS/target-IoU quantiles.

## Acceptance criteria

- [ ] Official model/checkpoint provenance is frozen.
- [ ] Option C passes byte-exact robot-core tests.
- [ ] Temporal state never crosses episode/camera boundaries.
- [ ] All five visual conditions are evaluated on identical frames.
- [ ] Failure coverage and p95 latency are reported.
- [ ] E4 contains a real Option C treatment, not precomputed cherry-picked images.
- [ ] The final paper wording reflects the actual visual and policy outcome.

## Handoff

`STATUS.md` must include service/checkpoint hashes, hardware, per-condition frame counts, preservation failures, visual table paths, E4 treatment ID, and a clear `MAIN_PAPER_GATE=PASS|FAIL`.