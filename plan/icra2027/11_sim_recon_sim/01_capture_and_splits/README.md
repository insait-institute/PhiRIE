# SR1 — Static capture, train/test split and private references

**Owner:** capture/data agent. **Priority:** P0. **Depends on:** SR0 native state loading. Read ../PROTOCOL.md. The main initial condition is explicitly ideal RGB-D with metric camera poses; posed RGB-only is a later separate ablation.

## Existing code to reuse carefully

`agents/recon/behavior_extract.py`, `run/run_behavior_recon.sh`, `agents/recon/gsplat_train.py`, the reconstruction scene contract, and `agents/core/common.py`.

The old BEHAVIOR path is NOT an RGB-only roundtrip: `init_points.ply` is backprojected from simulator depth even when `GT_MESH=0`; the default `GT_MESH=1` uses depth-fused geometry. Old clips can contain different object placements across episodes, and the shared-camera contract excludes cameras with incompatible K. Do not silently inherit these properties. Reuse serialization utilities, not the old experiment label.

## New capture driver

Implement `robo/roundtrip/capture.py` using the native adapter. For each task instance:

1. Load its canonical settled state with native reset/task identities recorded privately.
2. Declare whether robot is parked or hidden during scanning. Keep the robot and room state fixed over all capture views. Hide only the known robot, never task objects or occluders.
3. Sample a predeclared camera path from allowed workspace bounds: upper/wide views plus lower side views and open-container views. The path cannot use reconstructed errors or TEST outcomes. Record camera path generation parameters and any collision checks used by the privileged capture planner as an idealized acquisition aid.
4. Capture 120 TRAIN, 20 DEV and 40 TEST views at 1280x720 by default. Lock this before method comparison. Favor spatially separated held-out arcs; do not claim out-of-distribution generalization from adjacent frames. Save per-frame K and T_world_from_camera in an explicit OpenCV convention, depth in meters and native timestamps.
5. The constructor sees TRAIN only; DEV is accessible to a separate tuning role on development scenes. TEST remains private until build outputs are sealed. No TEST camera is used for source GS initialization or optimization.
6. Recheck static body poses at capture end. Any motion above predeclared numerical tolerance invalidates the static capture; preserve the attempt and recollect under a new capture ID before reconstruction, not after selecting favorable metrics.

Do not reveal original object asset IDs in public image paths. Independent evaluation can know them. Public task language and known robot config are allowed.

## Public and private bundles

```text
capture_public/<opaque_capture_id>/
  capture_manifest.json
  train/rgb/<frame_id>.png
  train/depth_m/<frame_id>.npy       # RGB-D only
  train/cameras.jsonl
  robot_config.json
  task_instruction.txt

<private-reference-vault>/<capture_id>/
  canonical_native_state.*
  native_role_bindings.json
  static_meshes/ object_meshes/ collision/ physics.json
  dev/ test/
  reference_hashes.json
```

For posed RGB-only, depth files and depth-derived init points/TSDF are absent from the mounted public bundle. SfM can use RGB camera poses as declared input. A geometry snapshot generated from all reference surfaces is never sensor depth.

Record `sensor_regime`, metric-pose assistance, camera trajectory, source image resolution, distortion/undistortion, depth semantics (ray length versus camera-z), missing pixels, color space and all masks. Main masks come from the method. Oracle segmentation is a separate labeled condition.

## Reference geometry and matching

Reference object visual surfaces and collision assets are evaluation-only. After construction freezes, the evaluator creates a one-to-one mapping between inferred instances and task/reference roles. Freeze matching rules using development data; record unmatched, merged and duplicate detections. Do not hide 3D geometry failures by choosing a different GT match per method after quality is known.

Sample area-weighted mesh surfaces with the same seed/count for all methods. Evaluate in the global metric frame without a post-hoc ICP alignment to TEST GT. A separately labeled canonical-shape metric can remove placement error for diagnosis. Report observed-surface versus full-object support separately when appropriate. Exact point splits of one mesh are not independent evidence if the entire mesh was already provided to the constructor.

## Proposed commands

```bash
python -m robo.roundtrip.capture --config configs/experiments/sim_recon_sim/capture.yaml --roster "$ROSTER" --public-out "$OUT/capture_public" --vault "$REFERENCE_VAULT"
python -m robo.roundtrip.capture --validate "$OUT/capture_public" --report "$OUT/capture_validation.json"
```

Implement first; reject unresolved source IDs and conflicting output paths. Avoid downloading full demonstration datasets: native capture is sufficient for this experiment.

## Smoke, tests and handoff

Smoke: one fixed room, 6 TRAIN + 2 held-out views, one depth plane, and an asymmetric object. Verify camera projection and depth reconstruction numerically before running SAM/generation.

Tests: w2c/c2w inverse, quaternion order, meter scale, depth-z versus range, different K, missing depth, mixed-state rejection, duplicate frame IDs, TRAIN/TEST disjointness, constructor cannot read vault, metadata contains no asset IDs, RGB-only mount contains no depth-derived files.

Acceptance: a reproducible capture yields identical hashes under the recorded determinism policy; native GT cannot reach constructor; TEST remains sealed; all planned captures including failures have records. Handoff `capture_manifest`, native-state hash, split IDs, actual view counts and first valid opaque capture ID to SR2.
