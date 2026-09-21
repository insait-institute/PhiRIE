# SR4 — Gaussian observations with correct occlusion

**Owner:** rendering agent. **Priority:** P1, parallel with SR3/SR5. Depends on one native imported scene. Read ../PROTOCOL.md. Reuse `robo/rendering/`, `robo/eval/observation_pipeline.py`, `integrations/harmonizer/` and `robo/eval/harmony_visual_metrics.py`. Do not make Harmonizer access a prerequisite for native/raw-GS evaluation.

## Main rendering contract

Use one live simulation state to transform accepted object Gaussians and render them against the automatically completed static GS background. The native robot remains the benchmark robot. Camera intrinsics, extrinsics, camera ordering, color space and preprocessing must match the native policy exactly. A benchmark camera can have a different K from the scanning camera; it is not re-estimated from images.

A robot segmentation overlay is insufficient if it draws the robot through a foreground object. Composite native robot RGB/depth against world/object RGB/depth, with valid visibility-aware masks. The renderer must also suppress old baked copies at the object's original location. Freeze a practical GS depth/opacity rule using development fixtures and mark transparent-object depth ambiguity; do not quietly use oracle scene depth in the main observation arm.

Implement or extend a native observer adapter, preferably `robo/roundtrip/observations.py`. It returns raw full-resolution layers and EXACT policy-preprocessed images. It must not modify qpos/qvel, object transforms, collision or scorers. Hash state before and after observation generation to prove side-effect freedom.

## Required conditions

- Native textured renderer of the B4 reconstructed assets.
- Raw factorized Gaussian composite of the exact same B4 compiled physics.
- Optional Harmonizer full-frame + robot restoration (Option C).
- Privileged REF_GS diagnosis: reference physics drives reconstructed visual assets. Requires a sealed role/pose mapping and is labeled oracle-physics, never complete reconstruction.

For REF_GS, document whether asset placement is snapped to reference state or preserves canonical estimation offset. Use one fixed choice, `reference_pose_controlled`, for this appearance diagnostic. Do not compare its coverage or success as though it were end-to-end construction.

## Option C

Use I_C = A_R * I_raw + (1-A_R) * H(I_raw). A_R=1 on a nonempty eroded VISIBLE robot core, zero outside the expanded visible robot mask, with feathering only at the boundary. Copy core pixels exactly before resize/encoding. Verify the actual tensor fed to the policy, not only a saved preview. Resize may mix boundary pixels; do not claim bit equality on an arbitrarily resized full robot silhouette.

Temporal history is isolated by platform/run/treatment/scene/task/reset/camera. Explicitly clear it on episode reset and service restart. Do not batch adjacent frames in a way that changes causal temporal conditioning. The checkpoint identity backend is smoke-only. An unavailable or failed enhancer produces a typed missing/failed arm, never a silent raw fallback.

## Offline rendering is not a policy result

First render all visual conditions at identical saved native states and cameras to isolate image differences. In closed loop, trajectories can diverge after different observations; do not claim those videos depict equal states after divergence. Pretty replay must not replace the actual low-resolution policy input used for inference.

Save full-resolution RGB/depth/robot mask, policy input RGB, native body-state hash, camera state, enhancer metadata and per-step timings. Reconstruction TEST views and reference images remain evaluation-only.

## Tests and pilot

Fixtures: robot in front of object; object in front of robot; gripper holding object; wrist camera; moved object reveals clean background; two camera intrinsics; blank robot core; shared-state no-mutation. Wrong depth ordering, reused temporal history and duplicated baked object must fail.

Pilot: 30 consecutive frames from two cameras across two episodes, then one continuous raw-GS policy episode through the canonical harness. Measure observation p50/p95 and total step latency separately. Slow synchronous simulation is allowed and must be labeled; do not claim real time from video FPS.

## Proposed command and handoff

```bash
python -m robo.roundtrip.observations --build "$BUILD_MANIFEST" --states "$TRACE" --config configs/experiments/sim_recon_sim/observations.yaml --out "$OUT/observations"
```

Deliver observer plugin, render manifest, occlusion tests, side-effect report, task/robot mask sources, matched visual strips and latency report. Option C stays optional until real weights, target preservation and history isolation pass. Keep raw GS/native results even if Harmonizer is blocked.
