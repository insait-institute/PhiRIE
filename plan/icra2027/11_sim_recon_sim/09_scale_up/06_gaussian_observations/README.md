# N6 — Gaussian policy observations without physics changes

**Priority P1, parallel. Owner: rendering agent.** This task is NOT a prerequisite for the first native multi-instance comparison. Current successful milestone uses a uniform-color native mesh and records `source_gaussian=NOT_RUN`. Do not relabel its footage as Gaussian simulation.

## Read / reuse

Read parent SR4, existing `agents/recon/gsplat_train.py`, Gaussian removal/completion modules, `robo/rendering/`, `robo/eval/observation_pipeline.py`, `harmony_visual_metrics.py`, native policy observation keys and `harness_runner.run_native_episode`. Reuse these rather than introducing another policy input convention.

## TODO

- [ ] Train source scene GS on sealed TRAIN views from the appropriate canonical static capture. Use only authorized RGB-D/posed inputs. A noisy/RGB-only condition is a separate declared ablation, not an undocumented input substitution.
- [ ] Bind each object GS to the same persistent identity, canonical frame and live native body transform as mesh/collision. Verify scale exactly once, quaternion convention, body-versus-COM offsets and camera axes. A native-body state read to animate a rendered object is legitimate runtime state, not GT asset construction.
- [ ] Remove original object appearance and complete the background from permitted TRAIN information before inserting movable object GS. Render a move-and-reveal test. Keep failed removal and missing GS assets in coverage.
- [ ] Compose known robot RGB with correct world/robot occlusion using common depth conventions. Do not always paste the robot in front. Include foreground target, arm-behind-table and thin-gripper/contact cases. If GS depth is approximate, quantify the limitation rather than claiming exact occlusion.
- [ ] Validate B4_NATIVE vs B4_GS at identical frozen state/action trace: physics and scorer inputs identical, only RGB bytes differ. During independent closed-loop execution, actions and states are expected to diverge; do not force identical later states.
- [ ] Preserve native left/right/wrist inputs, intrinsics/extrinsics and the existing resize/preprocessing. Held-out metric renders must not contain demo overlays, different crops or per-image test-time exposure fitting.
- [ ] Use standard PSNR/SSIM/pinned LPIPS with matching cameras and common-view coverage. Follow common protocol for masked metrics; do not call bounding-box LPIPS masked LPIPS.
- [ ] Run native/GS visual diagnostic on DEV while native TEST is underway. After integrity checks, the prospective 240-reset scope subset adds B4_GS only, reusing the corresponding B4_NATIVE episodes. Do not re-run generation or entire native arms for this addition.

## Optional Harmonizer C

After real model access and genuine two-camera/two-episode smoke: full-frame enhancement followed by raw robot-core restoration; camera/episode temporal histories remain independent; no silent raw fallback. Measure target preservation, actual robot-core equality, temporal consistency with explicit motion compensation, p95 whole-observation latency and failures. Use the declared official checkpoint, never an identity backend in a claimed result. Do not claim realtime without measuring the complete policy-observation step on actual hardware.

Harmonizer, an unavailable base model, or one failed GS example must not block native B0/B3/B4/BM. Any supported finite adverse visual effect may be measured as an observation ablation; adopting it as the main system requires actual visual and task evidence. Missing outputs are typed failures, not a different renderer under the same label.

## Outputs and tests

`gs_build_manifest.json`, source/background/object assets and hashes, `native_pose_to_gs.json`, per-camera validation renders, `observation_invariance.json`, held-out frame manifest/metrics, visual failure inventory, measured render/service timings and N5-compatible treatment config. Real renderer integration is tested on recorded native states before a policy run.

Unit/scene tests: transform roundtrip, rotation/scale/COM handling, camera matrix projection, static pose invariance, robot-in-front/behind depth, moved-object ghosting, renderer cannot modify simulation state, stream reset/isolation and model timeout accounting. No hidden GT object segmentation in a method input.

## Demo handoff and completion

Provide native/GS synchronized frames from the SAME recorded state trajectory, and separate continuous closed-loop videos from paired resets. Do not visually synchronize divergent policies by replacing their actual physics. The final video labels `fixed-state rendering comparison` versus `independent closed-loop episodes` correctly.

`STATUS.md` records which scopes have genuine GS observations, view/frame coverage, model access, failed preservation checks, measured native/GS task results and `HARMONIZER=NOT_RUN|MEASURED`. No GS or Harmonizer benefit is claimed from a native mesh-only experiment.
