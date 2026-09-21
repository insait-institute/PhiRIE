# Task 23 — NVIDIA Harmonizer Integration

**Priority:** P0 for visual observation quality, P1 for the paper claim  
**Suggested owner:** rendering / systems engineer, followed by evaluation lead  
**Depends on:** immutable manifests, `CompositeObs`, paired runner, build audit  
**Blocks:** harmonized-observation ablation, task-local visual reliability study

## Objective

Integrate NVIDIA Harmonizer into SimAny as an **optional RGB observation post-processor** after neural scene composition and before policy preprocessing.

The integration must answer two separate questions:

1. **Visual question:** Does Harmonizer reduce appearance, lighting, shadow, and novel-view artifacts in SimAny's reconstructed-scene observations?
2. **Robotics question:** Does that visual improvement preserve task geometry and improve, or at least not damage, frozen-policy behavior under a fully matched control contract?

Harmonizer modifies RGB observations only. It must never modify simulator state, object poses, collision geometry, task scoring, reset states, or hidden ground truth.

## Research hypothesis

SimAny's physical state can be correct while its policy observation remains visibly inconsistent because:

- generated object Gaussians were reconstructed under lighting different from the room;
- the clean background and reinserted object do not cast mutually consistent shadows;
- novel-view 3DGS rendering creates blur, holes, ghosting, or boundary artifacts;
- the MuJoCo robot is composited with a different image formation process from the neural scene.

A single-step neural enhancer should improve task-local visual fidelity, but a generative enhancer can also hallucinate, shift boundaries, remove small targets, or change the robot silhouette. Therefore the method is useful only when **harmonization gain is paired with task-preservation gates**.

## Non-goals

- Do not claim Harmonizer improves geometry or physics.
- Do not call a visually enhanced frame ground truth.
- Do not silently replace raw observations when inference fails.
- Do not fine-tune before zero-shot compatibility and preservation are measured.
- Do not call the integration real-time unless measured p95 latency is below the declared 15 Hz control-period budget on the actual target hardware.
- Do not make Harmonizer a core SimAny contribution unless it improves a task-relevant or policy-level metric, not only human preference.

## Upstream dependency

Use the official `NVIDIA/harmonizer` release, pinned to an exact commit. The released system provides:

- temporal inference with `diffusion_harmonizer.pkl`;
- non-temporal inference using the same checkpoint with `--nontemporal`;
- a faster exported non-temporal checkpoint, `harmonizer_nontemporal.pt`;
- fixed 576 x 1024 RGB input/output in the released model card;
- Cosmos Predict2 0.6B as the backbone.

Upstream code is Apache-2.0. Model weights are governed separately by the NVIDIA Open Model License and must be recorded in the release audit.

## Current SimAny integration points

The existing observation path is:

```text
MuJoCo state
  -> clean background Gaussian rendering
  -> transformed object Gaussian rendering
  -> concatenated neural scene
  -> MuJoCo RGB + robot segmentation
  -> raw composite frame
  -> policy resize/pad to 224 x 224
```

Relevant existing files:

- `robo/rendering/pi05_render.py` — `CompositeObs`
- `robo/envs/pi05_env.py` — `DroidSimEnv.get_obs()`
- `robo/eval/pi05_eval.py` — policy preprocessing and single-condition evaluation
- `robo/eval/paired_runner.py` — matched reset and manifest-based matrix evaluation
- `robo/manifest/` — provenance and frozen-field validation
- `agents/eval/build_audit.py` — pre-rollout scene-quality evidence

Harmonizer belongs between the raw composite frame and policy resizing. Raw RGB must always remain available for audit and ablation.

---

# Proposed architecture

## 1. Layered rendering instead of one opaque composite

Refactor `CompositeObs.render()` into a layered interface:

```python
@dataclass
class RenderLayers:
    world_rgb: np.ndarray       # background + live object Gaussians
    robot_rgb: np.ndarray       # MuJoCo RGB
    robot_mask: np.ndarray      # exact robot segmentation mask
    raw_composite: np.ndarray   # current behavior
    camera_name: str
    frame_index: int
```

Add:

```python
CompositeObs.render_layers(camera_name) -> RenderLayers
CompositeObs.render(camera_name) -> np.ndarray
```

`render()` must remain backward compatible and return the same raw composite when no enhancer is configured.

## 2. Enhancer abstraction

Create a minimal renderer-independent interface:

```python
class ObservationEnhancer(Protocol):
    def reset(self, stream_prefix: str) -> None: ...
    def enhance(
        self,
        image: np.ndarray,
        *,
        stream_id: str,
        frame_index: int,
        camera_name: str,
    ) -> EnhancementResult: ...
```

`EnhancementResult` contains:

```text
image_uint8
latency_ms
model_id
checkpoint_hash
mode
used_history
failure
```

Implement:

- `IdentityEnhancer`
- `HarmonizerClient`

Do not import Cosmos or upstream Harmonizer inside the SimAny process.

## 3. Separate Harmonizer service

Harmonizer requires a different software stack from the SimAny gsplat / MuJoCo environment. Run it as a dedicated process or container.

Recommended layout:

```text
integrations/harmonizer/
  server.py
  protocol.py
  README.md
run/harmonizer/
  build_container.sh
  download_checkpoints.sh
  serve.sh
  smoke.sh
configs/rendering/
  harmonizer.yaml
robo/rendering/
  observation_pipeline.py
  harmonizer_client.py
```

The service owns the model and all temporal history. The SimAny process owns simulation and raw rendering.

### Request schema

```json
{
  "op": "enhance",
  "stream_id": "<run>/<condition>/<episode>/<camera>",
  "frame_index": 17,
  "mode": "temporal",
  "resolution": [1024, 576],
  "robot_preserve": "feathered_core",
  "image_encoding": "raw_rgb8"
}
```

### Other operations

```text
health
model_info
reset_stream
reset_all
shutdown
```

Use a Unix socket or local ZeroMQ transport for the first implementation. File-per-frame inference is allowed only for the offline smoke test, never for closed-loop evaluation.

## 4. Per-camera temporal state

The official temporal inference uses prior enhanced frames. Exterior and wrist cameras must never share history.

Stream identity must include:

```text
run_id / condition / scene / task / reset_state_id / camera_name
```

History is reset on:

- every environment reset;
- every new episode;
- camera-intrinsic or extrinsic change;
- resolution change;
- condition change;
- server restart or model reload.

Default temporal offsets follow the upstream release: `[-1, -2, -3, -4]`.

The first frames without sufficient history use the released non-temporal path. This warm-up status must be logged per frame.

## 5. Robot-preserving harmonization

A generative enhancer must not freely redraw the robot used by the policy. Evaluate three strategies:

### A. Full-frame enhancement

```text
raw composite -> Harmonizer -> output
```

Maximum lighting/shadow integration, highest hallucination risk.

### B. World-only enhancement

```text
background + objects -> Harmonizer -> overlay raw MuJoCo robot
```

Exact robot preservation, but no robot-aware shadow correction.

### C. Full-frame enhancement with robot restoration — recommended main mode

```text
raw composite -> Harmonizer
harmonized frame + raw robot RGB + robot mask -> feathered restoration
```

Restore the robot core exactly while retaining Harmonizer changes outside the mask, including possible contact-shadow corrections. Use a distance-transform feather band around the robot boundary. Record mask dilation, erosion, and feather width in the manifest.

Do not preserve manipulated objects by default because object harmonization is one of the intended effects. Instead, enforce task-object preservation through metrics and rejection gates.

## 6. Evaluation conditions

Use explicit, immutable observation conditions:

```text
raster
composite_raw
composite_harmonizer_nontemporal
composite_harmonizer_temporal
composite_harmonizer_temporal_robot_preserve
```

Optional diagnostic conditions:

```text
composite_color_match             # cheap non-generative baseline
composite_harmonizer_world_only
```

Never overwrite `composite_raw`. Every harmonized episode must retain raw policy-camera frames or deterministic hashes plus sampled diagnostic frames.

---

# Deliverables

## Code

- `integrations/harmonizer/server.py`
- `integrations/harmonizer/protocol.py`
- `integrations/harmonizer/README.md`
- `robo/rendering/observation_pipeline.py`
- `robo/rendering/harmonizer_client.py`
- updates to `robo/rendering/pi05_render.py`
- updates to `robo/envs/pi05_env.py`
- updates to `robo/eval/pi05_eval.py`
- updates to `robo/eval/paired_runner.py`
- updates to `robo/manifest/schema.py`
- updates to `robo/manifest/io.py`
- `agents/eval/harmonizer_eval.py`
- `agents/eval/harmonizer_policy_report.py`

## Configuration

- `configs/rendering/harmonizer.yaml`
- `configs/experiments/harmonizer_smoke.yaml`
- `configs/experiments/harmonizer_visual_eval.yaml`
- `configs/experiments/harmonizer_policy_ablation.yaml`
- pinned upstream commit and model/checkpoint hashes

## Runtime scripts

- `run/harmonizer/build_container.sh`
- `run/harmonizer/download_checkpoints.sh`
- `run/harmonizer/serve.sh`
- `run/harmonizer/smoke.sh`
- `run/slurm/harmonizer_visual_eval.sbatch`
- `run/slurm/harmonizer_policy_matrix.sbatch`

## Tests

- `tests/test_harmonizer_protocol.py`
- `tests/test_harmonizer_client.py`
- `tests/test_harmonized_renderer.py`
- `tests/test_harmonizer_manifest.py`
- `tests/test_harmonizer_temporal_reset.py`
- `tests/test_harmonizer_preservation.py`

## Documentation

- `docs/HARMONIZER_INTEGRATION.md`
- update `docs/ENVIRONMENTS.md`
- update `docs/ROBOT.md`
- update `docs/DATA_AND_WEIGHTS.md`
- update baseline / license audit

---

# Work packages

## 23.0 — Dependency, model, and license audit

1. Pin the upstream Git commit.
2. Record code license, model license, Cosmos dependency, checkpoint names, and hashes.
3. Build the official container unchanged.
4. Download both temporal and exported non-temporal checkpoints.
5. Verify model loading on one Ampere GPU and one Hopper GPU when available.
6. Measure peak VRAM and cold/warm load time.
7. Confirm output size and dtype.

### Pass artifact

```text
validation/23/dependency_audit.json
```

Required fields:

```text
upstream_commit
code_license
model_license
checkpoint_hashes
base_model_hashes
container_digest
hardware
load_time_s
peak_vram_gb
```

### Stop condition

If the model or base-model license is incompatible with planned release or use, stop before integration. Do not substitute unlicensed weights.

## 23.1 — Offline zero-shot smoke test

Before touching policy code:

1. Export 100 consecutive raw SimAny composite frames from one exterior-camera sequence.
2. Run official non-temporal inference.
3. Run official temporal inference.
4. Confirm dimensions, frame order, history warm-up, and no crashes.
5. Produce raw / non-temporal / temporal side-by-side video.
6. Measure latency distribution and temporal consistency.

### Input sequences

Use at least:

- a static-camera sequence with moving object;
- a moving exterior camera or changing robot configuration;
- a wrist-camera sequence;
- a sequence containing transparent or reflective objects;
- a deliberately bad clean-background / object-boundary case.

### Fast command

```bash
bash run/harmonizer/smoke.sh \
  --input validation/23/raw_sequence \
  --out validation/23/offline_smoke
```

### Acceptance

- every output frame exists and matches the input frame index;
- no temporal state crosses camera streams;
- temporal reset reproduces the first-frame behavior;
- raw inputs are unchanged;
- identical input + identical history produces equivalent output within a declared numeric tolerance.

## 23.2 — Service and client integration

1. Implement the server with one persistent model instance.
2. Add `health` and `model_info` endpoints.
3. Add per-stream temporal ring buffers.
4. Implement explicit `reset_stream`.
5. Add request IDs and strict monotonic frame-index checking.
6. Implement client-side timeout and error classification.
7. Add server latency and transport latency separately.

### Failure policy

For benchmark runs:

- timeout, malformed response, model error, or history mismatch becomes `enhancer_failure`;
- the episode remains in the denominator;
- no fallback to raw RGB is permitted.

For demo-only runs, optional raw fallback may be enabled but must be watermarked in metadata and excluded from metrics.

## 23.3 — Layered renderer and preservation modes

1. Refactor `CompositeObs` to expose `RenderLayers`.
2. Preserve backward-compatible raw rendering.
3. Add `ObservationPipeline` that composes renderer, enhancer, and preservation blend.
4. Reset enhancer history from `DroidSimEnv.reset()`.
5. Save diagnostic layers for selected frames:
   - world RGB;
   - raw composite;
   - enhanced frame;
   - robot mask;
   - restored result;
   - difference heatmap.
6. Validate two-camera behavior in the same control tick.

### Preservation tests

- robot core pixels after restoration equal raw robot pixels exactly;
- no enhancement leaks across episode reset;
- target-object masks remain measurable after enhancement;
- hard and feathered restoration produce no invalid RGB or shape changes.

## 23.4 — Manifest and experiment-contract integration

Add observation-enhancer fields to every rollout manifest:

```text
enhancer_name
enhancer_upstream_commit
enhancer_checkpoint_sha256
enhancer_base_model_sha256
enhancer_mode
enhancer_resolution
enhancer_timestep
enhancer_temporal_offsets
enhancer_dtype
enhancer_robot_preserve_mode
enhancer_mask_parameters
enhancer_server_hardware
enhancer_fallback_policy
```

The observation enhancer is an explicit treatment variable. A raw-vs-Harmonizer ablation may differ only in enhancer-related manifest fields; robot, policy, camera, controller, reset, task, and rubric fields remain frozen.

Add provenance checks so a result table fails generation when:

- checkpoint hash is missing;
- upstream commit is missing;
- temporal mode lacks an offset list;
- a benchmark episode used raw fallback;
- streams were not reset between episodes;
- raw and enhanced conditions used different reset-state IDs.

## 23.5 — Visual evaluation

### Datasets

Use paired or approximately paired data in this order:

1. **Hidden-oracle rendered sequences:** exact camera and state correspondence where available.
2. **ScanNet++ held-out views:** static-scene reconstruction artifacts and reinserted-object appearance.
3. **DROID aligned sequences:** real robot footage vs reconstructed environment and robot overlay, only where metric camera alignment passes its gate.
4. **Phone pilot:** qualitative and runtime evidence only unless pixel alignment is independently verified.

Split by scene, never by frame.

### Metrics

#### Global image metrics

- PSNR and SSIM only for pixel-aligned pairs;
- LPIPS;
- DINO feature distance;
- FID or KID at dataset level when sample count is sufficient.

#### Task-local metrics

Within manipulated-object, target, support, obstacle, and robot masks:

- LPIPS / DINO distance;
- target re-detection confidence;
- target-mask IoU between raw and enhanced predictions;
- robot silhouette preservation;
- boundary displacement;
- color / illumination discontinuity across insertion boundaries;
- residual ghosting score.

#### Temporal metrics

- temporal LPIPS;
- optical-flow warping error;
- frame-to-frame feature flicker;
- target-detection stability over time.

#### Runtime metrics

- model load time;
- p50 / p90 / p95 inference latency per camera;
- transport latency;
- peak VRAM;
- total rollout slowdown.

### Required baselines

- raw composite;
- simple non-generative color matching;
- Harmonizer non-temporal;
- Harmonizer temporal;
- temporal + robot preservation.

Do not compare against an unavailable method by assigning zero or `N/A` as a numeric result.

## 23.6 — Hallucination and task-preservation gates

A visual metric gain is insufficient. Before policy evaluation, require:

1. no systematic loss of task-object detections;
2. no systematic target-boundary displacement;
3. exact robot-core preservation in the selected main mode;
4. no cross-episode temporal leakage;
5. no large increase in task-local temporal flicker;
6. manual review of the worst metric-ranked examples, selected automatically rather than cherry-picked.

Predeclare a `bad_harmonization` label from task-preservation failures and report its rate. Failed frames remain in denominators.

If full-frame temporal mode improves image realism but damages task-object preservation, use temporal robot-preserving mode as the main condition. If all generative modes damage task evidence, keep Harmonizer as a demo-only component and do not include it in the paper method.

## 23.7 — Frozen-policy ablation

Run only after visual and preservation gates pass.

### Conditions

```text
composite_raw
composite_color_match
composite_harmonizer_nontemporal
composite_harmonizer_temporal_robot_preserve
```

### Protocol

- same verified policy checkpoint;
- same raw scene build;
- same robot/cameras/controller;
- same reset bank;
- same task language and rubric;
- same simulated horizon;
- observation treatment is the only allowed difference.

### Metrics

- staged-progress mean and paired delta;
- task success when non-degenerate;
- non-zero progress rate;
- action-sequence divergence under matched states;
- target approach distance;
- grasp / lift / place stage agreement;
- build and enhancer coverage;
- hierarchical bootstrap confidence intervals.

Report policy results only when the matrix contains enough non-degenerate task-policy pairs under the existing research contract. Do not promote a single favorable scene or demo episode to a general claim.

### Interpretation

A positive result supports:

> Online harmonization improves task-relevant visual observations and can improve frozen-policy behavior in reconstructed simulators.

It does not support:

> Harmonization improves scene geometry, physics, or real-world policy success.

## 23.8 — Optional SimAny-specific fine-tuning

Start only when pretrained zero-shot evaluation shows a consistent gain but clear domain-specific residual errors.

### Pair construction

#### A. Static reconstruction correction

```text
input: SimAny render at a held-out real camera
 target: aligned real frame
```

#### B. Object reinsertion

```text
input: clean background + object asset reinserted at original pose
 target: original captured frame
```

Filter pairs where object identity or geometry mismatch is too large; otherwise the model may learn shape hallucination rather than appearance harmonization.

#### C. Robot composition

```text
input: reconstructed environment + MuJoCo robot overlay
 target: aligned DROID frame
```

Use only sequences with independently validated camera and robot-base alignment.

### Training protocol

1. fine-tune non-temporal model first;
2. evaluate held-out scenes;
3. add temporal training only after per-frame gains are established;
4. split by room / lab, not frame;
5. retain upstream checkpoint as a fixed zero-shot baseline;
6. record all data provenance and licenses.

### Losses

- pixel reconstruction where alignment is valid;
- LPIPS;
- task-object feature preservation;
- temporal warping / TV loss for temporal stage;
- optional robot-mask preservation loss.

Do not train against policy reward in this task. That would change the paper question and risk policy-specific visual hacking.

---

# Test matrix

## Unit tests

- protocol serialization and malformed requests;
- stream reset and monotonic frame IDs;
- RGB normalization / denormalization;
- fixed-size resize and restoration;
- robot mask feathering;
- model metadata and checkpoint hashing;
- enhancer failure classification.

## Integration tests

- 20-frame exterior sequence;
- 20-frame wrist sequence;
- two cameras interleaved;
- episode reset at frame 10;
- server restart between episodes;
- paired raw / enhanced manifests with only declared observation fields differing;
- raw fallback rejected under benchmark mode.

## Regression tests

- `enhancer=none` reproduces current `CompositeObs` output;
- raw policy condition produces the same observation hashes as before the refactor;
- full-room physics states and task scores are unchanged by enabling frame logging;
- enhancer does not modify `qpos`, object poses, contacts, or scorer state.

---

# Acceptance gates

## G23.1 — Upstream reproducibility

Official pretrained non-temporal and temporal inference both run from a clean pinned environment with recorded model hashes.

## G23.2 — Runtime integration

A 100-frame, two-camera episode completes with correct per-camera history, no temporal leakage, complete manifests, and no silent fallback.

## G23.3 — Task preservation

The selected main mode preserves robot geometry by construction and does not materially degrade target detection, target-mask consistency, or temporal stability on held-out scenes.

## G23.4 — Visual gain

Harmonizer improves at least one predeclared task-local perceptual metric over raw composite and simple color matching on held-out scenes, without failing G23.3.

## G23.5 — Robotics value

To enter the main paper, harmonized observations must improve a predeclared policy-level or oracle-aligned metric with matched resets, or produce a clear selective benefit under the task-local audit. Purely cosmetic improvement stays in the video / supplement.

## G23.6 — Real-time wording

Use the phrase `real-time at 15 Hz` only when measured p95 end-to-end enhancement latency for both required policy cameras satisfies the actual control-period budget on the stated hardware. Otherwise report exact wall-clock slowdown and call the module `per-step online enhancement`.

---

# Paper artifacts unlocked

## Main-paper candidate

A figure with:

```text
real / raw SimAny composite / Harmonizer / difference map
```

for exterior and wrist views, with task-object and robot masks overlaid.

A compact table:

| observation | task-local LPIPS | target retention | temporal error | policy progress | latency |
|---|---:|---:|---:|---:|---:|
| raw composite | | | | | |
| color match | | | | | |
| Harmonizer non-temporal | | | | | |
| Harmonizer temporal | | | | | |
| temporal + robot preserve | | | | | |

## Claim decision

- **Visual + policy gain:** include as a method component and observation ablation.
- **Visual gain only:** include as rendering infrastructure / video result, not a core contribution.
- **No held-out visual gain:** cite Harmonizer as a baseline and remove the integration from the submission path.
- **Hallucination or task-preservation failure:** keep raw composite as the main renderer and report the negative result if space permits.

---

# Fast validation command

```bash
# 1. Start pinned Harmonizer service
bash run/harmonizer/serve.sh --mode nontemporal --device cuda:0

# 2. Protocol and renderer tests
pytest -q \
  tests/test_harmonizer_protocol.py \
  tests/test_harmonizer_client.py \
  tests/test_harmonized_renderer.py \
  tests/test_harmonizer_temporal_reset.py \
  tests/test_harmonizer_preservation.py

# 3. Real-scene smoke
python -m robo.eval.paired_runner \
  --config configs/experiments/harmonizer_smoke.yaml \
  --episodes 2

# 4. Visual report
python -m agents.eval.harmonizer_eval \
  --config configs/experiments/harmonizer_visual_eval.yaml \
  --out validation/23/visual_report
```

## Pass artifacts

```text
validation/23/dependency_audit.json
validation/23/offline_smoke/
validation/23/service_smoke/
validation/23/visual_report/metrics.json
validation/23/visual_report/worst_cases/
validation/23/policy_report.json
```

# Agent handoff requirements

The handoff must state:

- exact SimAny commit;
- exact Harmonizer upstream commit;
- exact checkpoint and base-model hashes;
- container digest and GPU model;
- commands run;
- p50 / p95 latency and peak VRAM;
- output paths;
- whether any frame used fallback;
- task-preservation failures;
- all acceptance gates passed or failed;
- whether the result is suitable for main paper, supplement/video only, or should be dropped.
