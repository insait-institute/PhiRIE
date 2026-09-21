# Runbook: actual CLIs, inputs, and scientific handoff

## 0. Implementation versus native admission

The committed `robo.campaign` package executes real model APIs, command adapters, numerical state correction, dataset splits and ordinary jobs. It does not contain new model weights or licensed benchmark assets. Official PolaRiS composition remains its documented assisted workflow; native multi-backend policy integration must pass each backend's own DEV gates. Do not substitute a synthetic test or a successfully parsed USD for a robot rollout.

Run the old `12_simulation_submission` jobs to close F1/F2/F3 in parallel. Preserve their immutable cohort/results. Do not overwrite the old `robo.roundtrip` pipeline with this campaign.

## 1. Pin a clean checkout and use existing isolated environments

```bash
git fetch origin
git worktree add --detach ../PhiRoom-system-campaign origin/main
cd ../PhiRoom-system-campaign
export SIMANY_CAMPAIGN_PY=/absolute/existing/driver-env/bin/python
bash run/campaign/run.sh --help
"$SIMANY_CAMPAIGN_PY" -m pytest -q tests/test_system_campaign.py
```

The driver needs Python3.10+, numpy, scipy, pillow, PyYAML and pytest for tests. Mesh tests use trimesh; trainable-model tests use torch. Do not pip-upgrade the fixed native or model environments. Each model worker uses its own `runtime.environments.<name>.python`. Preserve venv launcher symlinks. Native jobs source their previously validated EGL setup inside an allocation. No heavy native work on login nodes.

## 2. Inventory real scene IDs, then freeze

```bash
export RUN=/absolute/new-immutable-campaign-root
bash run/campaign/run.sh catalog --out "$RUN/catalog"
# Agent now inventories actual authorized assets and writes inventory.jsonl.
bash run/campaign/run.sh freeze \
  --inventory "$RUN/inventory.jsonl" --seed 2027 --out "$RUN/frozen"
bash run/campaign/run.sh gallery \
  --inventory "$RUN/frozen/scenes.jsonl" --out "$RUN/source-gallery"
```

An inventory row has `dataset,release,scene_id,group_id,split,source_index,license_checked`, plus actual `bindings` below. Dataset IDs are listed by `catalog`. Paths are resolved by the agent from existing releases and official source indexes. They are not requests for the user to fill hundreds of placeholders. Gallery reads source thumbnails only; choose DEMO separately.

`--counts /path/counts.json` optionally selects a fixed number per dataset/split. Counts are real inventory scene rows, not a promise of distinct physical rooms. Check family/layout quotas explicitly before freeze. The existing cohort collector defines native task/reset units.

## 3. Binding and checkpoint formats

Every input file is a receipt:

```json
{"path":"/absolute/real/file.png","sha256":"actual_SHA256"}
```

Generate receipts through `robo.campaign.core.receipt(path)`, never copy example hashes. Example inventory `bindings` fields:

```text
video: receipt(TRAIN capture video)
objects: [{object_id, images:[receipt(anchor_rgba),receipt(next_view_rgba),...]}]
inpaint_cases: [{case_id, rgb:receipt(rgb), mask:receipt(binary_mask)}]
gaussians: {gaussians:receipt(scene_ply), gaussian_ids:receipt(unique_int64_npy)}
visual_sequences: [{sequence_id, inputs:{rgb000:receipt(...),state000:receipt(...),buffers000:receipt(...)},
  frames:[{rgb:"rgb000",state:"state000",buffers:"buffers000"},...]}]
polaris_bundle: {bundle:"/absolute/native_bundle", acquisition:"declared actual protocol", manual_minutes:measured_value}
native_block: {environment:"native", params:{purpose:"native_policy",argv:[...],output_manifest:"{out}/native_outputs.json",...}, inputs:{...}}
```

Model manifest is `{"files":{"relative/weight.file":"SHA256",...}}`. Snapshot manifests must bind ALL files the model loader reads, including config/tokenizer/backbone dependencies. Single-file manifests use the actual filename. Use offline cache/local snapshots and a pinned third-party source commit. Our loader validates listed identities; operator must not omit required dependency files. Harmonizer's pkl alone is not the Cosmos base closure.

Copy `models.template.yaml` and `runtime.template.yaml` outside tracked source and resolve real paths/IDs/resources. Mark only usable stage backends enabled. `execution_ready` stays false until a real stage smoke succeeds. First smoke can use a separately recorded DEV-only runtime after import/environment checks. Do not mark TEST admission from CPU tests.

## 4. Build finite component tasks

```bash
bash run/campaign/run.sh matrix \
  --scenes "$RUN/frozen/scenes.jsonl" \
  --models "$RUN/models.resolved.yaml" \
  --stage generator --stage inpaint --stage chorus \
  --out "$RUN/components"
```

Other actual stages: `video`, `harmonizer_sequence`, `official`, `native`. Read `tasks.jsonl` and `unbound.json` before launching. Matrix generation does not create source captures, native task IDs or hidden GT. A row lacking native admission remains unbound. Independent stage DAGs can be sealed as artifacts become available, rather than waiting for all optional APIs/datasets.

Custom dependency tasks reference prior artifacts:

```json
{"task":"earlier-task-id","artifact":"gaussians"}
```

Task kinds are defined in `runner.KINDS`. Parameters are the actual adapter signatures, not arbitrary shell fragments. `semantic_instances` takes `coords,features,query,gaussian_ids,text_metadata`, all receipts or dependency artifacts, with matching text_space and `coordinates_unit:"m"`. `text_query` uses the matching pinned SigLIP2 checkpoint. `rgb_video_reconstruction` invokes the official COLMAP conversion and 3DGS trainer from a TRAIN video. Its output remains SfM gauge until a measured metric transform is attached.

## 5. Prepare, dry-run, then execute ordinary jobs

```bash
bash run/campaign/run.sh prepare \
  --tasks "$RUN/components/tasks.jsonl" \
  --runtime "$RUN/runtime.resolved.yaml" --out "$RUN/jobs-components"
bash run/campaign/run.sh launch --bundle "$RUN/jobs-components" --max-jobs 1
# Submit only after inspecting the printed resources/worker command.
bash run/campaign/run.sh launch --bundle "$RUN/jobs-components" --max-jobs 1 --submit
```

The launcher uses explicit Slurm args, atomic submission intents, dependency receipts and its own RUNNING/PENDING cap. Also respect other user jobs/account limits. Inspect the first genuine output, then release up to the approved cap:

```bash
bash run/campaign/run.sh launch --bundle "$RUN/jobs-components" --max-jobs 4 --submit
```

Repeat only while real dependencies become ready. No job arrays. If `sbatch` fails ambiguously, inspect `squeue/sacct` before retry; do not delete an intent to pay/run twice. A failed method artifact and a scheduler/setup failure are different. Failed upstream jobs remain unresolved downstream, not fabricated completed negative results.

For independent same-model tasks, `worker-batch --bundle ... --task id1 --task id2` runs in ONE already allocated model environment and caches weights. Do not also submit those IDs through `launch`. This excludes native-policy and temporal sequence jobs. Native experiments retain their original process/RNG contract.

## 6. Official methods and native bridge

SimFoundry kind calls its actual `scripts/pipeline/A_reconstruction/run.sh`; returned artifact is the official scene JSON. A cloud-stage approval file and per-scene upload permission are mandatory. Separate conversion/import receipts bind each resulting asset and estimated pose. The generic `asset_bridge` writes the EXISTING PhiRoom import contract; it does not fetch hidden reference pose.

PolaRiS kind validates an actual official `scene.usda`, `initial_conditions.json` and asset closure after its documented COLMAP/2DGS/assisted-composition workflow. It does not claim to automate that GUI or invent a reconstruction when inputs are absent. Record extra captures/human minutes. Native-end-to-end and common-importer adapted labels are distinct.

A `command` task executes an argv list (no shell interpolation), with `{out}` and named `{input}` path substitution. It must publish:

```json
{"artifacts":{"ledger":{"path":"/actual/episode_ledger.jsonl","sha256":"actual_hash"}}}
```

For native policy tasks, pass hash-bound `same_engine_gate` and `rubric_gate` files with `passed:true`, `fresh_reference:true` and the declared `official_method_labels`. Run the existing `robo.roundtrip.local_policy_instance` / `harness_runner` recipes in the admitted backend; an adapter wrapper only enumerates their real outputs into this manifest. Do not implement a second policy ledger. Backend task integration not yet available is NOT_RUN, not an official-method failure.

## 7. State correction and optional training

Buffers NPZ: depth HxW metres/camera-Z, visible ids HxW (0 static, -1 unknown), robot_mask/contact_mask/confidence HxW, normals HxWx3 for learned mode. State JSON: stream_id (episode+camera), contiguous frame_index, sim_state_sha256, K, T_world_camera and object_poses keyed by persistent numeric ID. Raw/enhanced images must share state and camera. All buffers derive from the reconstructed simulator, not evaluator GT.

`harmonizer_sequence` runs the official model and optionally our state correction over these frames. `state_visual` can process already frozen official outputs. Protection is exact only for declared robot/contact/edge pixels, not a global geometry guarantee. Unknown/disoccluded warp samples do not carry old hallucinations. Cache final emitted outputs, reset per episode/camera.

Optional learned prototype:

```bash
bash run/campaign/run.sh train-visual \
  --config "$RUN/train_visual.resolved.yaml" --out "$RUN/state-residual-training"
```

Run in an allocated torch environment. A pair record includes unique sample_id, scene_group, split, identical input/target state hashes, `physical_geometry_equal:true`, and receipt of NPZ containing `inputs(16,H,W)` and `target(3,H,W)`. Channels: raw RGB3, official teacher3, clipped camera-Z/10m1, normals3, confidence1, warped emitted RGB3, valid1, protect1. RGB is [0,1]. TRAIN/DEV scene families do not overlap. The code trains and saves best.pt chosen on DEV. Use its checkpoint/manifest under visual correction's `learned` config for inference. This small trained residual is our prototype, not NVIDIA's weights or a claim of a new pretrained foundation model.

## 8. Collect and hand off

```bash
bash run/campaign/run.sh collect \
  --bundle "$RUN/jobs-components" --out "$RUN/component-release-001"
```

This output is COMPONENT execution status, not a robot success table. Feed actual native ledgers and canonical metric producer receipts to the existing paper_pipeline; never count one model call as an episode. Use a new paper output directory first, without `--paper-root`, and inspect complete denominators and contrasts.

Return `AGENT_REPORT.md`, `RUN_SUMMARY.json`, exact resolved input/source/model identities, jobs.csv, component artifacts index, native ledgers/tables, per-dataset/per-task failures, query/visual/collision diagnostics, API reservations and actual usage, and source-only gallery. Add SHA256SUMS and a compact tar.gz, excluding weights, private raw data, credentials and all giant videos. Include a 30s continuous progress clip with true scope labels. Only source-generated measured values enter the paper.

## 9. Gaussian DC cache implementation

`gaussian_color_cache` is a real sparse regularized solver and PLY writer, not only a diagram. Inputs are standard SH-DC PLY, a correspondence manifest, and an NPZ with `pixel_index,gaussian_index,weight,residual_rgb,confidence`. Weights must be actual renderer alpha-compositing responsibilities for exactly the same immutable PLY/order. Targets are same-state linear-RGB appearance corrections from TRAIN/DEV observations or an admitted teacher, not hidden TEST reference images. The solver estimates bounded per-Gaussian RGB residuals and changes only f_dc_0/1/2. All other numeric attributes are checked unchanged. The actual renderer responsibility exporter is backend-specific and must be validated before use; nearest-Gaussian assignment is not a valid replacement. View-dependent shadows remain image-space. Geometry invariance of this update is not evidence of visual/policy improvement.
