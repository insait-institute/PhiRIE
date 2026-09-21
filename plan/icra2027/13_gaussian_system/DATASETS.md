# Dataset acquisition, organization, and scene curation

## 1. A dataset capability is not an executable robot benchmark

The first inventory pass covers the following catalogue. `policy_ready` remains false until native task, robot, checkpoint, controller, camera, import and rubric checks pass. A missing permission or native adapter is BLOCKED/NOT_APPLICABLE, not a zero-success baseline. Do not download every large dataset before proving one usable sample.

| Dataset | Primary role | Prospective measured subset | Appearance / diversity | Native limitation |
|---|---|---:|---|---|
| RoboCasa365 | Main same-engine reconstruction-to-manipulation | 12 layouts x 3 rigid tasks x 2 canonical instances = 72 builds | Contemporary kitchens, islands, counters, daylight/material variation | Preserve the pinned MuJoCo/robosuite version and compatible policy. Old 48-instance results are historical, not fresh unseen TEST. |
| BEHAVIOR / OmniGibson | Independent multi-room rigid-task extension | 8 rooms x 3 admitted task instances = 24 builds | Kitchen, dining, living and office contexts | Native PhysX/BDDL. Checkpoint availability, object states and role bindings require a genuine pilot. No cloth/liquid/articulation claim from rigid assets. |
| ReplicaCAD Interactive | Interactive rearrangement and controlled context | 24 layouts; manipulation only after Habitat admission | Artist-created apartments and furnishing variations | Many layouts share one source apartment family. Do not count micro-variations as unrelated rooms. |
| ReplicaCAD Baked Lighting | Appearance / background completion | Same 24 paired layout identities where available | Better global-illumination appearance | Baked variant contains static furniture; do not silently replace interactive physics with baked scenery. |
| HSSD | Whole-scene reconstruction / semantics / DEMO | 30 scenes | Human-authored living rooms, dining rooms, bedrooms, offices | Primarily scene/navigation data. Robot task creation is a separate adapted benchmark, not an official manipulation result. |
| Hypersim | High-quality appearance and missing-region evaluation | 40 scenes | Professional interior lighting/materials | Released RGB/depth/semantics are not a turnkey dynamic simulator. Do not assume source asset redistribution or clean-background render access. |
| ScanNet++ | Real RGB-video / Gaussian construction | Keep original 50-room evidence; freeze 20 additional video rooms | Real clutter/material/sensor conditions | Raw video experiment must estimate cameras and scale, not inherit scan/GT depth. Check Chorus pretraining overlap. |
| Replica | Real-scan / semantic and appearance robustness | 10 available scenes | Clean scans suitable for fly-throughs | Static scanned surfaces are not separately movable GT assets. |
| HM3D | Broader real-scene reconstruction | 20 allowed, source-selected scenes | Larger diverse real homes | Navigation meshes do not establish object-level physics or labels. |
| ProcTHOR | Scale/generalization diagnostic | 30 houses | Layout diversity; less suitable for hero footage | AI2THOR native task/controller adapter is optional. No third/fourth backend should block main results. |
| InteriorVerse | Optional photorealistic visual stress test | Up to 10 public/authorized scenes | High-quality interior assets | Full access/licensing may require application. Skip quantitatively if not licensed, retaining the admission record. |
| DROID | Real observation / alignment external evidence | Existing 9 workspaces | Actual robot workspaces | Demonstration video is not a new autonomous robot trial. Do not mix moving episodes into one static GS. |
| Consented phone video | Qualitative RGB-video system demonstration | 3 rooms, separate DEMO split | Deliberately well-lit, visually legible scenes | Consent and API-upload permission separate. No physical policy-success claim. |

These are campaign targets, not availability promises or measured counts. Formal full means all rows of the sealed eligible roster, not every frame of every listed dataset. Failure to acquire enough scenes triggers an explicit roster revision BEFORE outcomes, never fabricated IDs or silent substitution.

## 2. Where the best presentation scenes should come from

For hero manipulation use RoboCasa first: a daylight kitchen with a broad visible counter, a colored mug/fruit and an open destination. For room-level fly-through use HSSD living/dining rooms and ReplicaCAD baked-lighting apartments. For fine shadow/material examples use Hypersim, but label offline reconstruction footage rather than robot simulation. Add one honest real phone room to show input reality.

Create source-only HTML/contact galleries with `robo.campaign gallery`. Record room type, lighting, material diversity, source resolution, visible interaction workspace and permission. Human aesthetic selection is allowed only for the separately declared DEMO split. TEST comes from the source inventory, strata and stable hash order, never downstream reconstruction quality or policy outcomes. Preserve hard scenes, reflective objects and dark corners in quantitative evaluation.

Do not claim that specific scene IDs were visually reviewed until actual source thumbnails are rendered. Avoid inventing IDs based on paper pictures.

## 3. Disk layout

```text
DATA_ROOT/
  upstream/<dataset>/<release>/                 # read-only original files
  indexes/<dataset>-<release>.jsonl             # genuine source scene index
  licenses/<dataset>/<approval-record>.json     # permissions, no tokens
  canonical/<dataset>/<scene-id>/<instance>/
    public/
      train_video.mp4                          # source for raw-video setting
      frames/                                  # TRAIN only
      task_instruction.txt
      capture_manifest.json
    evaluator_private/
      reference_scene/                         # native XML/USD/assets
      heldout_views/                           # not mounted to constructors
      labels/                                  # IDs, geometry, clean backgrounds
    estimated/
      cameras/                                 # actual video-estimated poses
      scale.json                               # source and observation evidence
      gaussians/                               # immutable original row IDs
      semantic/                                # Chorus + text-space identity
    objects/<instance-id>/
      observations/                            # shared crop/mask/view budget
      proposals/<generator>/<seed>/
      registered/<recipe>/
      exported/<recipe>/
  campaigns/<campaign-id>/
    inventory.jsonl
    frozen/scenes.jsonl
    tasks/                                      # finite stage DAGs
    model_locks/                                # source/weight file hashes
    runtime/                                    # interpreters and native admission
    outputs/                                    # immutable component artifacts
    robot_ledgers/                              # EXISTING canonical harness schema
    tables/
    demo/
```

Do not copy original video or candidate pools per seed/reset/method. Use immutable content-addressed artifacts; validate symlink/permission constraints of the receiving constructor sandbox. Each physical scene/family keeps one train/dev/test/demo assignment, including duplicates across datasets. Store shared asset families, policy-training overlap and Chorus-pretraining overlap as additional strata, not merely file-level deduplication.

## 4. Acquisition commands and gates

Use official installation/download instructions linked in SOURCES.md, pin their revision, then inventory the downloaded index. Prefer current authorized local copies.

ReplicaCAD's official utility supports:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset replica_cad_baked_lighting \
  --data-path "$DATA_ROOT/upstream/replicacad"
```

RoboCasa: use the pinned installation's asset downloader and enumerate its real layout/task registry. BEHAVIOR: use its licensed dataset installation and pre-sampled task instances. HSSD/ScanNet++/HM3D/Hypersim/InteriorVerse: obtain the official metadata and authorized release first, not a guessed public archive URL. No automatic acceptance of legal agreements, gated access bypass, or API image upload.

Download and test one scene before scheduling the complete eligible dataset. Training/data extraction can use compute GPUs; OmniGibson/Isaac rendering must pass compatibility on an available supported graphics GPU rather than assuming an H200 is equivalent to RTX.

## 5. Capture contract

For simulator-derived scenes, freeze task instance BEFORE method execution. Capture one static TRAIN video at native 1280x720 or higher, 100–200 distributed views after deterministic subsampling. Hold out a separate camera path, not every nth nearly identical frame alone. Keep room and object states fixed during scanning. Record optional robot visibility and camera-intrinsic knowledge.

Sensor rows: raw RGB-video (estimated cameras), posed RGB (known poses explicitly), ideal RGB-D (diagnostic), noisy/observed RGB-D (separate). Metric scale comes from estimated prior, known robot geometry or declared marker calibration; monocular SfM alone does not establish metres. Our executable COLMAP/GS adapter explicitly emits `metric_scale=NOT_ESTABLISHED` until a measured calibration is attached.

For inpainting, true holes with no observed background have no valid pixel GT in real video. Use hidden clean reference renders at the SAME state for synthetic paired evaluation, or predeclared visible-surface holdouts for real data. Unknown pixels are not scored as a known background. Eval-only clean images must not be mounted into any constructor/editor, including a commercial API request.

## 6. Coordinate / photometric normalization

Hypersim `depth_meters` is Euclidean camera distance, NOT camera-Z. Convert with the per-pixel ray norm before feeding state warp or backprojection. Its camera convention looks down negative Z with Y up, and translation is in asset units; use declared `meters_per_asset_unit` only on the evaluator/posed-RGB-D side, never to silently calibrate raw-RGB-video construction. Use lossless HDR/HDF5 with a fixed declared tone map for metrics, not the convenient lossy preview JPGs. Exact source camera conversion, distortion, depth definition and RGB transfer function are part of each dataset adapter's real identity test.

## 7. Inventory schema

Use `catalog` to write required fields. `source_index` is an actual official/local index reference; `license_checked` is an operator review, not generated consent. Every `bindings` file is a `{path,sha256}` receipt produced after acquisition, not a hand-invented filename. `matrix` never downloads hidden references or invents missing object/task identities. Missing native bindings appear in `unbound.json`.

Before TEST, print independent scene-family counts, canonical builds, resets per build, shared asset counts, eligible task families, sensor regimes, native backends, licenses and pretraining overlap. A reset bank is not a set of independent reconstructions.
