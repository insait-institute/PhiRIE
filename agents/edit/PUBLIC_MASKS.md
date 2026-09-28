# Automatic TRAIN-only SAM3 removal masks

This opt-in stage extends `inpaint_masks.py` and retains the existing projected-mask
union, strict IoU > 0.15 selection, 9-pixel dilation, SAM3 confidence 0.4, and
1008-pixel processing resolution. Legacy invocation remains available. The public
path never loads GT aliases, TEST images, alternative checkpoints, or a fallback
segmenter. It produces masks, not an erased image or cleaned Gaussian background.

```bash
SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT} PYTHONDONTWRITEBYTECODE=1 \
PINNED_SAM3_PYTHON -m agents.edit.inpaint_masks \
  --public-context configs/experiments/icra2027/NEW/masks_SCENE.yaml \
  --contract-manifest outputs/icra2027/NEW/contract/freeze_manifest.json
```

The new E0-bound context references completed immutable preparation from a prior
freeze; neither that preparation nor its source materialization is modified:

```yaml
schema_version: 1
scope: automatic_train_only_removal_masks
freeze_id: NEW
scene_id: SCENE
preparation:
  directory: OLD_FREEZE/fidelity/removal/SCENE/inpaint
  seal: {path: ABSOLUTE_SEAL, bytes: INTEGER, sha256: SHA256}
  context: {path: ABSOLUTE_PREPARATION_CONTEXT, bytes: INTEGER, sha256: SHA256}
  contract: {path: ABSOLUTE_PREPARATION_E0, bytes: INTEGER, sha256: SHA256}
  producer_commit: ORIGINAL_PREPARATION_COMMIT
sam3_source: {path: ABSOLUTE_SOURCE, tree_sha256: TREE_SHA256}
sam3_checkpoint: {path: ABSOLUTE_CHECKPOINT, bytes: INTEGER, sha256: SHA256}
python: ABSOLUTE_PINNED_INTERPRETER
runtime: {packages_sha256: HASH, targeted_bytes: TARGETED_RUNTIME_IDENTITY}
seed: 0
algorithm: {min_iou: 0.15, dilate: 9, confidence: 0.4, resolution: 1008}
```

Existing source/checkpoint/runtime fields are recorded in
`configs/experiments/icra2027/e3_discovery_cohort.yaml`: use `sam3_source`,
`sam3_checkpoint`, `sam3_python`, and `runtime.sam3`. The source tree hash is
`73c418359155da5da839853613260e84f48e5bd8e3d494c6614ca9e561a187ae`;
the 3,450,062,241-byte checkpoint hash is
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
These are identities to verify before creating an execution freeze, not permission
to reuse a changed runtime. No installation or model download occurs here.

Every preparation seal member, original E0/config/commit binding, automatic object
identity and terminal action, selected TRAIN frame, projected mask, RGB content,
and image size is authenticated. The original materializer validator authenticates
its own source closure in the recorded clean checkout. This does not rerun the
preparation stage or require its old commit to match the new mask producer.

Output is `NEW_FREEZE/fidelity/removal_masks/SCENE/`, with per-view
`obj_ID/mask_INDEX.png`, `public_masks.json`, and exact-member `seal.json`.
The sibling exclusive claim prevents repeat writes, including existing empty
outputs. Runtime failures retain a typed failure JSON, all planned view rows,
and any partial products. A failed bundle is not consumable as successful masks.
Pinned source, checkpoint, runtime, preparation, and RGB bytes are checked again
after inference and before sealing; detected mid-run drift retains failure
products. One SAM3 instance handles all nonempty views; image and prompt inference is shared
within each frame. Model/source/runtime failures cannot become projected-mask
fallback success. A valid SAM3 result with zero candidate masks retains the
predeclared projected-mask union and records zero candidates explicitly.

All planned objects remain in `result.objects`, including rejected/abstained
objects and accepted objects with `NO_PLANE`, `NO_VIEW`, or empty projections.
An empty projected view has status `EMPTY_PROJECTED_MASK` and no generated mask.
Zero applicable views skip model loading and return `NO_APPLICABLE_VIEWS`.
`plane_status` is a direct field of each accepted object entry. A `NO_PLANE` object
can have a `MASK_READY` projected view; a subsequent plane-dependent erase stage
must preserve `NOT_APPLICABLE_NO_PLANE` and must not count that view as fillable.

The read-only consumer API `validate_public_masks(directory)` rejects missing,
tampered, extra, unsealed, failed, relabeled, and incomplete products. It returns:

- `directory`, `seal_identity`, `context`, `context_identity`, `contract_identity`;
- authenticated `preparation`, `source_factory`, complete automatic `objects`,
  and `train_images` keyed by frame with original content identities;
- `rows` containing `object_slot`, `automatic_instance_id`, `view_index`, `frame`,
  `status`, nullable `mask_identity`, `projected_pixels`, and `final_pixels`;
- `result`, including the unchanged preparation object statuses and complete
  planned object/view denominators.

The API performs no inference and can be consumed by a separately frozen erase
stage. It neither mutates old outputs nor promotes masks to background success.
CPU smoke uses a mocked SAM3 processor with actual image/mask encoding, exact
threshold and dilation checks, source/result tampering, unavailable views,
zero accepted objects, runtime failure, and no-overwrite tests:

```bash
mkdir -p outputs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. python -m pytest -q tests/test_public_inpaint_masks.py \
  tests/test_public_inpaint_prepare.py tests/test_e3_factory_materializer.py \
  --basetemp=outputs/public_masks_tests
```

No real-data pilot, GPU inference, cleaned-background result, or scientific claim
is produced by this source-only handoff.
