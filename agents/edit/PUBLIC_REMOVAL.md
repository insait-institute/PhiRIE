# Automatic TRAIN-only removal preparation

The opt-in path extends `inpaint_prepare.py`; invoking it without flags preserves
the historical factory interface. It prepares masks, standard support-plane
estimates, and projected TRAIN views. It does not produce a cleaned background,
invoke an inpainting model, evaluate a scene, or certify simulation readiness.

```bash
python -m agents.edit.inpaint_prepare \
  --public-context configs/experiments/icra2027/NEW/removal_SCENE.yaml \
  --contract-manifest outputs/icra2027/NEW/contract/freeze_manifest.json
```

The context is an E0-bound YAML file in the executing clean checkout:

```yaml
schema_version: 1
scope: automatic_train_only_removal_preparation
freeze_id: NEW
scene_id: SCENE
policy_id: A4
materialization_manifest: {path: ABSOLUTE_MANIFEST, bytes: INTEGER, sha256: SHA256}
generation_config: {path: ORIGINAL_TRELLIS_CONFIG, bytes: INTEGER, sha256: SHA256}
seed: 0
algorithm: {radius_m: 0.03, top_k_views: 3, ring_m: 0.10, asset_samples: 5000}
```

The public automatic materializer must have produced this A4 factory in a separate immutable
source directory. Its destination and source closures remain authoritative.
The same public validator runs in the recorded clean materializer worktree,
which may belong to an older freeze; source commits and manifests are never
relabeled. The new preparation context/E0 can therefore bind completed inputs
without a circular dependency on outputs from its own freeze. The original TRELLIS generation config is bound
to the sealed canonical initial-pool receipt. Existing `discovery_binding` and
Gaussian provenance validators authenticate the complete source TRAIN roster,
RGB content, official split, cameras, source E0, and trained Gaussian. All
registered camera metadata may be parsed, but only the frozen TRAIN frames are
ranked or projected. No TEST RGB or scan annotations are consumed. Public
preparation calls the explicit automatic instance loader; the materialized
objects must use `automatic_instance_id` and may not carry a GT identity alias.

The existing removal radii, view scoring/top-K, ring selection, plane fitting,
and mask projection are retained. Raw canonical `trellis_mesh.ply` samples use
seed 0; the selected alignment is applied exactly once. Materialized
`mesh_sim.obj` is not used as a canonical mesh. Missing or malformed accepted
assets fail the whole preparation; there is no asset-removal fallback.

Output is the fresh `NEW_FREEZE/fidelity/removal/SCENE/inpaint/` directory, with the existing per-object
files, `removal_union_idx.npy`, and `prepare_meta.json`, plus
`public_prepare.json` and an exact-member SHA256 `seal.json`. The complete
planned denominator includes rejected and abstained objects. No-plane, no-view,
and non-converged trimming are explicit statuses. Per-view projected pixel counts
also expose selected subpixel objects with empty masks. These are not successful background
completion. Zero accepted objects is valid `NO_ACCEPTED_OBJECTS` and removes
nothing. The downstream background producer must separately handle these
statuses and authenticate its actual model/backend; this stage never labels an
untouched background plus inserted assets as cleaned factorization.

An exclusive claim in `NEW_FREEZE/fidelity/removal/SCENE/` prevents concurrent or repeated writes. Existing
output directories, including empty ones, are rejected. Runtime failures retain
`inpaint_failed_partial/` and `inpaint_public_failure.json`; a new attempt needs a
new source/config freeze. The source factory, including every materialized object artifact, remains
unchanged. The downstream fill worker must explicitly consume the new inpaint
directory; no old factory directory is mutated to make a legacy path resolve.

CPU smoke (synthetic inputs, including real projection/removal and negative
source, identity, transform, overwrite, and unavailable-support cases):

```bash
mkdir -p outputs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. python -m pytest -q tests/test_public_inpaint_prepare.py \
  --basetemp=outputs/public_removal_tests
```
