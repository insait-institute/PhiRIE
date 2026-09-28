# Automatic TRAIN-only Gaussian background fill

The opt-in path extends the existing `inpaint_fill.py` carve, plane-grid fill,
and differentiable refinement producer. It consumes separately frozen automatic
preparation, SAM3 masks, and explicit LaMa erasure through the authenticated
`validate_public_erasure` handoff. It writes a new directory; source factories,
preparation, masks, erasures, and TRAIN images remain unchanged.

```yaml
schema_version: 1
scope: automatic_train_only_background_fill
freeze_id: NEW
scene_id: SCENE
erasure_seal: {path: ABSOLUTE_ERASURE_SEAL, bytes: INTEGER, sha256: SHA256}
python: ABSOLUTE_MINI_VIEWER_INTERPRETER
runtime: {packages_sha256: HASH, targeted_bytes: TARGETED_RUNTIME_IDENTITY}
algorithm:
  grid: 0.005
  hull_offset: 0.035
  iterations: 1500
  render_scale: 0.5
  disk: [0.004, 0.004, 0.0008]
  primary_view_index: 0
  mask_votes: 2
  seed: 0
  supervision: TRAIN_erasure_targets_only
```

Generate the runtime identity with the exact intended interpreter and disabled
user-site imports, then freeze the context in a clean source/E0 contract:

```bash
PYTHONNOUSERSITE=1 PINNED_PYTHON -m agents.edit.inpaint_fill --runtime
SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT} PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 PINNED_PYTHON -m agents.edit.inpaint_fill \
  --public-context CONTEXT.yaml --contract-manifest NEW/contract/freeze_manifest.json
```

The existing package-list hash and targeted executable/package RECORD/entrypoint
helper additionally cover gsplat, scipy, and plyfile. This is explicitly a
targeted runtime identity, not a complete operating-system or JIT-cache snapshot.
Source E0 binds the fill producer and common Gaussian loader/renderer. Runtime,
source Gaussian, camera, TRAIN RGB, and all predecessor bytes are authenticated
before execution and again before final sealing. Network and TEST-image reads
are prohibited. The source Gaussian and intrinsics derive from the original
TRAIN discovery manifest already bound to preparation; there is no caller-chosen
background substitution. Per-view poses must equal the original recorded cameras.

The grid spacing, footprint offset, disk scales, mask-vote carving, color
initialization, stable object-order compositing, view-0 supervision, 1,500-step
optimizer schedule, and comparison renderer retain the existing recipe. The
initial kNN colors remain the declared initialization for fill points outside
the primary view; they are not a fallback for a missing erasure target. The
strict path requires that target before running. Empty masks retain explicit
non-invocation status; another view cannot silently replace a missing primary.

Every planned object and erasure view remains in `public_fill.json`. Accepted
objects with no support plane, no view, or no successful primary erasure produce
`BLOCKED_UNFILLABLE_ACCEPTED`, retain their reasons, and write no
`clean_background.ply`. A sealed FAILED erasure cannot enter the fill stage.
Degenerate/empty fill grids, non-finite optimization, invalid removal indices,
or incomplete output arrays fail closed and retain partial artifacts. No
untouched background is labeled a successful fill for these objects.

A scene with zero accepted objects and an empty removal union is valid
`NO_REMOVAL`: it byte-copies the source Gaussian without optimization, retains
zero accepted coverage, and records `clean_background_is_unchanged_source=true`.
It is not evidence of successful object factorization or improved image quality.

Successful output is `NEW/fidelity/background_fill/SCENE/`, containing the
existing `fill_gaussians.npz`, `clean_background.ply`, comparison images,
`fill_stats.json`, `fill_stats_summary.json`, plus `public_fill.json` and an
exact-member SHA256 seal. The declared count identity is checked:
source Gaussians minus removed Gaussians plus fill Gaussians equals output
Gaussians. Every accepted object must have a fill slice. Output floats must be
finite. Existing output, claims, orphan failure receipts, and partial directories
are never overwritten; runtime failures retain the complete planned roster.

The existing hole PSNR is measured against model-erased TRAIN images. Its
`held` diagnostic means a secondary construction view withheld from the fill
optimizer; it is **not** an official TEST view or independent fidelity evidence.
These diagnostics must not populate Table II. Independent appearance metrics
must later use the existing fidelity evaluator on the separately frozen TEST
cameras. This source handoff launches no GPU job and claims no real-data result.

CPU smoke includes the actual complete 1,500-step optimizer with a small
differentiable renderer double, real image/NPZ/PLY publication, and negative
source, camera, runtime, denominator, unsupported-support, and no-overwrite tests:

```bash
mkdir -p outputs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=. python -m pytest -q tests/test_public_inpaint_fill.py \
  tests/test_public_inpaint_erase.py tests/test_public_inpaint_masks.py \
  tests/test_public_inpaint_prepare.py tests/test_e3_factory_materializer.py \
  --basetemp=outputs/public_fill_tests
```

For an immutable base interpreter missing schema dependencies, the strict
launcher may declare `SIMANY_FILL_DEPENDENCY_OVERLAY` and add that exact
directory after the executing checkout in `PYTHONPATH`. The runtime identity
then includes the complete overlay tree and actual Python/native import bytes
for Pydantic and its dependencies. The existing pre/post runtime equality checks
reject a missing, changed, or shadowed overlay. Bind the overlay and this launch
environment in the new E0/protocol before execution. The default runtime identity
is unchanged when no overlay is declared; no package is installed into the base
environment. A prior runtime-readiness failure requires a new freeze.
