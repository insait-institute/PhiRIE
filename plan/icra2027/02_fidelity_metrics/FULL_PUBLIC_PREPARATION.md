# Full-cohort TRAIN preparation in immutable batches

This stage reuses the sealed 09c1414f1b pilot and executes the existing public
removal preparation for every factory available at config creation. The first
batch freezes 38 additional scenes; 11 absent factory inputs remain explicit.
The population remains 50 scenes, 1,871 planned object jobs and 400 TEST views.
No TEST pixels, appearance metrics, SAM3, erasure or Gaussian fitting are read or
run by this stage. E4 qualification or export failures do not select scenes.

Every context, existing factory manifest and original generation config is a
real file hashed into E0. The availability snapshot identifies all 49 remaining
scenes, including null entries for unavailable factories; later availability
cannot change this batch. Further preparations require a new source/config
freeze. A final authenticated union must include all 50 scene preparations or
typed failures; this partial batch cannot publish a full-cohort aggregate.

The original fixed pilot remains a negative scene: one required accepted asset
has no support plane. The unchanged construction-only admission recipe retains
that scene's eight missing TEST views. All accepted object rows must have a
support plane, a selected TRAIN view and a nonempty primary projection before a
complete-background model job is admitted. A zero-accepted scene is explicit
unchanged-background evidence with zero accepted-object coverage.

The model pilot is the first structurally fillable scene with accepted assets
in the original lexicographic roster. `pilot-admission` authenticates the entire
prefix through that scene, including earlier structural blockers. An earlier code/environment preparation
failure blocks model pilot selection because it does not prove structural
unfillability. It waits if an earlier scene lacks a terminal preparation; it may
release the fixed pilot before later CPU preparations finish. No model quality
or TEST outcome selects this scene. Full inventory waits for all 50 scenes.

```bash
SIMANY_AUTO=1 SIMANY_NO_GT=1 SIMANY_MESH_SRC=derived \
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python -m run.icra2027.e2_full_preparation \
 --config configs/experiments/icra2027/e2_full_preparation_batch1/execution.yaml \
 --contract /group/worldcept/code/SimAny/outputs/icra2027/20260906-e7c60da-v2/contract/freeze_manifest.json \
 --phase run --scene 0d2ee665be
```

Use `--phase context` for immutable input admission, `--phase pilot-admission`
for the prefix gate and `--phase inventory` for complete-only coverage. Outputs
are under `20260906-e7c60da-v2/fidelity/removal/<scene>/inpaint`, with typed
`preparation_failures` retained. Jobs use 4 CPUs, 32 GB and 30 minutes, no GPU,
ordinary independent submissions without arrays. Existing attempts and seals
cannot be overwritten. CPU mocks include source/config/snapshot drift,
no-overwrite, all-object structural failure and missing-prefix refusal.

The prior 20260906-e7c60da-v1 contract/context-only attempt is preserved; no scene
preparation or model job ran before the failure-prefix guard correction.
