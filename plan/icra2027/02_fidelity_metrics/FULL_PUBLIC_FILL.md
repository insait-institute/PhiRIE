# Full frozen TRAIN-only Gaussian background fill

Owner: E2 execution. Branch: `agent/icra-e2-full-fill`.

The existing strict `agents.edit.inpaint_fill` producer consumes authenticated
preparation, mask, and LaMa erasure bundles. Each stage writes to a new immutable
freeze and never modifies its predecessor. The fixed population remains 50
scenes, 1,871 planned object jobs, 399 accepted objects, and 400 planned TEST
views. The original first structurally fillable pilot is reused. Ten original
TRAIN structural blockers retain eight missing TEST views each; four scenes
with no accepted object copy their original background byte-for-byte under
`NO_REMOVAL`, which is not successful object construction.

For each of the 35 other structurally fillable scenes, preserve the existing
1,500-iteration fill, grid, carve, primary-view and seed-zero recipe. Gaussian
optimization uses TRAIN erasure targets only. Diagnostic hole PSNR and other
fit residuals are construction diagnostics, not independent TEST metrics.
No missing mask/erasure/raw-background fallback is permitted. Any failure stays
in the planned population and blocks that room's complete background condition.

Each positive scene runs as an ordinary single-GPU Slurm job with four CPUs
and 32 GiB host memory; zero-accepted no-ops use CPU jobs. The pinned mini-viewer
Python and existing immutable Pydantic overlay are required. No environment is
installed or changed. Positive full execution follows the authenticated real
pilot, and later evaluation requires complete predecessor/producer/E0 seals.

```bash
export SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny
export SIMANY_FILL_DEPENDENCY_OVERLAY=/group/worldcept/code/SimAny/.cache/icra2027/e3-gaussian-runtime-py310-pydantic-2.13.4
export PYTHONPATH="$PWD:$SIMANY_FILL_DEPENDENCY_OVERLAY"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python -m agents.edit.inpaint_fill \
  --public-context configs/experiments/icra2027/e2_full_fill/SCENE.yaml \
  --contract-manifest /group/worldcept/code/SimAny/outputs/icra2027/FREEZE/contract/freeze_manifest.json
```

Final source/E0/job IDs and outputs are recorded in the immutable dispatch
receipts. Held-out rendering and canonical PSNR/SSIM/LPIPS evaluation use a later
consumer freeze. Background availability, accepted-object coverage, and
conditional image quality remain separate quantities.
