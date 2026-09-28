# Python environments

The pipeline needs three interpreters. This is not accidental complexity we
failed to remove — the torch builds are mutually incompatible:

| name | path (override) | torch | used by |
|---|---|---|---|
| main | `.venv` (`SIMANY_PY`) | 2.4.1+cu124 | everything except the two rows below |
| sam3 | shared env (`SIMANY_SAM3_PY`) | 2.10 | SAM3 segmentation, Qwen-Image-Edit |
| gsplat | mini-viewer env (`SIMANY_GSPLAT_PY`) | 2.x / py3.10 | every gsplat CUDA render |

Why they cannot be merged:

- **gsplat** ships only a `cp310` prebuilt CUDA wheel. Building it for the
  main env's py3.11 needs a JIT compile the cluster's compilers cannot do, so
  every rasterization stage (`render/`, `edit/inpaint_fill`,
  `eval/factory_eval_render`, `discover/derive_mesh_from_splat render`) runs
  under the py3.10 env.
- **Qwen-Image-Edit-2509** calls `enable_gqa`, which needs torch >= 2.5, and
  `diffusers` 0.39; the main env is pinned to torch 2.4.1 / diffusers 0.36
  because 0.39 breaks there. It therefore runs in the sam3 env, and on a
  48 GB A6000 it additionally needs `enable_sequential_cpu_offload`.
- **SAM3** lives in the sam3 env for the same torch reason. Note that env has
  numpy 2.x, where `ndarray.ptp` was removed — use `np.ptp(...)` in anything
  that runs there.

`run/env.sh` resolves all three and exposes `run` / `run_sam3` / `run_gs` /
`run_qwen` so no launcher hardcodes an interpreter path.

## Other environment facts worth keeping

- `.venv-mc` is a separate env for the MaskClustering baseline
  (`agents/baselines/maskclustering.py`), which pins incompatible deps.
- Batch/H200 nodes (`msp3-*`) do **not** mount the shared group storage or the shared
  home, so they cannot run this pipeline at all; every measured number is
  A6000.
- `debug` QoS allows at most 4 concurrent GPU jobs per user.
- CoACD must be imported *before* torch or it segfaults.
- `dreamsim` downloads ~1.9 GB of weights into `$CWD/weights`; since launchers
  now `cd` to the repo root, that is `./weights/` (gitignored).
