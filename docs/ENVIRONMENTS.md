# Python environments

The construction pipeline needs three interpreters. This is not accidental
complexity — the torch builds are mutually incompatible:

| name | path (override) | torch | used by |
|---|---|---|---|
| main | `.venv` (`SIMANY_PY`) | 2.4.1+cu124 | everything except the two rows below |
| sam3 | `.envs/sam3` (`SIMANY_SAM3_PY`) | >= 2.5 | SAM3 segmentation, Qwen-Image-Edit |
| gsplat | `.envs/mini-viewer` (`SIMANY_GSPLAT_PY`) | 2.x / py3.10 | every gsplat CUDA render and 3DGS training |

Why they cannot be merged:

- **gsplat** ships a `cp310` prebuilt CUDA wheel matching torch 2.4.1+cu124.
  Building it for another interpreter needs a JIT compile with a compatible
  compiler, so every rasterization stage (`agents/render/`,
  `agents/edit/inpaint_fill`, `agents/eval/factory_eval_render`,
  `agents/discover/derive_mesh_from_splat render`, `agents/recon/gsplat_train`)
  runs under the py3.10 env. `run/env.sh` probes for `g++-12`/`g++-13` and pins
  `TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"` so a JIT cache built once is reused.
- **Qwen-Image-Edit-2509** calls `enable_gqa`, which needs torch >= 2.5, and
  `diffusers` 0.39; the main env is pinned to torch 2.4.1 / diffusers 0.36
  because 0.39 breaks there. It therefore runs in the sam3 env, and on a
  48 GB GPU it additionally needs `enable_sequential_cpu_offload`
  (`SIMANY_QWEN=force`).
- **SAM3** lives in the sam3 env for the same torch reason. Note that env has
  numpy 2.x, where `ndarray.ptp` was removed — use `np.ptp(...)` in anything
  that runs there.

`run/env.sh` resolves all three and exposes `run` / `run_sam3` / `run_gs` /
`run_qwen` so no launcher hardcodes an interpreter path. The `phiroom` CLI
resolves the same variables (or `configs/runtime.local.json`); see
[MODULES.md](MODULES.md) for the alias table.

## Optional environments

| name | path (override) | used by |
|---|---|---|
| sam3d | `.envs/sam3d-objects` (`SIMANY_SAM3D_PY`) | `agents/models/s4_sam3d.py` (SAM 3D Objects: torch 2.5.1+cu121, pytorch3d, kaolin, flash-attn, built per `third_party/sam-3d-objects/doc/setup.md`); `run_sam3d` in `run/env.sh` |
| trellis2 | `.envs/trellis2` (`SIMANY_TRELLIS2_PY`) | `agents/models/s4_trellis2.py` (TRELLIS.2) |
| h5 | `.envs/h5` (`SIMANY_H5_PY`) | `agents/recon/droid_extract.py`, `agents/recon/behavior_extract.py` (h5py + numpy + cv2) |
| `.venv-mc` | manual | the MaskClustering baseline (`agents/baselines/maskclustering.py`), which pins incompatible dependencies |
| openpi | `${OPENPI_ROOT}/.venv` | the pi0.5 policy server (`run/pi05_serve.sh`), a JAX environment managed by the openpi checkout |
| control | `envs/control/.venv` | the CPU `phiroom` CLI and release checks (`uv sync --project envs/control --locked`) |
| PhiView | `tools/phiview` uv profiles | CPU, studio, inference and generation lockfiles of the bundled viewer ([PHIVIEW.md](PHIVIEW.md)) |

## Other environment facts worth keeping

- CoACD must be imported *before* torch or it segfaults.
- `dreamsim` (a ReconViaGen dependency) downloads ~1.9 GB of weights into
  `$CWD/weights`; since launchers `cd` to the repo root, that is `./weights/`
  (gitignored).
- All Hugging Face loads run with `HF_HUB_OFFLINE=1` by default; prefetch the
  weights once with `HF_HUB_OFFLINE=0` ([DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md)).
- Stage modules are run as `python -m <package>.<module>` from the repo root
  (or with `PYTHONPATH` set to it); the packages are not installed into the
  GPU environments. `bash run/smoke_imports.sh` imports every module in a
  fresh process and is the quickest check that an environment is complete.
