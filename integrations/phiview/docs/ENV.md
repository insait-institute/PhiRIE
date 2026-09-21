# SimAny Studio environment (`.envs/studio`)

Python env + GPU self-test tooling for the interactive Studio viewer
(viser + gsplat + MuJoCo + openpi policy client). Everything in this
directory is driven from `run/env.sh`; nothing here touches the other
envs (`.venv`, `.envs/sam3d-objects`, mini-viewer, sam3).

| | |
|---|---|
| env path | `/group/worldcept/code/SimAny/.envs/studio` (main checkout, gitignored; every worktree's `.envs` symlinks there) |
| interpreter | uv-managed CPython 3.11.15, `$STUDIO_PY` after `source run/env.sh` |
| torch | 2.9.1+cu128 / torchvision 0.24.1+cu128 (`download.pytorch.org/whl/cu128`) |
| gsplat | 1.5.3 from the PyPI sdist, CUDA backend JIT-built for `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"` into `.envs/studio/torch_extensions/gsplat_cuda/gsplat_cuda.so` (torch omits its usual `py311_cu128/` level when `TORCH_EXTENSIONS_DIR` is set) |
| mujoco / viser | 3.12.0 / 1.1.0 |
| numpy | 1.26.4 (main-pipeline pin; `openpi-client` requires `<2`; agents/* still use `ndarray.ptp`) |
| open3d / trimesh / plyfile / scipy | 0.19.0 / 4.12.2 / 1.1.3 / 1.17.1 (same as `.venv`) |
| utils3d | `EasternJournalist/utils3d` @ tag 1.7 (`a29e9c13`); PyPI's `utils3d` is an unrelated package |
| openpi-client | `/group/worldcept/code/openpi/packages/openpi-client` (local path install) |
| full pin list | `run/requirements-studio.txt` (uv-compiled, every transitive dep pinned) |

## Why a fourth env

The cluster's `rtx6000` gres is the **NVIDIA RTX PRO 6000 Blackwell Server
Edition** (compute capability 12.0, 98 GB). The existing torch 2.4.1+cu124
envs and the prebuilt `gsplat 1.5.3+pt24cu124` wheel (cubins for sm_70..sm_90
only) cannot run there. CUDA 12.8 is the first toolkit with sm_120 support, so
the Studio env is built on the cu128 wheels and compiles gsplat itself for all
four GPU generations we have. One `.so` on cephfs serves every node.

## GPU support matrix (measured)

Measured 2026-09-04 with `run/selftest_all_gpus.sh` (reports in
`outputs/studio/selftest/<gpu>.json`); driver 595.x everywhere, torch
2.9.1+cu128, one shared `gsplat_cuda.so` with **sm_80 + sm_86 + sm_90 + sm_120**
cubins (20 kernels each, checked with `cuobjdump --list-elf`). Render = gsplat
640x360 rasterization of the 1.5 M-gaussian / SH-3 scene `c50d2d1d42` (warm,
mean RGB 0.664 on every card, i.e. bit-for-bit the same image); MuJoCo =
3.12.0 EGL offscreen 320x240 render, single frame.

| gres (partition) | node | GPU | CC | VRAM | self-test | gsplat render | MuJoCo EGL | total wall |
|---|---|---|---|---|---|---|---|---|
| `a6000` (debug) | hala | NVIDIA RTX A6000 | 8.6 | 48 GB | PASS | 2.12 ms | 10 ms | 5 s |
| `a100-80g` (batch) | gcp-eu1-a100-80g-qrfh | NVIDIA A100-SXM4-80GB | 8.0 | 80 GB | PASS | 3.26 ms | 239 ms | 321 s (*) |
| `h200` (batch, `--exclude=msp3-[0-7]`) | sof1-h200-5 | NVIDIA H200 | 9.0 | 141 GB | PASS | 1.37 ms | 77 ms | 8 s |
| `rtx6000` (batch) | gcp-eu1-rtx6000-vz3w | NVIDIA RTX PRO 6000 Blackwell Server Edition | 12.0 | 96 GB | PASS | 1.20 ms | 159 ms | 271 s (*) |

All checks passed on all four: torch CUDA matmul == CPU reference, gsplat
import without any rebuild, non-black splat render, MuJoCo EGL render,
`viser` / `openpi_client` imports, and every repo module below imports.

(*) The GCP nodes spend their time on the **first cold read of the env from
cephfs**: `import torch` took 166-185 s and the gsplat import 60-64 s there
(vs 2-4 s and 0.6 s on hala/H200), and the first MuJoCo EGL renderer init
0.8-1.1 s vs 0.13-0.28 s. That is I/O, not compute -- subsequent runs on the
same node are as fast as hala. The MuJoCo per-frame numbers above are likewise
single cold frames (the a6000 10 ms figure is representative of steady state).

Repo modules verified to import under this env (`PYTHONPATH=<repo root>`),
no failures: `agents.core.common`, `robo.envs.pi05_env`,
`robo.policy.clients.pi05_client`, `robo.rendering.pi05_render`,
`interface.mujoco_live_viewer`, `robo.rigs.pi05_rig` (menagerie found).

## Build steps (what `setup_env.sh` does)

```bash
# 1. wheels only -- any node, ~2 min warm / ~10 min cold (torch+nvidia libs ~3 GB)
run/setup_env.sh
# 2. gsplat CUDA backend -- needs a GPU node (compile ~8 cores)
srun -p debug --gres=gpu:a6000:1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00 \
     bash -lc 'run/setup_env.sh'
# 3. verify everywhere
run/selftest_all_gpus.sh          # -> outputs/studio/selftest/{a6000,a100,h200,rtx6000}.json
```

Step by step, all idempotent:

1. `uv venv --python 3.11 --seed .envs/studio` (uv at `~/.local/bin/uv`, cache
   forced to `.envs/.uv-cache` because `~/.cache` is a node-local `/scratch`
   symlink).
2. `uv pip install --index-strategy unsafe-best-match -r requirements-studio.txt`
   (the file carries the cu128 `--extra-index-url`; `PIP_CONFIG_FILE=/dev/null`
   because the site `/etc/pip.conf` lists a dead NGC index).
3. `BUILD_NO_CUDA=1 uv pip install --no-build-isolation gsplat==1.5.3` -- the
   sdist imports torch in `setup.py`, and `BUILD_NO_CUDA` stops it from
   compiling `csrc` at pip time with whatever arch list / compiler happens to
   be around.
4. `uv pip install /group/worldcept/code/openpi/packages/openpi-client`.
5. Import smoke test (also checks `utils3d.numpy` uses wxyz quaternions).
6. On a GPU node (`nvidia-smi -L` **and** `torch.cuda.is_available()`; or
   `STUDIO_FORCE_JIT=1`): `python -c "from gsplat.cuda._backend import _C"`
   with `CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1`,
   `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"`,
   `TORCH_EXTENSIONS_DIR=.envs/studio/torch_extensions`, `MAX_JOBS=$SLURM_CPUS_PER_TASK`,
   then `cuobjdump --list-elf gsplat_cuda.so` and a check that sm_80/86/90/120
   cubins are all present. `STUDIO_SKIP_JIT=1` skips this step.

`gsplat/cuda/_backend.py` passes no `-gencode` flags of its own: it calls
`torch.utils.cpp_extension._jit_compile`, which turns `TORCH_CUDA_ARCH_LIST`
into `-gencode arch=compute_XY,code=sm_XY` per entry (torch 2.9 knows `12.0`).
No `-allow-unsupported-compiler` is needed: nvcc 12.8 accepts the system gcc
14.2.

## Runtime contract (`run/env.sh`)

`source run/env.sh` exports `STUDIO_PY`, `STUDIO_ENV_DIR`,
`CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1` (+`PATH`), `TORCH_CUDA_ARCH_LIST`,
`TORCH_EXTENSIONS_DIR`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`MUJOCO_EGL_DEVICE_ID` (first GPU SLURM gave us), `HF_HUB_OFFLINE=1`,
`PIP_CONFIG_FILE=/dev/null`, `PYTHONPATH=<repo root>` (the repo's packages are
not pip-installed) and unsets the leaked `TCNN_CUDA_ARCHITECTURES`,
`NVCC_APPEND_FLAGS`, `CC/CXX/CUDAHOSTCXX`.

**Keep `TORCH_CUDA_ARCH_LIST` and `CUDA_HOME` exactly as env.sh sets them at
runtime too.** torch hashes the build flags into the extension version; a
different arch list (e.g. the `9.0+PTX` the submission shell leaks) makes
`import gsplat` silently rebuild for ~5 minutes into a different-flavoured .so.

## Self-tests

* `selftest_gpu.py --out X.json` (run under `$STUDIO_PY` on a GPU node): GPU
  name/CC/driver, torch CUDA matmul vs CPU, gsplat import + `.so` cubin list +
  a 640x360 rasterization of `/data/ScanNetppv2_gsplat/splats/c50d2d1d42.ply`
  (camera at the splat centroid, identity rotation; must not be black), MuJoCo
  3.12 EGL offscreen render of a tiny model, `viser` / `openpi_client` imports,
  and the repo modules `agents.core.common`, `robo.envs.pi05_env`,
  `robo.policy.clients.pi05_client`, `robo.rendering.pi05_render`,
  `interface.mujoco_live_viewer`, `robo.rigs.pi05_rig`. Non-zero exit on any
  critical failure.
* `selftest_all_gpus.sh [gpu...]`: runs the above via `srun` on
  `debug/a6000`, `batch/a100-80g`, `batch/h200 --exclude=msp3-[0-7]`,
  `batch/rtx6000` in parallel (`--time=00:20:00 --mem=32G`, 25 min wall cap
  incl. queue -> `TIMEOUT`), writes `outputs/studio/selftest/<gpu>.{json,log}`
  and prints the matrix.

## Known caveats

* **Blackwell needs this env.** `.venv` / mini-viewer (torch 2.4.1+cu124, gsplat
  pt24cu124 wheel) fail on `rtx6000` with "no kernel image is available"; only
  the studio env has sm_120 cubins. torch 2.9.1+cu128's own arch list is
  `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`.
* **One shared JIT cache.** The `.so` lives on cephfs under
  `.envs/studio/torch_extensions/gsplat_cuda/`; never let `TORCH_EXTENSIONS_DIR`
  fall back to `~/.cache/torch_extensions` (node-local `/scratch` on hala,
  evicted, absent elsewhere). Two processes must not JIT-build from scratch at
  the same time: `gsplat/cuda/_backend.py` deletes the lock file and `rmtree`s
  the build dir when no `.so` exists yet, so a concurrent first build can
  clobber the other one. Build once (setup_env.sh on one GPU node), then fan
  out. A stale `lock` file after a killed build makes later imports hang;
  setup_env.sh deletes it.
* **Leaked shell variables.** Interactive shells here carry
  `CUDA_HOME=/opt/modules/nvidia-cuda-12.4.1`, `TORCH_CUDA_ARCH_LIST=9.0+PTX`,
  `TCNN_CUDA_ARCHITECTURES=90`; `env.sh` overrides them unconditionally.
  nvcc 12.4 would also reject the system gcc 14.2 (12.8 accepts it).
* **SLURM specifics.** The A100 gres is `a100-80g` (not `a100`); `--gres=gpu:1`
  without a type is rejected; always `--exclude=msp3-[0-7]` for h200 (those
  nodes do not mount `/group/worldcept`); `srun` from a cwd under `/tmp` fails
  to chdir on the remote node, so launch from the repo root; the login node
  intermittently lists hala's A6000s in `nvidia-smi -L` but `nvidia-smi` also
  intermittently exits 6 ("No devices were found") -- env.sh tolerates both.
  `debug`/hala and `batch` queues can hold jobs for 10-30 min; `selftest_all_gpus.sh`
  reports `TIMEOUT` after 25 min instead of hanging.
* **EGL in jobs.** MuJoCo's EGL backend enumerates all physical GPUs;
  `MUJOCO_EGL_DEVICE_ID` is pinned to the first SLURM-visible GPU by env.sh.
* **numpy stays 1.26.4.** `openpi-client` pins `numpy<2`; bump only together
  with an openpi update and after grepping `agents/` for `.ptp(`.
* **`utils3d` on PyPI is the wrong package** (pins `open3d<0.14`); the
  requirements file installs `EasternJournalist/utils3d` from git (needs
  outbound git access on the installing node). Only `utils3d.numpy.
  quaternion_to_matrix` / `matrix_to_quaternion` (wxyz) are used by the repo.
* **`third_party/mujoco_menagerie` is gitignored.** `robo.rigs.pi05_rig`
  resolves it relative to the repo, so a fresh worktree needs
  `ln -s /group/worldcept/code/SimAny/third_party/mujoco_menagerie third_party/`
  (the self-test reports `menagerie_present`).
* The repo's packages (`agents`, `robo`, `interface`, ...) are not
  pip-installed; everything runs with `PYTHONPATH=<repo root>` from env.sh.
