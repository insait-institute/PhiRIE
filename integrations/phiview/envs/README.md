# Runtime environments

Use `uv sync --locked` for CPU development, or
`bash tools/env/sync.sh studio|inference|generation` from a source checkout.
Each GPU profile is an independent uv project with its own lockfile and `.venv`.
Profiles are Linux x86_64 / Python 3.11; weights and datasets are separate downloads.

| Profile | Purpose | PyTorch / CUDA |
|---|---|---|
| root | CLI, packaging, CPU tests | 2.9.1 CPU in the dev group |
| studio | full Gaussian rendering, MuJoCo, viewer, captures | 2.9.1 / 12.8 |
| inference | SAM3 and Qwen Image Edit 2511 | 2.9.1 / 12.8 |
| generation | TRELLIS, registration, physical annotation | 2.4.1 / 12.4 |

`uv sync` resolves Python packages. GPU execution additionally needs the NVIDIA
driver and matching CUDA toolkit (`nvcc` on PATH or `CUDA_HOME`). gsplat builds its
CUDA extension on first use. For a chosen GPU, set `TORCH_CUDA_ARCH_LIST` accordingly;
do not reuse a JIT cache built with incompatible toolchain settings.

TRELLIS also needs its pinned source and native extensions. Follow the pinned
repository's `setup.sh` for `nvdiffrast`, `diff_gaussian_rasterization` and its other
native extensions, installing into `envs/generation/.venv`; their compilation is a
separate GPU gate from resolving this lockfile. `physicalview doctor` reports source
and interpreter prerequisites; it does not assert model quality or kernel compatibility.

SAM3D, the OpenPI policy server, and BEHAVIOR/Isaac Sim retain their upstream dedicated
environments and asset/license requirements. Configure their interpreter paths in
`configs/local.yaml`. The OpenPI *client* is pinned in studio; installing that client
does not install or launch a policy server. Existing `run/setup_*.sh` scripts are the
historical cluster recipes and remain available for reproducing older results.

CPU tests do not validate any GPU model. See `docs/RELEASE.md` for the checks actually
run for this release. uv source and index behavior follows the
[official uv documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/).
