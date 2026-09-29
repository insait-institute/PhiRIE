# CPU control environment

`uv sync --project envs/control --locked` installs the editable PhiRoom control
package, YAML support for SimFactory, and the default `dev` dependency group
(ruff, CPU PyTorch, numpy, scipy, Pillow, pydantic, trimesh). It does not
replace the GPU environments. For inspection without the development
dependencies, use `uv run --project envs/control --locked --no-dev phiroom modules`.
The convenience shell launcher `run/phiroom.sh` uses the default group.

An optional `regression` group adds the CPU simulator stack (MuJoCo, PyBullet,
Open3D, OpenCV, the openpi client) for running CPU-only backends from this
environment: `uv sync --project envs/control --locked --group regression`.

Run `bash run/phiroom.sh modules` from the source checkout. Runtime bindings are
documented in [MODULES.md](../../docs/MODULES.md). PhiView owns its separate
lockfiles under `tools/phiview`.
