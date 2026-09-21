# CPU control environment

`uv sync --project envs/control --locked` installs the editable PhiRoom control
package, YAML support for SimFactory, and the default test dependency group. The
test group includes CPU PyTorch and trimesh for the existing campaign preflight;
it does not replace the GPU environments. For inspection without test dependencies,
use `uv run --project envs/control --locked --no-dev phiroom modules`.
The convenience shell launcher uses the default development group.

Run `bash run/phiroom.sh modules` from the source checkout. Runtime bindings are
documented in [MODULES.md](../../docs/MODULES.md). PhiView owns its four separate
lockfiles under `integrations/phiview`.
