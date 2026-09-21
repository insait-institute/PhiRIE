#!/bin/bash
# Build (idempotently) the SimAny Studio python env.
#
#   run/setup_env.sh            # any node: venv + wheels (+ JIT build if a GPU is visible)
#   srun -p debug --gres=gpu:a6000:1 --mem=48G --cpus-per-task=8 --time=01:00:00 \
#        bash -lc 'run/setup_env.sh'   # GPU node: also builds the gsplat CUDA backend
#
# Layout
#   $STUDIO_ENV_DIR               <PhysicalView>/.envs/studio (uv venv, cpython 3.11; may symlink a shared build)
#   $STUDIO_ENV_DIR/torch_extensions   torch cpp_extension JIT cache (gsplat_cuda.so, all 4 archs)
#
# Steps: 1) uv venv  2) pinned wheels (requirements-studio.txt)  3) gsplat sdist
# (BUILD_NO_CUDA, no build isolation)  4) openpi-client  5) import smoke test
# 6) gsplat multi-arch JIT build + cuobjdump arch listing (GPU node, or STUDIO_FORCE_JIT=1;
#    STUDIO_SKIP_JIT=1 skips it).
# Every step is a no-op when already satisfied, so re-running is cheap.
set -euo pipefail

STUDIO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# env.sh fixes CUDA_HOME / TORCH_CUDA_ARCH_LIST / TORCH_EXTENSIONS_DIR / PIP_CONFIG_FILE,
# and clears the variables the submission shell leaks (CUDA_HOME=…12.4.1, 9.0+PTX).
# shellcheck source=env.sh
source "$STUDIO_DIR/env.sh"

UV=${UV:-/home/runyi_yang/.local/bin/uv}
PYTHON_VERSION=${STUDIO_PYTHON_VERSION:-3.11}
GSPLAT_SPEC=${GSPLAT_SPEC:-gsplat==1.5.3}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-/group/worldcept/code/openpi/packages/openpi-client}
REQ="$STUDIO_DIR/requirements-studio.txt"
# uv's default cache is ~/.cache, a node-local /scratch symlink here; keep it on cephfs.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$(dirname "$STUDIO_ENV_DIR")/.uv-cache}
export UV_PYTHON_PREFERENCE=only-managed     # the 3.11 we want is uv-managed; ignore /usr/bin/python3.13
UVPIP=("$UV" pip install --python "$STUDIO_PY" --index-strategy unsafe-best-match)

log() { printf '\n[setup_env %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[ -x "$UV" ] || { echo "uv not found at $UV"; exit 2; }
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "WARN: $CUDA_HOME/bin/nvcc missing on $(hostname); the gsplat JIT build will not work here"
mkdir -p "$(dirname "$STUDIO_ENV_DIR")" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# ---- 1. venv --------------------------------------------------------------
if [ ! -x "$STUDIO_PY" ]; then
  log "creating venv $STUDIO_ENV_DIR (cpython $PYTHON_VERSION)"
  "$UV" venv --python "$PYTHON_VERSION" --seed --no-project --allow-existing "$STUDIO_ENV_DIR"   # dir may pre-exist (torch_extensions/)
else
  log "venv exists: $STUDIO_ENV_DIR ($("$STUDIO_PY" -c 'import sys;print(sys.version.split()[0])'))"
fi

# ---- 2. pinned wheels -----------------------------------------------------
log "installing pinned requirements from $REQ"
"${UVPIP[@]}" -r "$REQ"

# ---- 3. gsplat (sdist; needs torch importable -> no build isolation) ------
# BUILD_NO_CUDA=1 skips setup.py's CUDA compile, which would otherwise happen
# at pip time with whatever arch list/compiler is around. The backend is built
# instead by torch.utils.cpp_extension at first import (step 6), into
# $TORCH_EXTENSIONS_DIR, for TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST".
log "installing $GSPLAT_SPEC (BUILD_NO_CUDA=1, --no-build-isolation)"
BUILD_NO_CUDA=1 "${UVPIP[@]}" --no-build-isolation "$GSPLAT_SPEC"

# ---- 4. openpi-client (local path, deps already pinned in step 2) ---------
if [ -d "$OPENPI_CLIENT_SRC" ]; then
  log "installing openpi-client from $OPENPI_CLIENT_SRC"
  "${UVPIP[@]}" "$OPENPI_CLIENT_SRC"
else
  echo "WARN: $OPENPI_CLIENT_SRC not found; robo.policy.clients.pi05_client will not import"
fi

# ---- 5. import smoke test (no GPU needed) ---------------------------------
log "import smoke test"
"$STUDIO_PY" - <<'PY'
import importlib, sys
mods = ["torch", "torchvision", "numpy", "scipy", "trimesh", "plyfile", "open3d", "utils3d",
        "mujoco", "viser", "PIL", "imageio", "imageio_ffmpeg", "cv2", "yaml", "websockets",
        "msgpack", "msgpack_numpy", "pydantic", "pytest", "openpi_client", "ninja"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {m:16s} {type(e).__name__}: {e}"); bad.append(m)
import torch
print(f"  torch {torch.__version__} cuda={torch.version.cuda} archs={torch.cuda.get_arch_list()}")
import utils3d.numpy as u3
import numpy as np
R = u3.quaternion_to_matrix(np.array([0.7071068, 0.7071068, 0.0, 0.0]))  # 90 deg about x, wxyz
assert np.allclose(R, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-6), R
print("  utils3d.numpy quaternion convention: wxyz OK")
# gsplat's python side must import without compiling anything when no toolkit
# is used: only check the metadata here, the CUDA backend is step 6.
import importlib.metadata as md
print(f"  gsplat {md.version('gsplat')} (CUDA backend built in step 6)")
sys.exit(1 if bad else 0)
PY

# ---- 6. gsplat multi-arch JIT build ---------------------------------------
# A GPU is "visible" when nvidia-smi lists one AND torch can actually open it
# (the login node intermittently lists hala's A6000s but cannot use them).
have_gpu=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
   && "$STUDIO_PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  have_gpu=1
fi
if [ "${STUDIO_SKIP_JIT:-0}" = 1 ]; then
  log "STUDIO_SKIP_JIT=1: skipping the gsplat JIT build"
elif [ "$have_gpu" = 1 ] || [ "${STUDIO_FORCE_JIT:-0}" = 1 ]; then
  export MAX_JOBS=${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}
  log "gsplat JIT build: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS -> $TORCH_EXTENSIONS_DIR"
  [ "$have_gpu" = 1 ] && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
  # a stale lock from a killed build makes torch's cpp_extension hang silently
  find "$TORCH_EXTENSIONS_DIR" -maxdepth 3 -name lock -path '*gsplat_cuda*' -delete 2>/dev/null || true
  t0=$(date +%s)
  VERBOSE=${VERBOSE:-0} "$STUDIO_PY" -c "from gsplat.cuda._backend import _C; assert _C is not None, 'gsplat CUDA backend disabled'; print('gsplat backend:', _C.__file__)"
  log "JIT step finished in $(( $(date +%s) - t0 ))s"
  so=$(find "$TORCH_EXTENSIONS_DIR" -name 'gsplat_cuda.so' | head -1)
  if [ -n "$so" ]; then
    archs=$("$CUDA_HOME/bin/cuobjdump" --list-elf "$so" | grep -oE 'sm_[0-9]+[a-z]*' | sort -u | tr '\n' ' ')
    log "built $so"
    log "cubin archs in gsplat_cuda.so: ${archs:-<none>}"
    for want in sm_80 sm_86 sm_90 sm_120; do
      case " $archs " in *" $want "*) ;; *) echo "WARN: $want missing from $so";; esac
    done
  else
    echo "WARN: gsplat_cuda.so not found under $TORCH_EXTENSIONS_DIR"
  fi
else
  log "no GPU visible on $(hostname): skipping the gsplat JIT build (run this script under srun on a GPU node, or STUDIO_FORCE_JIT=1)"
fi

log "done. STUDIO_PY=$STUDIO_PY"
