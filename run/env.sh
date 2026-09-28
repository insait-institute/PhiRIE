#!/bin/bash
# Shared environment for every SimAny launcher. Source it, don't execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Every path below can be overridden from the outside, so nothing in the
# pipeline hardcodes a cluster location any more.

# Repo root, derived from this file so the tree can be moved or cloned.
SIMANY_ROOT=${SIMANY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SIMANY_ROOT
ROOT=$SIMANY_ROOT

# ---- python environments -------------------------------------------------
# Three envs are unavoidable, for reasons recorded in docs/ENVIRONMENTS.md:
#   VENV   main pipeline (torch 2.4.1+cu124, open3d/trimesh/coacd/xformers)
#   SAM3PY SAM3 segmentation + Qwen-Image-Edit (torch 2.10)
#   MVPY   gsplat CUDA rendering (only prebuilt wheel is cp310)
VENV=${SIMANY_PY:-$ROOT/.venv/bin/python}
SAM3PY=${SIMANY_SAM3_PY:-$ROOT/.envs/sam3/bin/python}
MVPY=${SIMANY_GSPLAT_PY:-$ROOT/.envs/mini-viewer/bin/python}
QWEN_PY=${QWEN_PY:-$VENV}   # override with SAM3PY on nodes with torch>=2.5 + VRAM
# SAM 3D Objects (models.s4_sam3d): torch 2.5.1+cu121 + pytorch3d/kaolin,
# built per third_party/sam-3d-objects/doc/setup.md into its own conda env
SAM3D_PY=${SIMANY_SAM3D_PY:-$ROOT/.envs/sam3d-objects/bin/python}

# ---- datasets ------------------------------------------------------------
SCANNETPP_ROOT=${SIMANY_SCANNETPP_ROOT:-/data/ScanNetpp}
export SIMANY_SCANNETPP_ROOT=$SCANNETPP_ROOT
export SIMANY_SPLATS_ROOT=${SIMANY_SPLATS_ROOT:-/data/ScanNetppv2_gsplat/splats}
SPLITS=$SCANNETPP_ROOT/splits

# ---- scene selection -----------------------------------------------------
export SIMANY_SCENE=${SIMANY_SCENE:-${SIMF_SCENE:-c50d2d1d42}}
SD=$SCANNETPP_ROOT/data/$SIMANY_SCENE                       # scene dir
IMG=$SD/dslr/resized_undistorted_images                     # posed RGB frames

# ---- build/runtime flags -------------------------------------------------
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# HF_HOME / TORCH_HOME are honoured as-is. Point them at a shared cache when
# the compute node cannot see the default per-user cache directory.
# gsplat's JIT fallback prefers gcc<=13, but hala only ships gcc-14: probe for
# the newest usable one instead of hardcoding (a missing $CXX kills even
# cache-reuse loads — torch verifies the compiler before checking the cache).
for _gxx in g++-12 g++-13; do
  if [ -x "/usr/bin/$_gxx" ]; then
    export CC=${CC:-/usr/bin/${_gxx/g++/gcc}}
    export CXX=${CXX:-/usr/bin/$_gxx}
    export CUDAHOSTCXX=${CUDAHOSTCXX:-/usr/bin/$_gxx}
    break
  fi
done
export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}
# MUST match the prebuilt gsplat JIT cache (compute_86 + compute_90) or torch
# rebuilds from scratch; "8.6" alone triggered exactly that on 2026-08-01.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;9.0+PTX"}

# Stages are modules: `run <package>.<module> [args...]` from the repo root,
# e.g. `run agents.assets.s5_align` or `run_sam3 models.s1_segment`.
# `python -m` puts the repo root on sys.path, which is why we cd there.
cd "$ROOT"
run()      { local m=$1; shift; $VENV      -m "$m" "$@"; }
run_sam3() { local m=$1; shift; $SAM3PY    -m "$m" "$@"; }
run_gs()   { local m=$1; shift; $MVPY      -m "$m" "$@"; }
run_qwen() { local m=$1; shift; $QWEN_PY   -m "$m" "$@"; }
# PYTHONNOUSERSITE: ~/.local/lib/python3.11 leaks into the conda env's 3.11
# and shadows its packages (observed with hydra-core during the env build)
run_sam3d() { local m=$1; shift; PYTHONNOUSERSITE=1 $SAM3D_PY -m "$m" "$@"; }

# ---- logging helpers -----------------------------------------------------
stage() { echo; echo "=== [$(date +%H:%M:%S)] $1 ==="; }

# stage_timed additionally appends "<name> <seconds>" to $SIMANY_OUT/timings.txt,
# which is what the efficiency table is computed from.
_CUR=""; _T0=$(date +%s)
stage_timed() {
  local now=$(date +%s)
  if [ -n "$_CUR" ]; then
    mkdir -p "$SIMANY_OUT"; echo "$_CUR $((now-_T0))" >> "$SIMANY_OUT/timings.txt"
  fi
  _CUR="$1"; _T0=$now
  echo; echo "=== [$(date +%H:%M:%S)] $1 ==="
}

# RESUME=1 makes a stage skip itself when its output already exists.
done_skip() { [ "${RESUME:-0}" = "1" ] && [ -e "$1" ]; }
