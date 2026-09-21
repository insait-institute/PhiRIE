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
SAM3PY=${SIMANY_SAM3_PY:-/group/streetsplat/worldcept/.envs/sam3/bin/python}
MVPY=${SIMANY_GSPLAT_PY:-/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python}
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
# ~/.cache is a symlink to hala's NODE-LOCAL /scratch, so an HF/torch-hub
# checkpoint cached there is invisible from H200/GCP nodes -- with
# HF_HUB_OFFLINE=1 that's a hard FileNotFoundError, not a slow online
# fallback. On hala, leave HF_HOME/TORCH_HOME unset so the fast local
# scratch cache is still used; everywhere else, fall back to the
# /group/worldcept-hosted shared copies (populated on demand as new
# models are needed off-hala -- see docs/DATA_AND_WEIGHTS.md).
#
# CAVEAT (found 2026-08-30): hala's /scratch also evicts entries over
# time independent of node choice -- TRELLIS-image-large and
# DA3METRIC-LARGE both vanished from it between two runs hours apart.
# This fallback (triggered only when the whole hub/ dir is empty) does
# NOT catch a specific model going missing from an otherwise-populated
# cache. Any launcher whose exact model set is known and fully mirrored
# in the shared cache should export HF_HOME/TORCH_HOME explicitly before
# sourcing this file (see run/slurm/oracle_matrix.sbatch,
# run/slurm/simfoundry_baseline.sbatch) rather than relying on this
# heuristic alone.
if [ -z "$(ls -A "$HOME/.cache/huggingface/hub" 2>/dev/null)" ]; then
  export HF_HOME=${HF_HOME:-/group/worldcept/hf_cache}
fi
if [ -z "$(ls -A "$HOME/.cache/torch/hub" 2>/dev/null)" ]; then
  export TORCH_HOME=${TORCH_HOME:-/group/worldcept/torch_hub_cache}
fi
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
