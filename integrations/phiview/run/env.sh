#!/bin/bash
# Sourceable runtime environment for the SimAny Studio (viser + gsplat +
# MuJoCo + policy client). Source it, don't execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Every export here must stay consistent with run/setup_env.sh: the
# gsplat CUDA backend is JIT-built by torch.utils.cpp_extension and torch
# REBUILDS it if TORCH_CUDA_ARCH_LIST / CUDA_HOME differ from the build.

# Repo root resolved from this file (run/env.sh -> two levels up), so
# the same script works from any worktree/clone.
STUDIO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHYSICALVIEW_ROOT=$(cd "$STUDIO_DIR/.." && pwd)
export PHYSICALVIEW_ROOT
# SimAny checkout (agents/, robo/, models/, stage scripts); mirrors configs/default.yaml simany_root
SIMANY_ROOT=${SIMANY_ROOT:-$(sed -n "s/^simany_root:[[:space:]]*//p" "$PHYSICALVIEW_ROOT/configs/default.yaml" | head -1)}
SIMANY_ROOT=${SIMANY_ROOT:-/group/worldcept/code/SimAny-wt/studio}
export SIMANY_ROOT

# ---- interpreter -----------------------------------------------------------
# The env lives in the MAIN checkout's gitignored .envs/; worktrees symlink
# .envs there, readlink -f resolves the symlink so the path is canonical.
# readlink -f the env dir ITSELF: it may be a symlink to a shared built env, and torch keys
# its JIT cache (TORCH_EXTENSIONS_DIR below) on the textual path -> a non-canonical path
# would trigger a needless multi-minute gsplat rebuild.
_env=$(readlink -f "$PHYSICALVIEW_ROOT/.envs/studio" 2>/dev/null || echo "$PHYSICALVIEW_ROOT/.envs/studio")
export STUDIO_ENV_DIR=${STUDIO_ENV_DIR:-$_env}
export STUDIO_PY=${STUDIO_PY:-$STUDIO_ENV_DIR/bin/python}

# ---- CUDA toolchain (must match the build) --------------------------------
# The submission shell leaks CUDA_HOME/CUDA_PATH=…12.4.1 and
# TORCH_CUDA_ARCH_LIST=9.0+PTX; both would trigger a gsplat rebuild (or a
# wrong-arch one), so override unconditionally.
export CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"     # A100 / A6000 / H200 / RTX PRO 6000 Blackwell
export TORCH_EXTENSIONS_DIR=$STUDIO_ENV_DIR/torch_extensions   # shared cephfs, NOT ~/.cache (node-local /scratch)
unset TCNN_CUDA_ARCHITECTURES NVCC_APPEND_FLAGS CC CXX CUDAHOSTCXX   # gcc 14.2 is supported by nvcc 12.8 as-is

# ---- headless rendering ----------------------------------------------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# In sbatch/srun jobs EGL enumerates every physical GPU; pin it to the first
# one SLURM gave us (nvidia-smi is already namespaced by cgroups).
# (nvidia-smi exits 6 with "No devices were found" on the login node; keep that
# harmless under `set -e -o pipefail` in the scripts that source this file.)
if [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  _egl=$( { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | head -1)
  case "$_egl" in ''|*[!0-9]*) ;; *) export MUJOCO_EGL_DEVICE_ID=$_egl ;; esac
fi

# ---- misc ------------------------------------------------------------------
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PIP_CONFIG_FILE=/dev/null     # site pip.conf carries a dead NGC index
export PYTHONPATH=$PHYSICALVIEW_ROOT:$SIMANY_ROOT${PYTHONPATH:+:$PYTHONPATH}   # neither repo is pip-installed
export PYTHONUNBUFFERED=1
