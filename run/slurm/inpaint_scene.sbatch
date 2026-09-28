#!/bin/bash
# Object removal + background completion for ONE factory scene -> clean_background.ply,
# which is what photoreal CompositeObs (and therefore every pi0.5 composite demo)
# requires. Only 4 of 50 factory scenes had it before 2026-08-04, and that pool —
# not scene availability — is what constrained demo/scene choice.
#
#   sbatch --export=ALL,SIMANY_SCENE=825d228aec run/slurm/inpaint_scene.sbatch
#
# Partition note: the Qwen erase stage runs ~5 h, over the 4 h debug wall, so this
# uses `batch` (infinite). Pinned to hala because it is the only a6000 (sm_86) and
# the prebuilt gsplat JIT cache holds sm_86/sm_90 only — an A100 (sm_80) would
# silently rebuild it. msp3-* are excluded everywhere in this repo: they are
# region-us and do not mount ${PHIRIE_WORKSPACE:-$HOME}.
#SBATCH -p batch
#SBATCH -w hala
#SBATCH --gres=gpu:a6000:1
#SBATCH -t 12:00:00
#SBATCH --mem=200G
#SBATCH -J inpaint
#SBATCH -o outputs/inpaint_%x_%j.log
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
: "${SIMANY_SCENE:?set SIMANY_SCENE=<scene id>}"
export SIMANY_SCENE
source "$SIMANY_ROOT/run/env.sh"

# Qwen-Image-Edit-2509 needs torch>=2.5 (enable_gqa); the main venv is 2.4.1, so
# the erase stage must run in the sam3 env. QWEN=force selects the sequential
# CPU-offload path, which is what makes the 2509 checkpoint fit on a 48 GB A6000.
export QWEN_PY="${SIMANY_SAM3_PY:-${SIMANY_SAM3_PY:-${SIMANY_ROOT:-$PWD}/.envs/sam3/bin/python}}"
export SIMANY_QWEN=force
export MUJOCO_GL=egl
GPU_IDX=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
export MUJOCO_EGL_DEVICE_ID=${GPU_IDX:-0}
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"
export CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
for g in /usr/bin/g++-12 /usr/bin/g++-13 /usr/bin/g++-14 /usr/bin/g++; do
  [ -x "$g" ] && export CXX=$g CUDAHOSTCXX=$g && break
done
echo "[inpaint] scene=$SIMANY_SCENE out=$SIMANY_OUT CXX=$CXX"

# The per-view inpainted_k.png cache makes this resumable, so a requeue after a
# timeout picks up where it stopped rather than redoing the diffusion.
exec bash "$SIMANY_ROOT/run/run_inpaint.sh"
