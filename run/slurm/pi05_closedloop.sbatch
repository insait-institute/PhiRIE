#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:a6000:1
#SBATCH -t 4:00:00
#SBATCH -J pi05_eval
#SBATCH -o outputs/pi05_eval_%j.log
# The policy server holds the 12 GB checkpoint in host RAM while jax stages it
# to the GPU; the cgroup default (2G) OOM-kills it mid-restore (job 645991).
# This used to be passed on the sbatch command line and got dropped.
#SBATCH --mem=100G
# Closed-loop pi0.5 eval: policy server + MuJoCo client on one A6000.
# Env knobs:
#   SCENES  space-separated scene ids (default: the 6-scene suite)
#   OBS     raster | composite (composite = photoreal, single scene/run)
#   EXTRA   extra pi05_eval.py args (e.g. "--episodes 3 --jitter 0.02")
set -e
# The packages (robo/, agents/, ...) are NOT pip-installed into .venv, so the
# repo root has to be on PYTHONPATH for `from robo ...`.
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
# cwd must stay one level below $ROOT: the 07-22 task suites hard-code
# "scene_xml": "../outputs/<scene>_factory/sim_export/scene.xml" (pi05_tasks.py
# wrote str(out_dir/...) with a relative out_dir), so ../outputs has to resolve
# to $SIMANY_ROOT/outputs. PYTHONPATH=$ROOT keeps the packages importable from here.
cd $SIMANY_ROOT/robo
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
SCENES=${SCENES:-"c50d2d1d42 578511c8a9 45b0dac5e3 d755b3d9d8 f3d64c30f8 3db0a1c8f3"}
OBS=${OBS:-raster}
RUN=${RUN:-pi05_$(date +%m%d_%H%M)}

# unique port per job: concurrent jobs share the node (localhost!)
PORT=$((8000 + SLURM_JOB_ID % 1000))
bash $SIMANY_ROOT/run/pi05_serve.sh --port $PORT &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
for i in $(seq 1 360); do  # up to 30 min: first checkpoint restore from
  curl -sf localhost:$PORT/healthz >/dev/null 2>&1 && break  # CephFS is slow
  sleep 5
done
curl -sf localhost:$PORT/healthz >/dev/null || { echo "server never came up"; exit 1; }
echo "[closedloop] server ready on :$PORT"

export MUJOCO_GL=egl
# EGL enumerates all physical GPUs; point MuJoCo at the one slurm gave us
GPU_IDX=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
export MUJOCO_EGL_DEVICE_ID=${GPU_IDX:-0}
echo "[closedloop] EGL device $MUJOCO_EGL_DEVICE_ID"
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
export CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1
# keep in sync with the prebuilt gsplat JIT cache or torch will rebuild it.
# Unconditional (not :-default): --export=ALL leaks the login profile's
# TORCH_CUDA_ARCH_LIST=9.0+PTX (an H200 setting) into the job, which silently
# changes the torch-extension hash and forces a rebuild on this A6000 node.
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"
if [ "$OBS" = "composite" ]; then
  for s in $SCENES; do
    SIMANY_SCENE=$s SIMANY_OUT=$SIMANY_ROOT/outputs/${s}_factory \
    $SIMANY_ROOT/.venv/bin/python -m robo.eval.pi05_eval --port $PORT \
      --tasks $SIMANY_ROOT/outputs/${s}_factory/sim_export/pi05_tasks.json \
      --out $SIMANY_ROOT/outputs/pi05_runs/${RUN}_${s}_composite \
      --obs composite --video $EXTRA
  done
else
  TASKS=""
  for s in $SCENES; do TASKS="$TASKS $SIMANY_ROOT/outputs/${s}_factory/sim_export/pi05_tasks.json"; done
  $SIMANY_ROOT/.venv/bin/python -m robo.eval.pi05_eval --port $PORT --tasks $TASKS \
    --out $SIMANY_ROOT/outputs/pi05_runs/${RUN}_raster --video $EXTRA
fi
