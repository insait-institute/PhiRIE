#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:a6000:1
#SBATCH -t 6:00:00
#SBATCH -J paired_matrix
#SBATCH -o /group/worldcept/PhiRIE/code/SimAny/outputs/paired_matrix_%j.log
# Same 100G note as run/slurm/pi05_closedloop.sbatch: a served policy holds
# its checkpoint in host RAM while jax stages it to the GPU, and the cgroup
# default (2G) OOM-kills mid-restore. Irrelevant for POLICY=scripted_sinusoid
# but harmless to always request.
#SBATCH --mem=100G
#
# mujoco_paired protocol matrix (Task 09, robo/eval/paired_runner.py):
# runs the SAME policy/task/seed matrix against BOTH the SimAny
# reconstruction and the hand-built primitive reference variant
# (robo/eval/reference_scene.py) of each scene, per
# docs/MUJOCO_PAIRED_PROTOCOL.md.
#
# Env knobs:
#   CONFIG    matrix config yaml (default: the smoke config -- pass a real
#             production config once one is authored)
#   OUT       output dir (default: outputs/paired_runs/paired_<job id>)
#   EPISODES  --episodes override (optional; ignored on resume of an --out
#             that already has a persisted reset_states.json, by design)
#   POLICY    policy id to run. "scripted_sinusoid" (default, matches the
#             smoke config) needs no server. Any other id declared in
#             configs/experiments/frozen_fields.yaml starts a real openpi
#             server first, using the exact recipe
#             run/slurm/pi05_closedloop.sbatch already uses.
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
source "$SIMANY_ROOT/run/env.sh"
# Packages (robo/, agents/, ...) are not pip-installed into .venv, so the
# repo root must be on PYTHONPATH for `from robo ...` -- same reasoning as
# run/slurm/pi05_closedloop.sbatch.
cd $SIMANY_ROOT/robo
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}

CONFIG=${CONFIG:-$SIMANY_ROOT/configs/experiments/paired_smoke.yaml}
OUT=${OUT:-$SIMANY_ROOT/outputs/paired_runs/paired_${SLURM_JOB_ID:-manual}}
POLICY=${POLICY:-scripted_sinusoid}
EXTRA_ARGS=()
[ -n "${EPISODES:-}" ] && EXTRA_ARGS+=(--episodes "$EPISODES")

export MUJOCO_GL=egl
# EGL enumerates all physical GPUs; point MuJoCo at the one slurm gave us.
GPU_IDX=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
export MUJOCO_EGL_DEVICE_ID=${GPU_IDX:-0}
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
export CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1
# Keep in sync with the prebuilt gsplat JIT cache, per
# run/slurm/pi05_closedloop.sbatch's comment on the same line.
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"

if [ "$POLICY" != "scripted_sinusoid" ]; then
  # Unique port per job: concurrent jobs share the node (localhost!) --
  # identical recipe to run/slurm/pi05_closedloop.sbatch.
  PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
  bash $SIMANY_ROOT/run/pi05_serve.sh --port $PORT &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null" EXIT
  for i in $(seq 1 360); do  # up to 30 min: first checkpoint restore from
    curl -sf localhost:$PORT/healthz >/dev/null 2>&1 && break  # CephFS is slow
    sleep 5
  done
  curl -sf localhost:$PORT/healthz >/dev/null || { echo "server never came up"; exit 1; }
  echo "[paired_matrix] server ready on :$PORT"
  EXTRA_ARGS+=(--policy "$POLICY" --host localhost --port "$PORT")
fi

$SIMANY_ROOT/.venv/bin/python -m robo.eval.paired_runner \
  --config "$CONFIG" --out "$OUT" "${EXTRA_ARGS[@]}"
