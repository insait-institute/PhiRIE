#!/usr/bin/env bash
set -euo pipefail
source run/roundtrip/native_env.sh
: "${SR0_POLICY_PY:=${SIMANY_ROOT:-$PWD}/worktrees/sr0-policy/.venv-policy/bin/python}"
: "${SR0_OPENPI_ROOT:=${SIMANY_ROOT:-$PWD}/worktrees/sr0-policy/third_party/robocasa-openpi}"
: "${SR0_CHECKPOINT_ROOT:=${SIMANY_ROOT:-$PWD}/worktrees/sr0-policy/checkpoints/robocasa/pi05_pretrain_human300/multitask_learning/75000}"
: "${SR0_CHECKPOINT_RECEIPT:=${SIMANY_ROOT:-$PWD}/worktrees/sr0-policy/outputs/sr0-policy-discovery/checkpoint_download_receipt.json}"
: "${SR0_OPENPI_CACHE:=${SIMANY_ROOT:-$PWD}/worktrees/sr0-policy/checkpoints/openpi-cache}"
export SR0_POLICY_PY SR0_OPENPI_ROOT SR0_CHECKPOINT_ROOT SR0_CHECKPOINT_RECEIPT SR0_OPENPI_CACHE
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$python_native" -m robo.roundtrip.local_policy_instance "$@"
