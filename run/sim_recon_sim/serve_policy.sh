#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SR0_POLICY_PY:=$ROOT/.venv-policy/bin/python}"
: "${SR0_OPENPI_ROOT:=$ROOT/third_party/robocasa-openpi}"
: "${SR0_CHECKPOINT_ROOT:=$ROOT/checkpoints/robocasa/pi05_pretrain_human300/multitask_learning/75000}"
: "${SR0_CHECKPOINT_RECEIPT:=$ROOT/outputs/sr0-policy-discovery/checkpoint_download_receipt.json}"
export PYTHONPATH="$ROOT:$SR0_OPENPI_ROOT/src:$SR0_OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${SR0_OPENPI_CACHE:-$ROOT/checkpoints/openpi-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$SR0_POLICY_PY" -m robo.roundtrip.native_policy_server \
  --checkpoint "$SR0_CHECKPOINT_ROOT" --receipt "$SR0_CHECKPOINT_RECEIPT" \
  --openpi-root "$SR0_OPENPI_ROOT" "$@"
