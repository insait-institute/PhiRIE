#!/bin/bash
# Serve exactly one registry policy from an explicit clean OpenPI worktree.
# Unlike the historical pi05_serve.sh, checkpoint/config overrides are not
# accepted independently: policy id selects both from configs/policies/*.yaml.
set -euo pipefail

: "${SIMANY_POLICY_ID:?Set SIMANY_POLICY_ID to a real configs/policies id}"
: "${SIMANY_OPENPI_ROOT:?Set SIMANY_OPENPI_ROOT to an isolated clean worktree}"
: "${SIMANY_OPENPI_COMMIT:?Set SIMANY_OPENPI_COMMIT to its exact full commit}"
: "${SIMANY_POLICY_IDENTITY_FILE:?Set an absolute identity receipt path}"
: "${SIMANY_CHECKPOINT_CACHE_ROOT:?Set an absolute node-local checkpoint cache root}"

if [[ -n "${SIMANY_PI05_CKPT:-}" || -n "${SIMANY_PI05_CONFIG:-}" ]]; then
  echo "[pi05_serve_bound] independent checkpoint/config overrides are forbidden" >&2
  exit 2
fi
if [[ "${SIMANY_POLICY_IDENTITY_FILE}" != /* ]]; then
  echo "[pi05_serve_bound] SIMANY_POLICY_IDENTITY_FILE must be absolute" >&2
  exit 2
fi
if [[ "${SIMANY_CHECKPOINT_CACHE_ROOT}" != /scratch/* ]]; then
  echo "[pi05_serve_bound] cache root must be a scoped directory beneath /scratch" >&2
  exit 2
fi

SIMANY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_PYTHON=${SIMANY_OPENPI_PYTHON:-${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/.venv/bin/python}
if [[ ! -x "$OPENPI_PYTHON" ]]; then
  echo "[pi05_serve_bound] OpenPI Python is not executable: $OPENPI_PYTHON" >&2
  exit 2
fi

PORT=8000
SAMPLING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deterministic-sampling)
      SAMPLING_ARGS=(--deterministic-sampling)
      shift
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "[pi05_serve_bound] --port needs a value" >&2; exit 2; }
      PORT=$2
      shift 2
      ;;
    *)
      echo "[pi05_serve_bound] unsupported argument: $1" >&2
      exit 2
      ;;
  esac
done

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${SIMANY_ROOT:-$PWD}/checkpoints/openpi_cache}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
export PYTHONPATH="$SIMANY_OPENPI_ROOT/src:$SIMANY_OPENPI_ROOT/packages/openpi-client/src:$SIMANY_ROOT"

cd "$SIMANY_OPENPI_ROOT"
exec "$OPENPI_PYTHON" -m robo.policy.bound_server \
  --policy-id "$SIMANY_POLICY_ID" \
  --openpi-root "$SIMANY_OPENPI_ROOT" \
  --openpi-commit "$SIMANY_OPENPI_COMMIT" \
  --port "$PORT" \
  --identity-file "$SIMANY_POLICY_IDENTITY_FILE" \
  --checkpoint-cache-root "$SIMANY_CHECKPOINT_CACHE_ROOT" "${SAMPLING_ARGS[@]}"
