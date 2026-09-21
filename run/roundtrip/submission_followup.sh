#!/usr/bin/env bash
# Entry point for implemented simulation-only follow-up tools.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PY=${SIMANY_NATIVE_PY:-${SIMANY_PY:-python}}
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "Python interpreter is unavailable: $PY" >&2; exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODE=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  mechanism) exec "$PY" -m robo.roundtrip.mechanism_followup "$@" ;;
  scorer) exec "$PY" -m robo.roundtrip.scorer_sensitivity "$@" ;;
  fidelity) exec "$PY" -m robo.roundtrip.object_fidelity "$@" ;;
  test) exec "$PY" -m pytest -q tests/test_simulation_submission.py tests/test_simulation_submission_integration.py tests/test_native_scale_tables.py "$@" ;;
  help|--help|-h)
    cat <<'EOF'
Usage: bash run/roundtrip/submission_followup.sh {mechanism|scorer|fidelity|test} ...
  mechanism prepare --config resolved.yaml --out NEW_BUNDLE
  mechanism launch --bundle NEW_BUNDLE --max-jobs 1           # DRY RUN
  mechanism launch --bundle NEW_BUNDLE --max-jobs 1 --submit  # ordinary Slurm job
  mechanism collect --bundle NEW_BUNDLE --out NEW_COLLECTION
  scorer --planned ORIGINAL_PLAN --ledger ORIGINAL_LEDGER --out NEW_OUTPUT
  fidelity masks --bindings ORIGINAL_BINDINGS --out NEW_MASKS
  fidelity metrics --masks MASK_MANIFEST --renders ORIGINAL_RENDER_ROOT --out NEW_METRICS
Set SIMANY_NATIVE_PY to the existing pinned native interpreter.
Native rendering commands require an allocated compatible headless runtime.
See plan/icra2027/12_simulation_submission/README.md for inputs and validation.
EOF
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
