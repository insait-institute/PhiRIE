#!/usr/bin/env bash
set -euo pipefail
: "${E4_CODE:?E4_CODE must name the frozen source checkout; Slurm copies launch scripts}"
[[ "$E4_CODE" = /* ]] || { echo "E4_CODE must be absolute" >&2; exit 2; }
code_root="$(cd "$E4_CODE" && pwd -P)"
[[ -f "$code_root/run/icra2027/e4_automatic_materialization.py" ]] || { echo "E4_CODE lacks the canonical launcher" >&2; exit 2; }
cd "$code_root"
python=${SIMANY_PY:-$code_root/.venv/bin/python}
[[ -x "$python" ]] || python="$code_root/envs/control/.venv/bin/python"
[[ -x "$python" ]] || python=$(command -v python3)
export PYTHONPATH="$code_root"
export SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT:-$code_root}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec "$python" -m run.icra2027.e4_automatic_materialization \
  --config "$code_root/configs/experiments/icra2027/e4_automatic_materialization.yaml" \
  --freeze-root "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/20260905-4d0787c-v4" \
  --phase "${1:-materialize}"
