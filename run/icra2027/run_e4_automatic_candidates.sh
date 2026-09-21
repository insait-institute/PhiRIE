#!/usr/bin/env bash
set -euo pipefail
: "${E4_CODE:?E4_CODE must name the frozen source checkout; Slurm copies launch scripts}"
[[ "$E4_CODE" = /* ]] || { echo "E4_CODE must be absolute" >&2; exit 2; }
code_root="$(cd "$E4_CODE" && pwd -P)"
[[ -f "$code_root/run/icra2027/e4_automatic_candidates.py" ]] || { echo "E4_CODE lacks the canonical candidate launcher" >&2; exit 2; }
cd "$code_root"
export PYTHONPATH="$code_root" SIMANY_EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export SIMANY_AUTO=1 SIMANY_NO_GT=1 SIMANY_MESH_SRC=derived SIMANY_SCENE=09c1414f1b
exec /group/worldcept/PhiRIE/code/SimAny/.venv/bin/python -m run.icra2027.e4_automatic_candidates \
 --config "$code_root/configs/experiments/icra2027/e4_automatic_candidates.yaml" --phase "${1:-prepare}"
