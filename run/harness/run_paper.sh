#!/usr/bin/env bash
set -euo pipefail

MODE=run
if [[ ${1:-} == --smoke ]]; then
  MODE=smoke
  shift
fi
CONFIG=${1:-configs/experiments/harness_paper.yaml}
OUT=${2:-}
PYTHON=${SIMANY_PY:-python}

if [[ "$MODE" == smoke && -z "$OUT" ]]; then
  echo 'Usage: run_paper.sh --smoke CONFIG NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
MODULE=robo.eval.harness_runner
if [[ "$MODE" == smoke ]]; then
  MODULE=robo.eval.harness_smoke
fi
args=("$PYTHON" -m "$MODULE" --config "$CONFIG")
if [[ -n "$OUT" ]]; then
  args+=(--out "$OUT")
fi

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
