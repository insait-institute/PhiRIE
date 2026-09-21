#!/usr/bin/env bash
# CPU checks only: no model execution, installations, Slurm submission, or API calls.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ "$#" -gt 0 ]]; then
  if [[ "$#" -eq 1 && ( "$1" == '--help' || "$1" == '-h' ) ]]; then
    printf '%s\n' 'Usage: bash run/campaign/final_preflight.sh' 'Set SIMANY_CAMPAIGN_PY to an existing CPU test interpreter. This command does not run experiments.'
    exit 0
  fi
  printf '%s\n' 'Unexpected arguments. Use --help. No experiment has been launched.' >&2
  exit 2
fi
PY="${SIMANY_CAMPAIGN_PY:-${SIMANY_FINAL_PY:-python}}"
command -v "$PY" >/dev/null 2>&1 || { printf 'Missing interpreter: %s\n' "$PY" >&2; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SIMANY_TEST_REPO_ROOT="$ROOT"
export SIMANY_CAMPAIGN_PY="$PY"
for path in robo/campaign/finalize.py robo/eval/final_release.py robo/rendering/final_observation.py configs/experiments/final_submission/protocol.yaml; do
  test -f "$path" || { printf 'Missing current finalization file: %s\n' "$path" >&2; exit 1; }
done
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/simany-final-preflight.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT
# Isolate these dependency-light tests from unrelated repository conftest imports.
cp tests/test_system_campaign.py tests/test_final_experiments.py tests/test_final_entrypoints.py "$TEST_ROOT/"
"$PY" -m pytest -q "$TEST_ROOT"
bash -n run/campaign/finalize.sh run/campaign/final_preflight.sh run/finalize/run.sh
bash run/campaign/finalize.sh --help
printf '%s\n' 'CPU preflight passed. Native DEV admission and experiments are still required.'
