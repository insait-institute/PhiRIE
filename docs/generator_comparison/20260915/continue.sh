#!/usr/bin/env bash
set -euo pipefail
run_root=/group/worldcept/code/SimAny/outputs/icra2027/generator-comparison-bounded-20260914
worker_root=/group/worldcept/code/SimAny/worktrees/generator-comparison-final-20260915
mode="${1:-status}"
if (( $# )); then shift; fi
case "$mode" in status|dispatch|collect|admit) ;; *) echo "Usage: $0 {status|dispatch|collect|admit} [admission options]" >&2; exit 2;; esac
cd "$worker_root"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=/group/worldcept/code/SimAny/worktrees/generator-comparison-bounded-20260914/envs/control/.venv
exec bash run/phiroom.sh run experiments generator-comparison --config "$run_root/runtime-control.json" -- "$mode" --run "$run_root" "$@"
