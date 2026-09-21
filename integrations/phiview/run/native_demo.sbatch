#!/usr/bin/env bash
#SBATCH --job-name=physicalview-native
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/native-demo-20260908/job-%j.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
export PYTHONPATH=/group/worldcept/code/SimAny-wt/20260907T212314Z-simsub-f1fix:/group/worldcept/code/PhysicalView
source /group/worldcept/code/SimAny-wt/20260907T212314Z-simsub-f1fix/run/roundtrip/native_env.sh
DEMO_ROOT=/group/worldcept/code/PhysicalView/outputs/native-demo-20260908
"$SR_NATIVE_PY" run/native_demo_worker.py "$DEMO_ROOT" > "$DEMO_ROOT/worker.log" 2>&1 &
WORKER=$!
trap 'kill "$WORKER" 2>/dev/null || true' EXIT
.envs/studio/bin/python -m physicalview.native_demo --root "$DEMO_ROOT" --port 8092
