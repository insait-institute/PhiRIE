#!/usr/bin/env bash
#SBATCH --job-name=phiview-behavior-rtx-probe
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913/behavior-native-preflight/rtx-%j.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
campaign_root="$PWD/outputs/paper-capture-20260913"
/group/worldcept/code/SimAny/.envs/studio/bin/python -m physicalview.paper_require_complete --root "$campaign_root" --dataset libero
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/group/worldcept/code/SimAny/third_party/behavior1k_datasets
export OMNIGIBSON_APPDATA_PATH="$campaign_root/behavior-native-preflight/appdata-$SLURM_JOB_ID"
export TMPDIR="$campaign_root/behavior-native-preflight/tmp-$SLURM_JOB_ID"
export PYTHONPATH=/group/worldcept/code/SimAny/third_party/BEHAVIOR-1K/OmniGibson:/group/worldcept/code/SimAny/third_party/BEHAVIOR-1K/bddl3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR" "$OMNIGIBSON_APPDATA_PATH"
# This existing one-object RGB probe establishes renderer readiness only. It is
# outside the official feature folders and is never counted as a paper scene.
exec timeout 18m /group/streetsplat/.conda/envs/behavior1k/bin/python \
  /group/worldcept/code/SimAny/robo/sim/omnigibson_bridge/render_smoke_test.py \
  --out "$campaign_root/behavior-native-preflight/rtx-$SLURM_JOB_ID.png"
