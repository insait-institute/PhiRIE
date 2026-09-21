#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --job-name=simany_rowC_pilot
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/ablation_rowC_pilot.log
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
source "$SIMANY_ROOT/run/env.sh"
S=c50d2d1d42
OUT=$SIMANY_ROOT/outputs/${S}_rowC2
# SIMANY_AUTO=1: SAM3 discovery (not the GT whitelist); mirror of
# ablation_rowC_scene.sbatch - see the fix note there
export SIMANY_SCENE=$S SIMANY_OUT=$OUT SIMANY_MESH_SRC=derived SIMANY_AUTO=1
mkdir -p "$OUT"

echo "=== [1/4] render splat depth for TSDF fusion ==="
run_gs agents.discover.derive_mesh_from_splat render

echo "=== [2/4] TSDF-fuse -> derived_mesh.ply ==="
run agents.discover.derive_mesh_from_splat fuse

echo "=== [3/4] auto_segment discovery (SAM3) against the derived mesh ==="
run_sam3 agents.discover.auto_segment --scene-dir /data/ScanNetpp/data/$S --out-dir $OUT \
  --mesh-path $OUT/derived_mesh.ply

echo "SHARD_DONE_ROWC_PILOT"
