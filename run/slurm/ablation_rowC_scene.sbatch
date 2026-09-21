#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=simany_rowC
# Usage: sbatch --export=ALL,SCENE=<id> --output=... ablation_rowC_scene.sbatch
# Single job, single node, start to finish - a derived_mesh.ply overwritten
# across separate job submissions hit stale cross-node NFS reads once
# (see outputs/ablation_rowC_v4.log vs _clean.log); keep it one job.
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
source "$SIMANY_ROOT/run/env.sh"
# SIMANY_AUTO=1 is what makes this row SAM3 discovery (factory_prepare would
# otherwise silently fall back to the GT whitelist); _rowC2 is a fresh dir so
# reruns never mix with the old GT-mode _rowC outputs.
export SIMANY_SCENE=$SCENE SIMANY_OUT=$SIMANY_ROOT/outputs/${SCENE}_rowC2 SIMANY_MESH_SRC=derived SIMANY_AUTO=1
mkdir -p "$SIMANY_OUT"
IMG=/data/ScanNetpp/data/$SCENE/dslr/resized_undistorted_images


OLD_MESH=$SIMANY_ROOT/outputs/${SCENE}_rowC/derived_mesh.ply
if [ -f "$OLD_MESH" ]; then
  stage "derive mesh: reusing derived mesh $OLD_MESH"
  ln -sfn "$OLD_MESH" "$SIMANY_OUT/derived_mesh.ply"
else
  stage "derive mesh: render"
  run_gs agents.discover.derive_mesh_from_splat render
  stage "derive mesh: fuse"
  run agents.discover.derive_mesh_from_splat fuse
  sync
fi

stage "auto_segment (SAM3, against derived mesh)"
run_sam3 agents.discover.auto_segment --scene-dir /data/ScanNetpp/data/$SCENE \
  --out-dir "$SIMANY_OUT" --mesh-path "$SIMANY_OUT/derived_mesh.ply"

stage "factory_prepare"
run agents.discover.factory_prepare
stage "factory_refine_masks"
run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" --out-dir "$SIMANY_OUT"
stage "s4 TRELLIS"
run models.s4_trellis
stage "factory_align"
run agents.assets.factory_align
stage "s6 physics"
run agents.assets.s6_physics
stage "factory_report"
run agents.eval.factory_report

echo "ROWC_DONE $SCENE"
