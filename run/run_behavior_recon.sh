#!/bin/bash
# PointWorld-BEHAVIOR -> SimAny: extract posed clip frames + GT depth from the
# restored HDF5 episodes, train 3DGS on the emulated scene, then the same
# GT-free AUTO tail as run/run_video2sim.sh (auto_segment -> factory stages
# -> export_mjcf --test -> pi05_tasks). The stage list is
# duplicated here on purpose (run_video2sim.sh was built in parallel; keeping
# the tail inline avoids a cross-script dependency). The generation-gap eval
# additionally uses <scene>/gt/ written by behavior_extract.
#
# Usage: TASK=task-0000 [EPISODES=3] [STATIC_ONLY=1]
#        [BEH_FRAME_STRIDE=3] run/run_behavior_recon.sh
# BEH_FRAME_STRIDE: auto_segment frame stride. BEHAVIOR keyframes are
# far sparser in viewpoint than ScanNet++ DSLR sweeps (1 frame per
# 11-frame clip, robot occlusion), so the default 12 leaves the >=2-
# frame confirmation merge starved - 3 uses 4x more frames.
set -e

SIMANY_ROOT=${SIMANY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TASK=${TASK:-task-0000}
EPISODES=${EPISODES:-3}
# Static clips only by default: moving objects ghost in a static 3DGS.
STATIC_ONLY=${STATIC_ONLY:-1}
SCENE_NAME=${SCENE_NAME:-behavior_${TASK//-/}}          # task-0000 -> behavior_task0000
RECON_ROOT=${RECON_ROOT:-$SIMANY_ROOT/data/recon_scenes}

# common.py composes SCENE_DIR=$SCANNETPP_ROOT/data/$SCENE, so RECON_ROOT
# holds a data/ level and doubles as the emulated dataset root. Export BEFORE
# sourcing env.sh so its derived SD/IMG point at the emulated scene too.
export SIMANY_SCANNETPP_ROOT=$RECON_ROOT
export SIMANY_SPLATS_ROOT=$RECON_ROOT/splats
export SIMANY_SCENE=$SCENE_NAME
export SIMANY_OUT=$SIMANY_ROOT/outputs/behavior_${TASK}
# GT_MESH=1 (default): raycast against the simulator's own geometry, TSDF-
# fused from the release's GT depth by behavior_extract - the "original
# mesh", like ScanNet++ runs use the dataset scan. GT_MESH=0 falls back to
# the splat-derived mesh (measures the recon-only rung).
GT_MESH=${GT_MESH:-1}
export SIMANY_AUTO=1
export SIMANY_MESH_SRC=derived   # overwritten below when GT_MESH=1
source "$SIMANY_ROOT/run/env.sh"
mkdir -p "$SIMANY_OUT" "$SIMANY_SPLATS_ROOT"

# h5py is in none of the three SimAny envs (docs/ENVIRONMENTS.md); the
# artifixer venv has h5py+cv2+numpy and this stage is CPU-only.
H5PY_BIN=${SIMANY_H5_PY:-${SIMANY_H5_PY:-python3}}

STATIC_FLAG=""
[ "$STATIC_ONLY" = "1" ] && STATIC_FLAG="--static-only"

stage_timed "behavior_extract (HDF5 -> emulated scene dir + GT)"
done_skip "$SD/dslr/colmap/images.txt" || \
  $H5PY_BIN -m agents.recon.behavior_extract --task "$TASK" \
    --episodes "$EPISODES" $STATIC_FLAG \
    --scene-name "$SCENE_NAME" --root "$RECON_ROOT"

stage_timed "GT-depth TSDF mesh (open3d, main venv)"
done_skip "$SD/gt/mesh_gt.ply" || \
  run agents.recon.behavior_extract --fuse-only \
    --task "$TASK" --scene-name "$SCENE_NAME" --root "$RECON_ROOT"

# ---- resolution-scaled pixel gates for the AUTO tail ----------------------
# factory_prepare's MIN_BBOX_PX=48 / MIN_MASK_PX=400 were tuned on 1752x1168
# ScanNet++ DSLR frames; BEHAVIOR clips are 320x180, where a 48px bbox is 27%
# of the frame height and would gate out nearly every manipulandum. Scale by
# the extracted frame height (bbox linearly, mask area quadratically), unless
# the caller already exported explicit values. At 180px this yields 7 / 10.
GATES=$($H5PY_BIN -c "
import json, sys
h = json.load(open(sys.argv[1]))['h']
ref = 1168.0  # validated ScanNet++ height the gates were tuned on
print(max(6, round(48 * h / ref)), max(9, round(400 * (h / ref) ** 2)))
" "$SD/dslr/nerfstudio/transforms_undistorted.json")
export SIMANY_MIN_BBOX_PX=${SIMANY_MIN_BBOX_PX:-${GATES% *}}
export SIMANY_MIN_MASK_PX=${SIMANY_MIN_MASK_PX:-${GATES#* }}
echo "[behavior_recon] pixel gates: MIN_BBOX_PX=$SIMANY_MIN_BBOX_PX" \
     "MIN_MASK_PX=$SIMANY_MIN_MASK_PX"

stage_timed "gsplat_train (3DGS on posed BEHAVIOR frames)"
done_skip "$SIMANY_SPLATS_ROOT/$SCENE_NAME.ply" || \
  run_gs agents.recon.gsplat_train --scene-dir "$SD" \
    --init-ply "$SD/init_points.ply" \
    --out "$SIMANY_SPLATS_ROOT/$SCENE_NAME.ply"

if [ "$GT_MESH" = "1" ] && [ -f "$SD/gt/mesh_gt.ply" ]; then
  export SIMANY_MESH_SRC=$SD/gt/mesh_gt.ply
  MESH_FOR_SEG=$SIMANY_MESH_SRC
  echo "[behavior_recon] using ORIGINAL (GT-depth TSDF) mesh: $SIMANY_MESH_SRC"
else
  MESH_FOR_SEG=$SIMANY_OUT/derived_mesh.ply
  stage_timed "derive mesh: render"
  done_skip "$SIMANY_OUT/derived_mesh.ply" || \
    run_gs agents.discover.derive_mesh_from_splat render
  stage_timed "derive mesh: fuse"
  done_skip "$SIMANY_OUT/derived_mesh.ply" || {
    run agents.discover.derive_mesh_from_splat fuse && sync; }
fi

# ---- AUTO tail (mirrors run_video2sim.sh + run_auto.sh) --------------------
stage_timed "auto_segment (SAM3, against derived mesh)"
done_skip "$SIMANY_OUT/auto_instances.npz" || \
  run_sam3 agents.discover.auto_segment --scene-dir "$SD" \
    --out-dir "$SIMANY_OUT" --mesh-path "$MESH_FOR_SEG" \
    --frame-stride "${BEH_FRAME_STRIDE:-3}"

stage_timed "factory_prepare"
done_skip "$SIMANY_OUT/objects/objects.json" || run agents.discover.factory_prepare

stage_timed "factory_refine_masks"
done_skip "$SIMANY_OUT/objects/.masks_refined" || {
  run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" \
    --out-dir "$SIMANY_OUT" &&
  touch "$SIMANY_OUT/objects/.masks_refined"; }

stage_timed "s4 TRELLIS image-to-3D"
run agents.models.s4_trellis

stage_timed "factory_align"
run agents.assets.factory_align

stage_timed "s6 CoACD + physics + URDF"
run agents.assets.s6_physics

stage_timed "yield/tier report (GT-free in AUTO mode)"
run agents.eval.factory_report

stage_timed "export MJCF + Isaac manifest + MuJoCo settle test"
run robo.sim.export_mjcf --test

stage_timed "pi05_tasks (pick-and-place task suite)"
run robo.tasks.pi05_tasks --out-dir "$SIMANY_OUT"

stage_timed "DONE -> $SIMANY_OUT"
