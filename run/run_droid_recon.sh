#!/bin/bash
# DROID raw episode -> SimAny sim-ready twin: wrist-camera MP4 frames ->
# COLMAP poses -> Umeyama-align to the FK trajectory (METRIC, ROBOT BASE
# frame - no metricize, base z=0 is the mount plane / table) -> 3DGS splat ->
# the same GT-free AUTO tail as run/run_video2sim.sh. The stage list is
# duplicated inline on purpose (same reasoning as run_behavior_recon.sh:
# keeping the tail inline avoids a cross-script dependency).
#
# Dynamic-scene caveat (v1): the arm and the manipulated object move through
# the wrist frames; droid_extract prefers steps with a steady gripper and low
# joint speeds, but residual ghosting of the manipulandum is accepted.
#
# Usage: EPISODE=IPRL/success/2023-08-24/Thu_Aug_24_21:29:53_2023 \
#            run/run_droid_recon.sh
# Knobs: SCENE_NAME (default droid_<lab>_<timestamp>), CAMERA=wrist,
#        STRIDE=1, MAX_FRAMES=240, GS_ITERS=30000, POSE_BACKEND=colmap,
#        DROID_FRAME_STRIDE=3 (auto_segment sampling; wrist sweeps are
#        viewpoint-dense but short, like BEHAVIOR - stride 12 starves the
#        >=2-frame confirmation merge), RESUME=1 skips finished stages.
set -e

SIMANY_ROOT=${SIMANY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RAW_ROOT=${RAW_ROOT:-${SIMANY_ROOT:-$PWD}/data/droid/raw}
EPISODE=${EPISODE:?set EPISODE=<lab>/success/<date>/<ts> (path under $RAW_ROOT)}
EP_DIR=$EPISODE
[ -d "$EP_DIR" ] || EP_DIR=$RAW_ROOT/$EPISODE
[ -f "$EP_DIR/trajectory.h5" ] || { echo "no trajectory.h5 in $EP_DIR - fetch with run/gcs_fetch.py"; exit 1; }

# Scene name: droid_<lab>_<timestamp>, sanitized (it becomes a directory name
# and a MuJoCo model name, so anything outside [A-Za-z0-9_-] is squashed).
if [ -z "${SCENE_NAME:-}" ]; then
  TS=$(basename "$EP_DIR"); LAB=$(echo "$EP_DIR" | sed 's#.*raw/\([^/]*\)/.*#\1#')
  SCENE_NAME=droid_$(echo "${LAB}_${TS}" | tr -c 'A-Za-z0-9_-' '_' | tr '[:upper:]' '[:lower:]')
  SCENE_NAME=${SCENE_NAME%_}
fi

RECON_ROOT=${RECON_ROOT:-$SIMANY_ROOT/data/recon_scenes}
# common.py composes SCENE_DIR=$SCANNETPP_ROOT/data/$SCENE; export BEFORE
# sourcing env.sh so its derived SD/IMG point at the emulated scene too.
export SIMANY_SCANNETPP_ROOT=$RECON_ROOT
export SIMANY_SPLATS_ROOT=$RECON_ROOT/splats
export SIMANY_SCENE=$SCENE_NAME
export SIMANY_OUT=${SIMANY_OUT:-$SIMANY_ROOT/outputs/$SCENE_NAME}
export SIMANY_AUTO=1             # SAM3 discovery, never the GT whitelist
export SIMANY_MESH_SRC=derived   # pipeline mesh = TSDF from our own splat
source "$SIMANY_ROOT/run/env.sh"
mkdir -p "$SIMANY_OUT" "$SIMANY_SPLATS_ROOT"

# h5py is in none of the three SimAny envs (docs/ENVIRONMENTS.md); the
# artifixer venv has h5py+numpy and droid_extract is CPU-only.
H5PY_BIN=${SIMANY_H5_PY:-${SIMANY_H5_PY:-python3}}

CAMERA=${CAMERA:-wrist}
FRAMES=$SD/${CAMERA}_frames
RECON_DIR=$SIMANY_OUT/recon
SPLAT=$SIMANY_SPLATS_ROOT/$SCENE_NAME.ply
GS_ITERS=${GS_ITERS:-30000}
POSE_BACKEND=${POSE_BACKEND:-colmap}
mkdir -p "$RECON_DIR"

echo "droid_recon: $EP_DIR -> scene=$SCENE_NAME out=$SIMANY_OUT"

# ---- 1) episode -> frames + FK trajectory + poses + metric scene dir ------
stage_timed "droid_extract (MP4 frames + FK trajectory, h5py env)"
done_skip "$SD/gt/droid_traj.json" || \
  $H5PY_BIN -m agents.recon.droid_extract --episode "$EP_DIR" \
    --scene-name "$SCENE_NAME" --root "$RECON_ROOT" \
    --camera "$CAMERA" --stride "${STRIDE:-1}" \
    --max-frames "${MAX_FRAMES:-240}"

stage_timed "poses (backend: $POSE_BACKEND)"
if ! done_skip "$RECON_DIR/recon.npz"; then
  case "$POSE_BACKEND" in
    colmap)
      # gold-standard SfM (BA-consistent poses won splat PSNR 25.52 vs
      # 22.52 dB); loud fail below 80% registration -> feed-forward fallback,
      # Umeyama alignment downstream is backend-agnostic
      if ! run agents.recon.colmap_poses --images-dir "$FRAMES" \
             --out "$RECON_DIR/recon.npz" --workdir "$RECON_DIR/colmap"; then
        echo "[droid_recon] COLMAP failed - falling back to feed-forward"
        run agents.models.vggt_scene --images-dir "$FRAMES" \
          --out "$RECON_DIR/recon.npz" --backend omega || \
        run agents.models.vggt_scene --images-dir "$FRAMES" \
          --out "$RECON_DIR/recon.npz" --backend vggt
      fi ;;
    omega)
      run agents.models.vggt_scene --images-dir "$FRAMES" \
        --out "$RECON_DIR/recon.npz" --backend omega ;;
    *)
      run agents.models.vggt_scene --images-dir "$FRAMES" \
        --out "$RECON_DIR/recon.npz" --backend vggt ;;
  esac
fi

stage_timed "align_to_traj (Umeyama -> metric ROBOT BASE frame; no metricize)"
done_skip "$RECON_DIR/recon_base.npz" || \
  run agents.recon.align_to_traj --recon "$RECON_DIR/recon.npz" \
    --traj "$SD/gt/droid_traj.json" --out "$RECON_DIR/recon_base.npz"

stage_timed "make_scene_dir (emulated ScanNet++ layout)"
done_skip "$SD/dslr/colmap/images.txt" || \
  run agents.recon.make_scene_dir --recon "$RECON_DIR/recon_base.npz" \
    --frames-dir "$FRAMES" --scene "$SCENE_NAME" --root "$RECON_ROOT"

# ---- resolution-scaled pixel gates (same reasoning as run_behavior_recon.sh:
# factory_prepare's 48/400 px gates were tuned on 1752x1168; wrist MP4s are
# 1280x720, where they would gate out most tabletop manipulanda) -------------
GATES=$(python3 -c "
import json, sys
h = json.load(open(sys.argv[1]))['h']
ref = 1168.0  # validated ScanNet++ height the gates were tuned on
print(max(6, round(48 * h / ref)), max(9, round(400 * (h / ref) ** 2)))
" "$SD/dslr/nerfstudio/transforms_undistorted.json")
export SIMANY_MIN_BBOX_PX=${SIMANY_MIN_BBOX_PX:-${GATES% *}}
export SIMANY_MIN_MASK_PX=${SIMANY_MIN_MASK_PX:-${GATES#* }}
echo "[droid_recon] pixel gates: MIN_BBOX_PX=$SIMANY_MIN_BBOX_PX" \
     "MIN_MASK_PX=$SIMANY_MIN_MASK_PX"

stage_timed "gsplat_train (3DGS, mini-viewer env)"
done_skip "$SPLAT" || \
  run_gs agents.recon.gsplat_train --scene-dir "$SD" \
    --init-ply "$SD/init_points.ply" --out "$SPLAT" --iters "$GS_ITERS"

# ---- 2) AUTO + derived-mesh tail (mirrors run_video2sim.sh) ----------------
stage_timed "derive mesh: render (splat depth, mini-viewer env)"
done_skip "$SIMANY_OUT/derived_mesh.ply" || \
  run_gs agents.discover.derive_mesh_from_splat render

stage_timed "derive mesh: fuse (TSDF, .venv)"
done_skip "$SIMANY_OUT/derived_mesh.ply" || {
  run agents.discover.derive_mesh_from_splat fuse && sync; }

stage_timed "auto_segment (SAM3, against derived mesh)"
done_skip "$SIMANY_OUT/auto_instances.npz" || \
  run_sam3 agents.discover.auto_segment --scene-dir "$SD" \
    --out-dir "$SIMANY_OUT" --mesh-path "$SIMANY_OUT/derived_mesh.ply" \
    --frame-stride "${DROID_FRAME_STRIDE:-3}"

stage_timed "factory_prepare (auto instances + best-view crops)"
done_skip "$SIMANY_OUT/objects/objects.json" || run agents.discover.factory_prepare

stage_timed "factory_refine_masks (SAM3 evidence)"
done_skip "$SIMANY_OUT/objects/.masks_refined" || {
  run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" \
    --out-dir "$SIMANY_OUT" &&
  touch "$SIMANY_OUT/objects/.masks_refined"; }

stage_timed "s4 TRELLIS image-to-3D"
run agents.models.s4_trellis

stage_timed "factory_align (register to own extraction)"
run agents.assets.factory_align

stage_timed "s6 CoACD + physics annotation + URDF"
run agents.assets.s6_physics

stage_timed "factory_report (drop test + yield, GT-free in AUTO mode)"
run agents.eval.factory_report

stage_timed "export MJCF + Isaac manifest + MuJoCo settle test"
run robo.sim.export_mjcf --test

# soft-fail: a DROID tabletop can legitimately yield <2 graspable assets, and
# the recon metrics above are still the pilot's point
stage_timed "pi05_tasks (pick-and-place task suite)"
run robo.tasks.pi05_tasks --out-dir "$SIMANY_OUT" || \
  echo "[droid_recon] pi05_tasks FAILED (usually <2 graspable objects)"

stage_timed "DONE -> $SIMANY_OUT"
