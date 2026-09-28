#!/bin/bash
# Video -> simulation, fully automatic (no dataset, no GT): a casual phone
# video becomes an emulated ScanNet++-style scene dir + trained 3DGS splat,
# then runs the AUTO+derived-mesh pipeline inline (same stage list as
# run/slurm/ablation_rowC_scene.sbatch, plus export_mjcf --test as in
# run/run_auto.sh, plus a pi05_tasks suite; GT-only eval stages skipped).
#
# Usage (from anywhere; GPU node with the three envs, see docs/ENVIRONMENTS.md):
#   bash run/run_video2sim.sh --video /path/to/clip.mp4   # one video
#   VIDEO=/path/to/clip.mp4 bash run/run_video2sim.sh     # same, via env
#   bash run/run_video2sim.sh                             # every videos/*.mp4
# Knobs: RESUME=1 skips stages whose output exists; GS_ITERS (default 15000).
set -e

# ---- recon-scenes contract (single source of truth) ------------------------
# common.py:47 composes SCENE_DIR = $SIMANY_SCANNETPP_ROOT/data/$SIMANY_SCENE,
# so the emulated root NEEDS the extra data/ level:
#   scene dir   $RECON_ROOT/data/<scene>/dslr/{resized_undistorted_images/,
#               nerfstudio/transforms_undistorted.json, colmap/images.txt}
#   splats      $RECON_ROOT/splats/<scene>.ply   (= $SIMANY_SPLATS_ROOT/<scene>.ply)
# agents/recon/make_scene_dir.py takes --root $RECON_ROOT and writes the
# data/<scene> level itself; keep the two sides of this contract in sync.
RECON_ROOT=${SIMANY_ROOT:-$PWD}/data/recon_scenes

# ---- pick the video(s) -----------------------------------------------------
VIDEO=${VIDEO:-}
if [ "${1:-}" = "--video" ]; then VIDEO=$2; shift 2; fi

if [ -z "$VIDEO" ]; then
  # No video named: fan out over the drop folder. Each clip runs in a child
  # invocation of this same script so env.sh's source-time SD/IMG derivation
  # (and any per-scene state) can never leak between scenes; one bad clip
  # does not kill the batch.
  VDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../videos" && pwd)
  shopt -s nullglob; vids=("$VDIR"/*.mp4); shopt -u nullglob
  [ ${#vids[@]} -gt 0 ] || { echo "no .mp4 in $VDIR"; exit 1; }
  fails=0
  for v in "${vids[@]}"; do
    echo "### video2sim: $v"
    bash "${BASH_SOURCE[0]}" --video "$v" || { echo "### FAILED: $v"; fails=$((fails+1)); }
  done
  echo "### video2sim batch done: ${#vids[@]} video(s), $fails failure(s)"
  exit $((fails > 0))
fi
[ -f "$VIDEO" ] || { echo "video not found: $VIDEO"; exit 1; }

# Scene name = sanitized basename (it becomes a directory name and a MuJoCo
# model name, so anything outside [A-Za-z0-9_-] is squashed to _).
SCENE=$(basename "$VIDEO"); SCENE=${SCENE%.*}; SCENE=${SCENE//[^A-Za-z0-9_-]/_}

# Exports must precede the env.sh source: env.sh derives SD/IMG from
# SIMANY_SCENE + SIMANY_SCANNETPP_ROOT at source time, and common.py reads
# these in every python stage. Unconditional on purpose - pointing the whole
# pipeline at the emulated root is the point of this launcher.
export SIMANY_SCENE=$SCENE
export SIMANY_SCANNETPP_ROOT=$RECON_ROOT
export SIMANY_SPLATS_ROOT=$RECON_ROOT/splats
export SIMANY_AUTO=1           # SAM3 discovery, never the GT whitelist
export SIMANY_MESH_SRC=derived # pipeline mesh = TSDF from our own splat

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export SIMANY_OUT=${SIMANY_OUT:-$ROOT/outputs/video_${SCENE}}
mkdir -p "$SIMANY_OUT"

# Recon intermediates live under the run output, not the scene dir: the scene
# dir must stay byte-compatible with the ScanNet++ layout common.py expects.
RECON_DIR=$SIMANY_OUT/recon
FRAMES=$RECON_DIR/frames
SPLAT=$SIMANY_SPLATS_ROOT/$SCENE.ply
mkdir -p "$FRAMES" "$SIMANY_SPLATS_ROOT"
# Quality defaults ("use everything"): COLMAP poses when they register
# (gold standard; ~5-15 min CPU), feed-forward fallback otherwise; 30k
# training iters; 240 frames. POSE_BACKEND=colmap|omega|vggt overrides.
GS_ITERS=${GS_ITERS:-30000}
MAX_FRAMES=${MAX_FRAMES:-240}
POSE_BACKEND=${POSE_BACKEND:-colmap}

echo "video2sim: $VIDEO -> scene=$SCENE out=$SIMANY_OUT"

# ---- 1) video -> posed metric scene dir + splat ---------------------------
stage_timed "frames (ffmpeg uniform subsample)"
done_skip "$FRAMES/.frames_done" || {
  run agents.recon.frames --video "$VIDEO" --out-dir "$FRAMES" \
    --max-frames "$MAX_FRAMES" &&
  touch "$FRAMES/.frames_done"; }

stage_timed "poses + depth + points (backend: $POSE_BACKEND)"
if ! done_skip "$RECON_DIR/recon.npz"; then
  case "$POSE_BACKEND" in
    colmap)
      # gold-standard SfM; falls back to feed-forward when too few frames
      # register (textureless / fast motion)
      if ! run agents.recon.colmap_poses --images-dir "$FRAMES" \
             --out "$RECON_DIR/recon.npz" --workdir "$RECON_DIR/colmap"; then
        echo "[video2sim] COLMAP failed - falling back to feed-forward"
        run models.vggt_scene --images-dir "$FRAMES" \
          --out "$RECON_DIR/recon.npz" --backend omega || \
        run models.vggt_scene --images-dir "$FRAMES" \
          --out "$RECON_DIR/recon.npz" --backend vggt
      fi ;;
    omega)
      run models.vggt_scene --images-dir "$FRAMES" \
        --out "$RECON_DIR/recon.npz" --backend omega ;;
    *)
      run models.vggt_scene --images-dir "$FRAMES" \
        --out "$RECON_DIR/recon.npz" --backend vggt ;;
  esac
fi

stage_timed "metricize (monodepth scale + z-up + floor at z=0)"
done_skip "$RECON_DIR/recon_metric.npz" || \
  run agents.recon.metricize --recon "$RECON_DIR/recon.npz" \
    --images-dir "$FRAMES" --out "$RECON_DIR/recon_metric.npz"

stage_timed "make_scene_dir (emulated ScanNet++ layout)"
done_skip "$SD/dslr/colmap/images.txt" || \
  run agents.recon.make_scene_dir --recon "$RECON_DIR/recon_metric.npz" \
    --frames-dir "$FRAMES" --scene "$SCENE" --root "$RECON_ROOT"

# make_scene_dir writes init_points.ply next to its scene layout; probe the
# root too so a builder-side relocation fails loudly here, not inside training.
INIT_PLY=$SD/init_points.ply
[ -f "$INIT_PLY" ] || INIT_PLY=$(find "$RECON_ROOT" -name init_points.ply -path "*$SCENE*" 2>/dev/null | head -1)
[ -n "$INIT_PLY" ] && [ -f "$INIT_PLY" ] || { echo "init_points.ply not found under $SD or $RECON_ROOT"; exit 1; }

stage_timed "gsplat_train (3DGS, mini-viewer env)"
done_skip "$SPLAT" || \
  run_gs agents.recon.gsplat_train --scene-dir "$SD" --init-ply "$INIT_PLY" \
    --out "$SPLAT" --iters "$GS_ITERS"

# ---- 2) AUTO + derived-mesh pipeline (mirrors ablation_rowC_scene.sbatch,
# ---- then export_mjcf as in run_auto.sh, plus pi05_tasks) ------------------
stage_timed "derive mesh: render (splat depth, mini-viewer env)"
done_skip "$SIMANY_OUT/derived_mesh.ply" || \
  run_gs agents.discover.derive_mesh_from_splat render

stage_timed "derive mesh: fuse (TSDF, .venv)"
done_skip "$SIMANY_OUT/derived_mesh.ply" || {
  run agents.discover.derive_mesh_from_splat fuse && sync; }

stage_timed "auto_segment (SAM3, against derived mesh)"
done_skip "$SIMANY_OUT/auto_instances.npz" || \
  run_sam3 agents.discover.auto_segment --scene-dir "$SD" \
    --out-dir "$SIMANY_OUT" --mesh-path "$SIMANY_OUT/derived_mesh.ply"

stage_timed "factory_prepare (auto instances + best-view crops)"
done_skip "$SIMANY_OUT/objects/objects.json" || run agents.discover.factory_prepare

stage_timed "factory_refine_masks (SAM3 evidence)"
done_skip "$SIMANY_OUT/objects/.masks_refined" || {
  run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" --out-dir "$SIMANY_OUT" &&
  touch "$SIMANY_OUT/objects/.masks_refined"; }

stage_timed "s4 TRELLIS image-to-3D"
run models.s4_trellis

stage_timed "factory_align (register to own extraction)"
run agents.assets.factory_align

stage_timed "s6 CoACD + physics annotation + URDF"
run agents.assets.s6_physics

# factory_report is GT-free in AUTO mode (drop test + tiers from the
# pipeline's own aligned.json); factory_eval_render is NOT run - it scores
# renders against dataset GT recipes, meaningless for a phone video.
stage_timed "factory_report (drop test + yield)"
run agents.eval.factory_report

stage_timed "export MJCF + Isaac manifest + MuJoCo settle test"
run robo.sim.export_mjcf --test

stage_timed "pi05_tasks (pick-and-place task suite)"
run robo.tasks.pi05_tasks --out-dir "$SIMANY_OUT"

stage_timed "DONE -> $SIMANY_OUT"
