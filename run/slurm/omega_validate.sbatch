#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH -t 0:40:00
#SBATCH -J omega_validate
#SBATCH -o outputs/omega_validate_%j.log
# First-use validation of the VGGT-Omega backend (run_chunk_omega was written
# against the published API before the gated weights were available): run it
# on the pilot frames and score the poses against GT COLMAP, next to the
# recorded VGGT (12.3 cm / 0.5 deg) and COLMAP (82% reg) reference numbers.
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
FRAMES=$SIMANY_ROOT/outputs/video_pilot_a29cccc784/recon/frames
OUT=$SIMANY_ROOT/outputs/video_pilot_a29cccc784/recon/recon_omega.npz

run models.vggt_scene --images-dir "$FRAMES" --out "$OUT" --backend omega

$VENV /tmp/claude-10064/-group-worldcept/8444a2f5-faa5-44c5-bc1e-21ebe7059853/scratchpad/check_recon_vs_gt.py \
  "$OUT" /data/ScanNetpp/data/a29cccc784/dslr/colmap/images.txt 240
echo "OMEGA_VALIDATE_DONE"
