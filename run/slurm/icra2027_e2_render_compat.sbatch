#!/usr/bin/env bash
#SBATCH --job-name=icra-e2-render-compat
#SBATCH --partition=debug
#SBATCH --nodelist=hala
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gpus=a6000:1
#SBATCH --output=outputs/icra2027/slurm/e2-render-compat-%j.out
#SBATCH --error=outputs/icra2027/slurm/e2-render-compat-%j.err

# One-scene E2 compatibility smoke for CUDA gsplat rendering and LPIPS.
# This is not a paper evaluation. It reuses frozen source objects read-only and
# writes the two renderer products beneath a new job-specific ICRA output root.
set -euo pipefail

ROOT=${SIMANY_ROOT:-$PWD}
SCENE=5748ce6f01
SOURCE_OUT="$ROOT/outputs/${SCENE}_factory"
DATA_SCENE=/data/ScanNetpp/data/$SCENE
SPLAT=/data/ScanNetppv2_gsplat/splats/$SCENE.ply

: "${SLURM_JOB_ID:?this launcher must run under Slurm}"
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }

# Read-only input preflight. The selected scene has one accepted object, so it
# exercises both the background renderer and factorized-object composition.
required_inputs=(
  "$ROOT/run/env.sh"
  "$SOURCE_OUT/objects/objects.json"
  "$SOURCE_OUT/objects/obj_00/aligned.json"
  "$SOURCE_OUT/objects/obj_00/trellis_gs.ply"
  "$DATA_SCENE/dslr/train_test_lists.json"
  "$DATA_SCENE/dslr/nerfstudio/transforms_undistorted.json"
  "$DATA_SCENE/dslr/colmap/images.txt"
  "$SPLAT"
)
for path in "${required_inputs[@]}"; do
  [[ -s "$path" ]] || { echo "missing or empty required input: $path" >&2; exit 2; }
done
[[ -d "$DATA_SCENE/dslr/resized_undistorted_images" ]] || {
  echo "missing image directory: $DATA_SCENE/dslr/resized_undistorted_images" >&2
  exit 2
}
jq -e 'length == 1 and .[0].index == 0' \
  "$SOURCE_OUT/objects/objects.json" >/dev/null || {
  echo "source object population drifted; expected exactly obj_00" >&2
  exit 2
}
jq -e '.rejected == null and (.T | length == 4)' \
  "$SOURCE_OUT/objects/obj_00/aligned.json" >/dev/null || {
  echo "obj_00 is no longer an accepted aligned object" >&2
  exit 2
}

cd "$ROOT"
CACHE_ROOT="$ROOT/.cache/icra2027/e2-render-compat"
export TMPDIR="$ROOT/.tmp/icra2027/e2-render-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions-a6000"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

SMOKE_OUT="$ROOT/outputs/icra2027/e2-render-compat-$SLURM_JOB_ID/${SCENE}_factory"
if [[ -e "$SMOKE_OUT" || -L "$SMOKE_OUT" ]]; then
  echo "refusing to overwrite existing smoke output: $SMOKE_OUT" >&2
  exit 2
fi

mkdir -p \
  "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$SMOKE_OUT"
ln -s "$SOURCE_OUT/objects" "$SMOKE_OUT/objects"

source run/env.sh
[[ -x "$MVPY" ]] || { echo "gsplat Python is not executable: $MVPY" >&2; exit 2; }

SIMANY_SCENE="$SCENE" SIMANY_OUT="$SMOKE_OUT" \
  run_gs agents.eval.factory_eval_render

[[ -s "$SMOKE_OUT/render_metrics_v2.json" ]] || {
  echo "renderer did not produce render_metrics_v2.json" >&2
  exit 1
}
[[ -s "$SMOKE_OUT/eval_photo_bg_twin.jpg" ]] || {
  echo "renderer did not produce eval_photo_bg_twin.jpg" >&2
  exit 1
}
jq -e '
  .split == "official-test"
  and .n_objects_composited == 1
  and (.eval_frames | length == 8)
  and (.frames | length == 8)
  and .eval_frames == [.frames[].frame]
  and ([.frames[] |
        (has("bg") and has("twin")
         and (.bg | has("psnr") and has("ssim") and has("lpips"))
         and (.twin | has("psnr") and has("ssim") and has("lpips")))] | all)
' "$SMOKE_OUT/render_metrics_v2.json" >/dev/null

echo "E2_RENDER_COMPAT=PASS"
echo "scene=$SCENE"
echo "output=$SMOKE_OUT"
