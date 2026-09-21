#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source run/env.sh
export UV_CACHE_DIR="$PHYSICALVIEW_ROOT/.uv-cache"
uv pip install --python "$STUDIO_PY" --target .envs/phiview-inference --no-deps -r run/requirements-phiview-inference.txt
if [ ! -d .envs/phiview-sam3-source/.git ]; then
  git clone https://github.com/facebookresearch/sam3.git .envs/phiview-sam3-source
fi
SAM3_REV=$(cat run/phiview-sam3-revision.txt)
git -C .envs/phiview-sam3-source checkout "$SAM3_REV"
run/phiview_inference_python.sh -c 'from diffusers import QwenImageEditPlusPipeline; from sam3.model_builder import build_sam3_image_model; print("PhiView inference imports passed")'
