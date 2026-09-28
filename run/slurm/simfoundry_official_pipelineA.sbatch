#!/bin/bash
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=simfoundry_pipeA
#SBATCH --output=outputs/simfoundry_official/pipelineA_%j.log

# Stage 3 of the official-SimFoundry integration attempt: run their Pipeline A
# (video -> OmniGibson scene, stages 1b-14) on THEIR OWN example video first
# (docs/assets/example_videos/PutCupInBowl.mp4), as the cheapest end-to-end
# proof the install works, before pointing it at any of our scenes.
#
# Scene/video are overridable so the SAME script serves the follow-up run on
# our own video:
#   sbatch --export=ALL,SCENE_NAME=pilot_a29cccc784,VIDEO_FPATH=${SIMANY_ROOT:-$PWD}/videos/pilot_a29cccc784.mp4 \
#     run/slurm/simfoundry_official_pipelineA.sbatch

set -uo pipefail

SIMF=${SIMANY_ROOT:-$PWD}/third_party/SimFoundry
SCENE_NAME="${SCENE_NAME:-PutCupInBowl_repro}"
VIDEO_FPATH="${VIDEO_FPATH:-$SIMF/docs/assets/example_videos/PutCupInBowl.mp4}"

export PATH="$HOME/bin:$PATH"           # mamba -> micromamba shim (see install sbatch)
export PIP_CONFIG_FILE=/dev/null         # site pip.conf carries a dead NGC extra index
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} # reuse the existing gated-model cache (facebook/sam3 already present)

# Same CUDA-env sanitization as the install sbatch: the submission shell
# leaks CUDA_HOME=/opt/modules/nvidia-cuda-12.4.1 (+ CUDA_PATH /
# LD_LIBRARY_PATH), which would hijack any torch cpp_extension JIT compile
# at runtime the same way it broke install_da3.sh's xformers build.
unset CUDA_HOME CUDA_PATH PKG_CONFIG_PATH MODULES_CURRENTLY_ACTIVE
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/opt/modules/nvidia-cuda' | paste -sd:)"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/opt/modules/nvidia-cuda' | paste -sd:)"

# Gemini via the internal Cloudflare AI-Gateway (native endpoint), verified
# 2026-09-01 for every model real2sim_cfg.yaml requests (text: 2.5-pro/flash,
# 3.1-pro-preview; image out: 2.5-flash-image, 3-pro-image -- real decodable
# image bytes confirmed). Zero SimFoundry code changes: GEMINI_API_KEY makes
# their resolve_gemini_auth() take its documented "api_key" route, and
# google-genai natively honors GOOGLE_GEMINI_BASE_URL (google/genai/_base_url.py).
# Token is sourced from the existing credential file rather than copied to a
# new file on disk.
set -a; source ${PHIRIE_WORKSPACE:-$HOME}/code/openrouter/.env; set +a
export GEMINI_API_KEY="$OPENAI_API_KEY"
export GOOGLE_GEMINI_BASE_URL="https://ai-gateway-native-gemini.plain-flower-4887.workers.dev/"
unset OPENAI_API_KEY CF_PROJECT_ID CF_USER_ID MODEL_ID  # keep non-Gemini routes from grabbing the gateway token

echo "=== $(date -u) preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v mamba && mamba --version
echo "scene=${SCENE_NAME} video=${VIDEO_FPATH}"
echo "GEMINI key set: $([ -n "${GEMINI_API_KEY}" ] && echo yes) base_url=${GOOGLE_GEMINI_BASE_URL}"
hf auth whoami 2>/dev/null | head -1 || echo "WARN: hf auth whoami failed"

cd "$SIMF"

echo "=== $(date -u) checkpoints (idempotent; VOID + FoundationStereo failures tolerated) ==="
# --default: no prompts. The script tries every asset independently and exits
# non-zero if ANY failed; only a subset is load-bearing for video-mode
# Pipeline A: SAM2.1 + DepthPro (direct URLs), RealESRGAN (stage 7 hunyuan),
# FoundationPose refiner+scorer (stage 8 any6d, gdown -- the documented flaky
# one, so retry it). FoundationStereo is stereo-mode-only and netflix/void-model
# is auto-BG-only (and gated, not requested for this account) -- both may fail
# without affecting this run.
for i in 1 2 3; do
  bash scripts/installation/download_checkpoints.sh --default && break
  echo "--- checkpoint pass $i exited non-zero (expected for VOID/FoundationStereo; retrying for the gdown-backed ones) ---"
  sleep 30
done
MISSING=0
for f in \
  checkpoints/sam2.1_hiera_large.pt \
  deps/ml-depth-pro/checkpoints/depth_pro.pt \
  deps/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth \
  deps/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth \
  deps/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth; do
  if [ ! -f "$f" ]; then echo "MISSING REQUIRED CHECKPOINT: $f"; MISSING=1; fi
done
[ "$MISSING" -eq 1 ] && echo "WARN: proceeding anyway -- the stage that needs a missing checkpoint will fail visibly rather than silently"

echo "=== $(date -u) Pipeline A start: stages 1b-14 on ${SCENE_NAME} ==="
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name "${SCENE_NAME}" \
  --video-fpath "${VIDEO_FPATH}"
STATUS=$?
echo "=== $(date -u) Pipeline A exited ${STATUS} ==="

OUT_DIR="$SIMF/Data/${SCENE_NAME}"
echo "=== output tree (top 2 levels) ==="
find "$OUT_DIR" -maxdepth 2 2>/dev/null | head -60
echo "=== key artifacts ==="
ls -la "$OUT_DIR/s14_og/" 2>/dev/null || echo "no s14_og dir (pipeline did not reach stage 14)"
exit $STATUS
