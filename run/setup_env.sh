#!/bin/bash
# SimAny environment setup.
# Installs everything into the repo-local .venv (conda-created python 3.11).
# torch 2.4.1+cu124 is chosen to match the prebuilt gsplat wheel (pt24cu124),
# same combo as the known-good mini-viewer env.
set -x
# Repo root derived from this file's location, so the tree can be moved/cloned.
ROOT=${SIMANY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PIP="$ROOT/.venv/bin/pip"
PY="$ROOT/.venv/bin/python"
export PIP_CACHE_DIR="$ROOT/.pip-cache"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"

$PIP install --no-input torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 || exit 1

$PIP install --no-input "numpy<2.3" scipy opencv-python-headless pillow imageio imageio-ffmpeg \
    trimesh plyfile matplotlib scikit-learn pyyaml einops easydict tqdm rich lxml networkx || exit 1

$PIP install --no-input open3d || echo "WARN: open3d failed"
$PIP install --no-input pybullet || echo "WARN: pybullet failed"
$PIP install --no-input coacd || echo "WARN: coacd failed"

$PIP install --no-input transformers==4.57.6 accelerate safetensors sentencepiece || echo "WARN: transformers failed"

# xformers matching torch 2.4.1 (TRELLIS attention backend)
$PIP install --no-input xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 || echo "WARN: xformers failed"

# gsplat prebuilt wheel for pt24cu124
$PIP install --no-input gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124 || \
  $PIP install --no-input gsplat --find-links https://docs.gsplat.studio/whl/pt24cu124/gsplat/ || echo "WARN: gsplat failed"

# TRELLIS sparse conv backend
$PIP install --no-input spconv-cu120 || echo "WARN: spconv failed"

# DepthAnything3 (paper's V_im2depth). Try pypi then github.
$PIP install --no-input depth-anything-3 || \
  $PIP install --no-input "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" || echo "WARN: DA3 failed"

# LaMa inpainting (image inpaint backend); harmless if it fails (cv2 Telea fallback exists)
$PIP install --no-input simple-lama-inpainting || echo "WARN: lama failed"

# TRELLIS repo (vendored, code only; ckpts already in HF cache)
if [ ! -d "$ROOT/third_party/TRELLIS" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/microsoft/TRELLIS.git "$ROOT/third_party/TRELLIS" || echo "WARN: TRELLIS clone failed"
fi

echo "=== IMPORT SMOKE TEST ==="
$PY - <<'EOF'
import importlib
for m in ["torch", "torchvision", "numpy", "scipy", "cv2", "PIL", "imageio", "trimesh",
          "plyfile", "matplotlib", "sklearn", "yaml", "einops", "easydict", "open3d",
          "pybullet", "coacd", "transformers", "xformers", "gsplat", "spconv",
          "depth_anything_3", "simple_lama_inpainting", "lxml"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
import torch
print("torch cuda built:", torch.version.cuda)
EOF
echo "=== SETUP DONE ==="
