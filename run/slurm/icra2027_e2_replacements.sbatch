#!/usr/bin/env bash
#SBATCH --job-name=icra-e2-replace
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:50:00
#SBATCH --output=outputs/icra2027/slurm/e2-replace-%j.out
#SBATCH --error=outputs/icra2027/slurm/e2-replace-%j.err

# Build the five fixed leakage-remediation bundles (eight objects total).
# This script never edits outputs/<scene>_{factory,auto}; every job publishes
# one fresh directory beneath the frozen E2 output using an atomic rename.
# It is intentionally not submitted by this repository change.
set -euo pipefail

ROOT=${SIMANY_ROOT:-$PWD}
PYTHON_ENV="$ROOT/.envs/sam3d-objects"
PYTHON="$PYTHON_ENV/bin/python"
SAM3PY="$PYTHON"
CONFIG="$ROOT/configs/experiments/icra2027/fidelity_manifest.json"
DATASET_ROOT=/data/ScanNetpp

# Inherited import/optimization settings are not part of the frozen contract.
unset PYTHONPATH PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

: "${SLURM_JOB_ID:?this launcher must run under Slurm}"
: "${SLURMD_NODENAME:?missing Slurm execution node}"
: "${SLURM_JOB_PARTITION:?missing Slurm partition}"
: "${SLURM_JOB_QOS:?missing Slurm QoS}"
: "${E2_GPU_PROFILE:?replacement jobs require E2_GPU_PROFILE=hala-a6000 or gcp-a100}"
: "${E2_BUNDLE_ID:?submit one non-array job with a pinned E2_BUNDLE_ID}"
: "${E2_FREEZE_ID:?submit with a fresh E2_FREEZE_ID inheriting the E0 freeze ID}"
: "${E2_CODE_COMMIT:?submit with the exact 40-character code commit}"
: "${E2_CONTRACT_MANIFEST:?submit with the fresh repository-local E0 freeze_manifest.json}"

if [[ -n "${SLURM_ARRAY_JOB_ID+x}" \
  || -n "${SLURM_ARRAY_TASK_ID+x}" \
  || -n "${SLURM_ARRAY_TASK_COUNT+x}" \
  || -n "${SLURM_ARRAY_TASK_MIN+x}" \
  || -n "${SLURM_ARRAY_TASK_MAX+x}" \
  || -n "${SLURM_ARRAY_TASK_STEP+x}" ]]; then
  echo "replacement launcher rejects Slurm array jobs; submit one E2_BUNDLE_ID per job" >&2
  exit 2
fi

# Slurm directives cannot be selected from an exported variable, so the
# submitter must request the resources belonging to one explicit validated
# profile. The runtime then verifies every scheduler/device attribute exactly.
case "$E2_GPU_PROFILE" in
  hala-a6000)
    E2_EXPECTED_NODE=hala
    E2_EXPECTED_PARTITION=debug
    E2_EXPECTED_QOS=debug
    E2_EXPECTED_GPU_NAME="NVIDIA RTX A6000"
    E2_EXPECTED_GPU_CC=8.6
    E2_TORCH_CUDA_ARCH_LIST=8.6
    E2_CUMM_CUDA_ARCH_LIST=8.6
    ;;
  gcp-a100)
    E2_EXPECTED_NODE=gcp-eu1-a100-80g-qrfh
    E2_EXPECTED_PARTITION=batch
    E2_EXPECTED_QOS=normal
    E2_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"
    E2_EXPECTED_GPU_CC=8.0
    E2_TORCH_CUDA_ARCH_LIST=8.0
    E2_CUMM_CUDA_ARCH_LIST=8.0
    ;;
  *)
    echo "unsupported E2_GPU_PROFILE=$E2_GPU_PROFILE (expected hala-a6000 or gcp-a100)" >&2
    exit 2
    ;;
esac
[[ "$SLURMD_NODENAME" == "$E2_EXPECTED_NODE" ]] || {
  echo "profile=$E2_GPU_PROFILE requires node=$E2_EXPECTED_NODE; found $SLURMD_NODENAME" >&2
  exit 2
}
[[ "$SLURM_JOB_PARTITION" == "$E2_EXPECTED_PARTITION" ]] || {
  echo "profile=$E2_GPU_PROFILE requires partition=$E2_EXPECTED_PARTITION; found $SLURM_JOB_PARTITION" >&2
  exit 2
}
[[ "$SLURM_JOB_QOS" == "$E2_EXPECTED_QOS" ]] || {
  echo "profile=$E2_GPU_PROFILE requires qos=$E2_EXPECTED_QOS; found $SLURM_JOB_QOS" >&2
  exit 2
}
export E2_GPU_PROFILE E2_EXPECTED_GPU_NAME E2_EXPECTED_GPU_CC
[[ "$E2_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E2_FREEZE_ID contains unsafe path characters: $E2_FREEZE_ID" >&2
  exit 2
}
[[ "$E2_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E2_CODE_COMMIT must be an exact lowercase 40-character Git SHA" >&2
  exit 2
}
case "$E2_CONTRACT_MANIFEST" in
  "$ROOT"/*) ;;
  *) echo "E2_CONTRACT_MANIFEST must be an absolute path below $ROOT" >&2; exit 2 ;;
esac

case "$E2_BUNDLE_ID" in
  0d2ee665be-auto|0d2ee665be-factory|286b55a2bf-auto|286b55a2bf-factory|3f15a9266d-factory)
    BUNDLE_ID=$E2_BUNDLE_ID
    ;;
  *)
    echo "E2_BUNDLE_ID is not one of the five pinned replacement bundles: $E2_BUNDLE_ID" >&2
    exit 2
    ;;
esac
OUT="$ROOT/outputs/icra2027/$E2_FREEZE_ID/fidelity/replacements/$BUNDLE_ID"
RUNTIME="$ROOT/outputs/icra2027/$E2_FREEZE_ID/runtime/replacements/${SLURM_JOB_ID}"

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing main Python: $PYTHON" >&2; exit 2; }
[[ -x "$SAM3PY" ]] || { echo "missing SAM3 Python: $SAM3PY" >&2; exit 2; }
[[ -s "$CONFIG" && ! -L "$CONFIG" ]] || {
  echo "missing, empty, or symlinked E2 config: $CONFIG" >&2
  exit 2
}
[[ -s "$E2_CONTRACT_MANIFEST" && ! -L "$E2_CONTRACT_MANIFEST" ]] || {
  echo "missing, empty, or symlinked E0 contract: $E2_CONTRACT_MANIFEST" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$E2_CODE_COMMIT" ]] || {
  echo "code commit drift: expected $E2_CODE_COMMIT, found $(git rev-parse HEAD)" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "refusing replacement generation from a dirty worktree" >&2
  git status --short >&2
  exit 2
}

# The config is the E0-frozen authority for every source tree and checkpoint.
# Recompute all tree/file hashes before either reusing or generating a bundle.
mapfile -t MODEL_INPUTS < <(
  "$PYTHON" - "$CONFIG" "$ROOT" <<'PY'
import sys
from pathlib import Path

from robo.eval.fidelity_replacements import _load_config, _validate_model_inputs

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
config, _bundles = _load_config(config_path)
_manifest, resolved = _validate_model_inputs(config, root=root)
for name in (
    "trellis_source", "trellis_snapshot", "dinov2_source",
    "dinov2_checkpoint", "sam3_source", "sam3_checkpoint",
):
    print(resolved[name])
print(resolved["dinov2_source"].parent.parent)
print(config["leakage_remediation"]["model_inputs"]["sam3_source"]["package_version"])
PY
)
[[ "${#MODEL_INPUTS[@]}" == 8 ]] || {
  echo "model-input validator did not return six inputs, TORCH_HOME, and the SAM3 version" >&2
  exit 2
}
export SIMANY_TRELLIS_DIR="${MODEL_INPUTS[0]}"
export SIMANY_TRELLIS_MODEL="${MODEL_INPUTS[1]}"
export SIMANY_DINOV2_REPO="${MODEL_INPUTS[2]}"
SAM3_SOURCE="${MODEL_INPUTS[4]}"
export SIMANY_SAM3_CKPT="${MODEL_INPUTS[5]}"
export TORCH_HOME="${MODEL_INPUTS[6]}"
SAM3_VERSION="${MODEL_INPUTS[7]}"
export SIMANY_SCANNETPP_ROOT="$DATASET_ROOT"
unset SIMANY_ROOT SIMF_ROOT SIMANY_MESH_SRC SIMF_MESH_SRC SIMANY_FULL SIMF_FULL
unset SIMANY_OUT SIMF_OUT SIMANY_SCENE SIMF_SCENE SIMANY_AUTO SIMF_AUTO
unset SIMF_MIN_BBOX_PX SIMF_VIS_TOL_M SIMF_MIN_MASK_PX

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$VISIBLE_GPUS" && "$VISIBLE_GPUS" != NoDevFiles && "$VISIBLE_GPUS" != *,* ]] || {
  echo "replacement profile=$E2_GPU_PROFILE requires exactly one visible GPU; CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS" >&2
  exit 2
}
"$PYTHON" - "$E2_EXPECTED_GPU_NAME" "$E2_EXPECTED_GPU_CC" <<'PY'
import sys

import torch

expected_name, expected_cc = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to the E2 replacement job")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"E2 replacement requires one visible CUDA device; found {torch.cuda.device_count()}"
    )
observed_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
observed_cc = f"{major}.{minor}"
if observed_name != expected_name:
    raise SystemExit(
        f"GPU name mismatch: expected {expected_name!r}, found {observed_name!r}"
    )
if observed_cc != expected_cc:
    raise SystemExit(
        f"GPU compute capability mismatch: expected {expected_cc}, found {observed_cc}"
    )
print(f"gpu_profile={__import__('os').environ['E2_GPU_PROFILE']}")
print(f"gpu_name={observed_name}")
print(f"gpu_compute_capability={observed_cc}")
PY

VALIDATE_COMMAND=(
  "$PYTHON" -m robo.eval.fidelity_replacements
  --config "$CONFIG"
  --contract-manifest "$E2_CONTRACT_MANIFEST"
  --bundle-id "$BUNDLE_ID"
  --freeze-id "$E2_FREEZE_ID"
  --out "$OUT"
  --dataset-root "$DATASET_ROOT"
  --validate-existing
)
if [[ -e "$OUT" || -L "$OUT" ]]; then
  [[ -d "$OUT" && ! -L "$OUT" ]] || {
    echo "existing replacement path is not a regular directory: $OUT" >&2
    exit 2
  }
  "${VALIDATE_COMMAND[@]}"
  echo "E2_REPLACEMENT_JOB=REUSED"
  echo "bundle_id=$BUNDLE_ID"
  echo "output=$OUT"
  echo "gpu_profile=$E2_GPU_PROFILE"
  echo "gpu_name=$E2_EXPECTED_GPU_NAME"
  echo "gpu_compute_capability=$E2_EXPECTED_GPU_CC"
  exit 0
fi
[[ ! -e "$RUNTIME" && ! -L "$RUNTIME" ]] || {
  echo "refusing to reuse replacement runtime: $RUNTIME" >&2
  exit 2
}

# Every scratch/compiler/cache path stays below SimAny. All model source trees
# and weights are immutable repo-local inputs validated from the frozen config.
export TMPDIR="$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg"
if [[ "$E2_GPU_PROFILE" == gcp-a100 ]]; then
  export TORCH_EXTENSIONS_DIR="$RUNTIME/gcp-a100/torch-extensions-sm80"
  export CUDA_CACHE_PATH="$RUNTIME/gcp-a100/cuda-sm80"
  export TRITON_CACHE_DIR="$RUNTIME/gcp-a100/triton-sm80"
else
  export TORCH_EXTENSIONS_DIR="$RUNTIME/torch-extensions"
  export CUDA_CACHE_PATH="$RUNTIME/cuda"
  export TRITON_CACHE_DIR="$RUNTIME/triton"
fi
export TORCH_CUDA_ARCH_LIST="$E2_TORCH_CUDA_ARCH_LIST"
export CUMM_CUDA_ARCH_LIST="$E2_CUMM_CUDA_ARCH_LIST"
export MPLCONFIGDIR="$RUNTIME/matplotlib"
export HF_HOME="$RUNTIME/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PATH="$PYTHON_ENV/bin:$PATH"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME"

# Fail before generating any artifact if the single repository-local runtime
# cannot load the main pipeline, TRELLIS sparse stack, or its own Ninja CLI.
# Module-origin checks prevent a user-site or system package from silently
# satisfying the probe even if the imports happen to succeed.
"$PYTHON" - "$PYTHON_ENV" <<'PY'
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

environment = Path(sys.argv[1]).resolve()


def require_within(module_name: str):
    module = importlib.import_module(module_name)
    value = getattr(module, "__file__", None)
    if not value:
        raise SystemExit(f"{module_name} has no import origin")
    origin = Path(value).resolve()
    try:
        origin.relative_to(environment)
    except ValueError as exc:
        raise SystemExit(
            f"{module_name} resolved outside the repository-local environment: {origin}"
        ) from exc
    return module, origin


interpreter = Path(sys.executable).resolve()
try:
    interpreter.relative_to(environment)
except ValueError as exc:
    raise SystemExit(
        f"Python resolved outside the repository-local environment: {interpreter}"
    ) from exc

observed = {}
for name in (
    "torch", "cv2", "numpy", "PIL", "trimesh", "open3d", "xformers",
    "spconv", "spconv.pytorch", "ninja", "gsplat",
):
    module, origin = require_within(name)
    observed[name] = {
        "origin": str(origin),
        "version": getattr(module, "__version__", None),
    }

expected_versions = {
    "torch": "2.5.1+cu121",
    "spconv": "2.3.8",
    "ninja": "1.13.0",
}
for name, expected in expected_versions.items():
    if observed[name]["version"] != expected:
        raise SystemExit(
            f"{name} version drift: expected {expected}, found {observed[name]['version']}"
        )

ninja_value = shutil.which("ninja")
if not ninja_value:
    raise SystemExit("repository-local Ninja CLI is unavailable")
ninja_cli = Path(ninja_value).resolve()
try:
    ninja_cli.relative_to(environment)
except ValueError as exc:
    raise SystemExit(
        f"Ninja CLI resolved outside the repository-local environment: {ninja_cli}"
    ) from exc
if not ninja_cli.is_file() or not os.access(ninja_cli, os.X_OK):
    raise SystemExit(f"Ninja CLI is not executable: {ninja_cli}")

# models.s4_trellis installs its deliberate rembg shim and pinned TRELLIS path
# before the real pipeline import. Importing the pipeline reaches its sparse
# spconv/xformers stack without loading model weights or consuming a dataset.
importlib.import_module("agents.discover.factory_prepare")
importlib.import_module("agents.assets.factory_align")
importlib.import_module("models.s4_trellis")
pipeline = importlib.import_module("trellis.pipelines")
if not hasattr(pipeline, "TrellisImageTo3DPipeline"):
    raise SystemExit("pinned TRELLIS pipeline class is unavailable")

print("replacement_main_runtime=" + json.dumps({
    "python": str(interpreter),
    "ninja_cli": str(ninja_cli),
    "versions": {name: observed[name]["version"] for name in expected_versions},
}, sort_keys=True))
PY

# SAM3 must resolve from the content-addressed source declared by the frozen E2
# config, while using the same repository-local interpreter as every other
# replacement stage. Import the actual builder and processor, not just sam3's
# lightweight package initializer.
PYTHONPATH="$SAM3_SOURCE" "$SAM3PY" - "$SAM3_SOURCE" "$SAM3_VERSION" <<'PY'
import importlib
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]
sam3 = importlib.import_module("sam3")
builder = importlib.import_module("sam3.model_builder")
processor = importlib.import_module("sam3.model.sam3_image_processor")
for name, module in (
    ("sam3", sam3),
    ("sam3.model_builder", builder),
    ("sam3.model.sam3_image_processor", processor),
):
    origin = Path(module.__file__).resolve()
    try:
        origin.relative_to(source_root)
    except ValueError as exc:
        raise SystemExit(f"{name} resolved outside the frozen source: {origin}") from exc
if getattr(sam3, "__version__", None) != expected_version:
    raise SystemExit(
        f"SAM3 version drift: expected {expected_version}, "
        f"found {getattr(sam3, '__version__', None)}"
    )
if not hasattr(builder, "build_sam3_image_model"):
    raise SystemExit("SAM3 image-model builder is unavailable")
if not hasattr(processor, "Sam3Processor"):
    raise SystemExit("SAM3 image processor is unavailable")
print("replacement_sam3_runtime=" + json.dumps({
    "module_file": str(Path(sam3.__file__).resolve()),
    "package_version": sam3.__version__,
    "python": str(Path(sys.executable).resolve()),
}, sort_keys=True))
PY

# Exercise the argparse entry points that are safe without data/model loading.
"$PYTHON" -m robo.eval.fidelity_replacements --help >/dev/null
"$PYTHON" -m agents.discover.factory_prepare --help >/dev/null
PYTHONPATH="$SAM3_SOURCE" "$SAM3PY" \
  -m agents.discover.factory_refine_masks --help >/dev/null
echo "replacement_runtime_probe=PASS"

"$PYTHON" -m robo.eval.fidelity_replacements \
  --config "$CONFIG" \
  --contract-manifest "$E2_CONTRACT_MANIFEST" \
  --bundle-id "$BUNDLE_ID" \
  --freeze-id "$E2_FREEZE_ID" \
  --out "$OUT" \
  --dataset-root "$DATASET_ROOT" \
  --python "$PYTHON" \
  --sam3-python "$SAM3PY"

# Re-open every declared output/source artifact and enforce exact file/directory
# closure.  This is also the same gate used by the resume branch above.
"${VALIDATE_COMMAND[@]}"

echo "E2_REPLACEMENT_JOB=PASS"
echo "bundle_id=$BUNDLE_ID"
echo "output=$OUT"
echo "gpu_profile=$E2_GPU_PROFILE"
echo "gpu_name=$E2_EXPECTED_GPU_NAME"
echo "gpu_compute_capability=$E2_EXPECTED_GPU_CC"
