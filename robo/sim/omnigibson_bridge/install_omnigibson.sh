#!/bin/bash
# Install OmniGibson / BEHAVIOR-1K into the `behavior1k` conda env and run the
# headless smoke test. Needs an NVIDIA GPU, ~64 GB host RAM, and a Python 3.11
# conda env named `behavior1k` created beforehand (CONDA_ROOT overrides the
# default ~/miniconda3 install).
#
#   bash robo/sim/omnigibson_bridge/install_omnigibson.sh
#
# Start from a minimal known-good PATH: a stray venv prepended to PATH once
# made every bare `python`/`pip` call below silently hit the WRONG environment
# (packages were pip-installed into an unrelated project's venv). Resetting
# PATH keeps this script self-contained regardless of the calling shell.
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

SIMANY_ROOT=${SIMANY_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
REPO=$SIMANY_ROOT/third_party/BEHAVIOR-1K
cd "$REPO"

CONDA_BASE=${CONDA_ROOT:-$HOME/miniconda3}
set +u  # conda's shell hooks reference unset variables
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k
set -u
echo "[install] active env: $CONDA_PREFIX"
[[ "$CONDA_PREFIX" == *"/behavior1k" ]] || { echo "FATAL: wrong env active"; exit 1; }
python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print('python OK', sys.version)"

set +e
bash setup.sh --omnigibson --bddl --dataset --confirm-no-conda \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
INSTALL_RC=$?
set -e
if [ "$INSTALL_RC" -ne 0 ]; then
    echo "FATAL: setup.sh failed with exit code $INSTALL_RC"
    exit "$INSTALL_RC"
fi

nvidia-smi --query-gpu=name,driver_version --format=csv
which python pip
python -c "import sys; print('python', sys.version)"

echo "=== SMOKE TEST ==="
set +e
OMNIGIBSON_HEADLESS=1 python "$SIMANY_ROOT/robo/sim/omnigibson_bridge/smoke_test.py"
SMOKE_RC=$?
set -e
echo "=== SMOKE TEST EXIT CODE: $SMOKE_RC ==="
exit "$SMOKE_RC"
