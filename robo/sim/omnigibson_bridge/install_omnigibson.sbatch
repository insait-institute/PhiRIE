#!/bin/bash
#SBATCH --job-name=simany_og_install
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=hala
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/logs/og_install_%j.log

# --export=NONE + this hard reset is load-bearing: the interactive session
# that submits this job carries a stray venv (.venv-pointworld-data)
# prepended to PATH, which previously caused every bare `python`/`pip` call
# below to silently hit the WRONG environment (see incident writeup in
# omnigibson_bridge/ - accidentally pip-installed torch/bddl into an
# unrelated project's venv). Starting from a minimal known-good PATH makes
# this job's environment fully self-contained regardless of what the
# submitting shell had polluted.
unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
export HOME=/home/runyi_yang

set -ex
REPO=/group/worldcept/PhiRIE/code/SimAny/third_party/BEHAVIOR-1K
cd "$REPO"

CONDA_BASE=/group/streetsplat/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k
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
OMNIGIBSON_HEADLESS=1 python /group/worldcept/PhiRIE/code/SimAny/robo/sim/omnigibson_bridge/smoke_test.py
SMOKE_RC=$?
set -e
echo "=== SMOKE TEST EXIT CODE: $SMOKE_RC ==="
exit "$SMOKE_RC"
