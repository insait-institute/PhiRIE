#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:50:00
#SBATCH --job-name=simany_droid_recon
#SBATCH --output=outputs/droid_recon_%j.log
# DROID raw episode -> reconstructed + FK-aligned scene, end to end.
#   sbatch --export=ALL,EPISODE=IPRL/success/2023-08-24/Thu_Aug_24_21:29:53_2023 \
#       run/slurm/droid_recon.sbatch
# Bash shebang is load-bearing: an earlier pilot (job 703918) was launched
# via `sbatch --wrap="source run/env.sh; ..."`, which sbatch executes under
# /bin/sh (dash on this cluster) - dash has no `source` builtin (needs `.`),
# so it died immediately with "slurm_script: 4: source: not found" before
# the real pipeline ever ran. run_droid_recon.sh itself sources run/env.sh
# and uses bash-only syntax, so it must always be invoked as a .sbatch file
# (this one) or via `bash run/run_droid_recon.sh`, never --wrap or `sh`.
# For --partition=batch longer/unattended runs, add --exclude=msp3-[0-7]
# when requesting h200 GPUs (msp3 nodes do not reliably mount
# ${PHIRIE_WORKSPACE:-$HOME} - see docs/DROID_PROTOCOL.md).
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
[ -n "$EPISODE" ] || { echo "set EPISODE=<lab>/success/<date>/<ts>"; exit 1; }
export EPISODE
bash "$SIMANY_ROOT/run/run_droid_recon.sh"
echo "DROID_RECON_DONE $EPISODE"
