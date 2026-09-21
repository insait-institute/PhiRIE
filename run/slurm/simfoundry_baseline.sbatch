#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:50:00
#SBATCH --job-name=simany_sfrepro
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/simfoundry_repro_%j.log
# SimFoundry-repro baseline (Task 15 / baselines/simfoundry_repro.py) run
# against an arbitrary scene dir, for the head-to-head vs SimAny comparison
# in the oracle_causal / mujoco_paired predictive tracks.
#
# Usage: sbatch --export=ALL,SCENE_DIR=<path>,OUT_DIR=<path> run/slurm/simfoundry_baseline.sbatch
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
cd "$SIMANY_ROOT"
[ -n "$SCENE_DIR" ] || { echo "set SCENE_DIR=<path to scene dir>"; exit 1; }
[ -n "$OUT_DIR" ] || { echo "set OUT_DIR=<path for baseline output>"; exit 1; }
# hala's node-local /scratch (~/.cache's real target) evicts things over
# time, independent of which node a job lands on -- verified 2026-08-30:
# TRELLIS-image-large and DA3METRIC-LARGE both disappeared from it between
# runs despite being cached there just hours earlier. run/env.sh's
# empty-dir-triggered fallback doesn't catch this (the dir isn't empty,
# just missing specific models), so force the stable shared cache
# explicitly here, matching run/slurm/oracle_matrix.sbatch's precedent --
# every model this specific stage chain needs (SAM3, DA3METRIC-LARGE,
# TRELLIS-image-large+dinov2, Qwen2.5-7B-Instruct) is confirmed present
# there as of this date.
export HF_HOME=/group/worldcept/hf_cache
export TORCH_HOME=/group/worldcept/torch_hub_cache
source run/env.sh
.venv/bin/python -m baselines.simfoundry_repro --scene-dir "$SCENE_DIR" --out-dir "$OUT_DIR"
echo "SIMFOUNDRY_REPRO_DONE $SCENE_DIR -> $OUT_DIR"
