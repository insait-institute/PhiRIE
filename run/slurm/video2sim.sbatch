#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --time=03:50:00
#SBATCH --job-name=video2sim
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/video2sim_%j.log
# Usage:
#   sbatch --export=ALL,VIDEO=/abs/path/clip.mp4 run/slurm/video2sim.sbatch
#   sbatch run/slurm/video2sim.sbatch     # all of videos/*.mp4, sequentially
# One job, one node, start to finish: a derived_mesh.ply overwritten across
# separate submissions hit stale cross-node NFS reads once (see
# ablation_rowC_scene.sbatch) - same reasoning applies to the trained splat.
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
# env.sh pins TORCH_CUDA_ARCH_LIST/CXX so gsplat's JIT cache is reused instead
# of rebuilt; run_video2sim.sh re-sources it after exporting the recon-scenes
# roots, which is safe (its exports all honour pre-set values).
source "$SIMANY_ROOT/run/env.sh"
# export_mjcf --test only settles physics, but any MuJoCo render path that
# does trigger must not look for a display on a compute node.
export MUJOCO_GL=egl

bash "$SIMANY_ROOT/run/run_video2sim.sh" ${VIDEO:+--video "$VIDEO"}
