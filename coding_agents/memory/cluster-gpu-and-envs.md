---
name: cluster-gpu-and-envs
description: How to get GPUs on this SLURM cluster and which python envs have what
metadata: 
  node_type: memory
  type: project
  originSessionId: a84b4517-f014-4a7b-99cb-664d54b4d178
  modified: 2026-07-22T19:51:30.518Z
---

Login node has no GPU access (nvidia-smi empty). Use SLURM:
`srun --partition=debug --gpus=a6000:1 --mem=48G --time=HH:MM:SS ...` — debug partition
(node hala, A6000 48GB) allocates near-instantly, 4h cap. `--gres=gpu:1` fails; the GPU
type must be named. Batch partition (H200/A100 nodes) is usually congested.

Python envs:
- `/group/worldcept/code/affordancept/.envs/mini-viewer` — torch 2.4.1+cu124, gsplat 1.5.3,
  transformers 4.57.6, accelerate, sklearn, scipy, imageio, plyfile (built from Mini_Viewer env.yml).
- `/group/streetsplat/worldcept/.envs/sam3` — torch 2.10+cu128 + editable SAM3 from
  `/group/streetsplat/worldcept/code/third_party/sam3`. SAM3 inference REQUIRES
  `torch.autocast("cuda", bfloat16)` or it crashes with a bf16/fp32 dtype mismatch.
  Checkpoint cached: `~/.cache/huggingface/hub/models--facebook--sam3` (use HF_HUB_OFFLINE=1).
- Also cached in HF hub: Qwen2.5-7B-Instruct, SigLIP2 so400m-patch16-512.
- Conda base: /group/streetsplat/miniconda3 (shared; don't modify shared envs).

## 2026-07-20 correction: H200 (sof1-h200-*, partition `no-preempt`/`batch`, qos=`normal`) DOES work
Earlier note "H200/batch nodes don't mount /group/worldcept" was WRONG or has
since been fixed - verified directly:
- `/group/worldcept` mounts fine (ceph fs, same as hala).
- `/data/ScanNetpp` and `/data/ScanNetppv2_gsplat` ARE mounted via autofs
  (lazy/on-demand) - a `ls` on a DEEP path before the parent dir has been
  touched can show an empty listing; `ls /data/ScanNetpp` (one level up)
  first triggers the automount reliably. Don't conclude "not mounted" from
  one empty deep-path listing alone.
- gsplat (mini-viewer env wheel, same one built for A6000/sm_86) runs
  natively on H200 (sm_90/Hopper, compute capability 9.0) with NO
  recompilation - `render_view` produces correct output.
- QoS for this account on H200/batch/non-reserved partitions is `normal`
  (separate bucket from the hala `debug` QoS used for the main 4-concurrent
  fleet work) - `sbatch --partition=no-preempt --qos=normal --gpus=h200:1 ...`
  or `--partition=batch --qos=normal --gpus=h200:1` (no --nodelist pin;
  letting SLURM pick avoids "ReqNodeNotAvail" on a specific reserved node).
  hala's long-lived `rendering` partition (infinite walltime, for viewer-style
  always-on jobs) needs qos `rendering` - this account is NOT authorized for
  that QoS (`sacctmgr show assoc` only lists debug/eccv-2026/neurips-2026/
  normal) - use `--partition=batch --qos=normal --nodelist=hala` instead for
  a long-running a6000 job pinned to hala (gsplat only reliably JIT-builds
  there per the existing hala-specific note below).
- H200/no-preempt/normal-QoS is a SEPARATE concurrency pool from hala's
  `debug` QoS - jobs there don't compete with the main a6000 debug-partition
  fleet quota, so it's genuinely useful as an additional parallel pool, not
  just a faster GPU.

## 2026-07-20: gsplat JIT CUDA build gotchas (any fresh gsplat install, not just one project)
- Default shell gcc (login and compute) is 14.2.0 - too new for nvcc
  12.4/12.8's host-compiler check ("gcc versions later than 13 are not
  supported!"). Fix inside the srun job: `module load gcc-13.4.0` +
  `module load nvidia-cuda-12.8.1` (match the CUDA version torch was built
  against - 12.8.1 for torch 2.8.0's bundled cu128).
- Check for a stray inherited `TORCH_CUDA_ARCH_LIST` env var before any gsplat
  JIT build - one shell had it set to `9.0+PTX` (leftover from H200/sm_90
  work), which silently makes the build target the wrong arch on an A6000.
  Override explicitly: `export TORCH_CUDA_ARCH_LIST="8.6"` for A6000 (sm_86).
- First build after fixing both takes ~100s and caches to
  `~/.cache/torch_extensions/py<ver>_cu<ver>/gsplat_cuda/` (shared home ->
  one-time cost across nodes, not per-node). See [[sharp-single-image-feedforward]]
  for a worked example (frame30 cold build 2m10s, frame60 warm rerun 20s).

## 2026-07-22 CRITICAL scheduling gotcha: two h200 pools, one is broken for us
`--partition=batch --gpus=h200:1` can land on EITHER sof1-h200-[0-7] (good) or
msp3-[0-7] (does NOT mount /group/worldcept -> job dies exit 127 with NO log
file, since even the sbatch --output path is unwritable there; sacct NodeList
is the only clue). ALWAYS add `#SBATCH --exclude=msp3-[0-7]` for h200 jobs.
Also guard module loads: sbatch scripts with `set -e` die on
`eval "$(module bash-hook)"` (trailing completion check returns 1 in
non-interactive shells) - use `|| true` / `command -v module` guards.

## 2026-07-22 user-confirmed usable node pools
User: "sof1-h200-[0-7] can work. gcp-eu-* can work". sinfo shows: sof1-h200-[0-7]
= 8x H200 each (partitions batch/non-reserved/no-preempt; sof1-h200-[0-1] also
in `login`); gcp-eu1-a100-40g-px41 (8x A100-40G), gcp-eu1-a100-80g-{1tsm,qzcg}
(8x A100-80G each), gcp-eu1-rtx6000-3089 (8x RTX6000). Current interactive
login lands on `hala` (no GPU). For anything HF-cached, set HF_HOME to a
/group/worldcept path — ~/.cache is a node-local /scratch symlink (see below).

## 2026-07-22 pi05 session additions
- `uv` now installed at ~/.local/bin/uv (0.11.31). openpi (xuningy fork) venv at
  /group/worldcept/code/openpi/.venv (7.6GB, JAX cuda12). A first `uv sync` hung
  >100min with no children (CephFS lock?) — kill + rerun finished in minutes off
  the warm cache.
- SimFoundry venv gsplat JIT (py311_cu124 cache dir): default gcc-14 headers are
  REJECTED by nvcc 12.4 but ACCEPTED by CUDA 12.8.1: export
  CUDA_HOME=/opt/modules/nvidia-cuda-12.8.1 + NVCC_APPEND_FLAGS=-allow-unsupported-compiler.
  If no GPU is visible at build time gsplat defaults the arch to sm_90 ->
  "no kernel image"/"Failed to set maximum shared memory" on A6000; always
  export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX" (and keep exporting it at RUNTIME,
  or torch regenerates build.ninja and rebuilds). Stale `lock` file in
  ~/.cache/torch_extensions/.../gsplat_cuda after a killed build makes later
  builds hang silently — rm it. Compile needs NO GPU (login node ok, ~90s MAX_JOBS=32).
- sbatch on debug: default mem is small — pi05 eval needed --mem=100G (mujoco EGL
  + gsplat + splats). EGL in sbatch jobs: enumerate-all-GPUs issue — set
  MUJOCO_EGL_DEVICE_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader|head -1).
- msp3 mount status is FLAKY/recently changed: a 2026-07-22 probe (srun -w msp3-4)
  showed /group/worldcept mounted + internet OK, while a same-day job on msp3 died
  exit 127 unwritable. Spot-check the specific node before relying on it; keep
  `--exclude=msp3-[0-7]` as the safe default for h200 jobs.

## 2026-07-21 H200 usability boundary (important correction to the above)
`~/.cache` on hala is a SYMLINK to `/scratch/runyi_yang/.cache` (108GB,
contains SAM3/DINOv2/TRELLIS-adjacent HF+torch-hub caches) - `/scratch` is
NODE-LOCAL, not shared cluster storage. So:
- H200 (or any non-hala node) CAN run GPU-compute stages that only touch
  /group + /data: verified working end-to-end for derive_mesh_from_splat.py
  render+fuse (gsplat render + open3d TSDF), including fixing an OOM that
  hit on hala's more limited memory for a large scene (ac48a9b736).
- H200 CANNOT run stages that load a HF/torch-hub-cached checkpoint under
  `~/.cache` (SAM3 in auto_segment.py/factory_refine_masks.py; DINOv2 inside
  TRELLIS's image conditioning) - they FileNotFoundError immediately, since
  that path resolves to hala's local scratch which doesn't exist elsewhere.
- Fix (not yet done): copy just the specific needed checkpoints (SAM3 .pt,
  DINOv2 weights TRELLIS needs) to a /group/worldcept-hosted cache dir and
  point HF_HOME/TORCH_HOME/the hardcoded CKPT constant there when running
  off-hala. Do NOT try to mirror the whole 108GB scratch cache.
