---
name: sharp-single-image-feedforward
description: "Apple SHARP (single-image feedforward 3DGS) installed at /group/worldcept/sharp — working recipe, verified outputs"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5a519c9c-9352-4e87-a599-b17286d72384
  modified: 2026-07-22T14:49:29.618Z
---

Apple's SHARP (github.com/apple/ml-sharp, arXiv:2512.10685): single photo -> 3D
Gaussian splat params via one feedforward pass, metric scale, <1s pure inference,
gsplat-based renderer, OpenCV camera convention. Suggested by user as the
single-image option to try, as opposed to the multi-view pipelines in
[[simfoundry-repro-status]] / [[affordancept-pipeline]] which all need
images+poses+mesh. Installed and smoke-tested 2026-07-20 on two real frames
(ggpt/frame30.png, frame60.png) - both produced valid ~63MB binary PLYs plus
--render mp4/depth.mp4 trajectory videos. Not yet wired into any downstream
pipeline - this is just a working, verified base install.

Location: `/group/worldcept/sharp/` - repo at `repo/`, env at `.venv-sharp/`,
checkpoint at `sharp_2572gikvuh.pt` (2.8GB, from
ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt), test I/O under
`inputs/<name>/` and `outputs/<name>/`.

Results viewer artifact (self-contained, inputs+renders embedded as base64):
https://claude.ai/code/artifact/21621163-e780-4af0-9c1a-a0f08ba56e70 - built from
/tmp scratch via a small python b64-embed script (source template kept only in
that session's scratchpad, not persisted here). Update by rebuilding & re-publishing
to the same file_path from a session that has it, or start fresh from the
per-frame outputs listed below.

2026-07-22: user asked to pick a good desk-facing frame with many operable
objects (not just whatever was lying around) - found by cross-referencing which
raw frames [[affordancept-pipeline]]'s "turn_on_the_monitor" and
"write_on_the_paper" tasks already used on scene 09bced689e
(outputs/09bced689e/tasks/<task>/step3_masks/*_overlay.jpg names the frame IDs),
then visually inspected 2-3 candidates via Read (image render) to pick composition.
Landed on `DSC08561.JPG` (from
/data/ScanNetpp/data/09bced689e/dslr/resized_undistorted_images/) - straight-on
desk shot: 2 monitors, keyboard, mouse, pens, mug, loose paper, phone dock,
bottle, chair+door+whiteboard in frame. Ran in 6.9s predict pipeline (gsplat
build already warm from the earlier frame30/60 smoke test) -> outputs/DSC08561/
{DSC08561.ply (63MB), DSC08561.mp4, DSC08561.depth.mp4}. This is a reusable
pattern: ScanNet++ scenes with affordancept task outputs are a good place to
find "rich desk/object" single frames for future SHARP (or other single-image)
demos, since someone already curated which frames show which objects.

Recipe:
```
# login node (has network) - env + install + checkpoint
/group/streetsplat/miniconda3/bin/conda create -p /group/worldcept/sharp/.venv-sharp python=3.13 -y
cd /group/worldcept/sharp/repo && /group/worldcept/sharp/.venv-sharp/bin/pip install -r requirements.txt
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -O /group/worldcept/sharp/sharp_2572gikvuh.pt

# GPU node (debug/a6000, see [[cluster-gpu-and-envs]]) - inference
srun --partition=debug --gpus=a6000:1 --mem=48G --time=00:20:00 bash -c '
  eval "$(module bash-hook)"; module reset
  module load gcc-13.4.0
  module load nvidia-cuda-12.8.1
  export TORCH_CUDA_ARCH_LIST="8.6"
  /group/worldcept/sharp/.venv-sharp/bin/sharp predict \
    -i /group/worldcept/sharp/inputs/<name> \
    -o /group/worldcept/sharp/outputs/<name> \
    -c /group/worldcept/sharp/sharp_2572gikvuh.pt --render -v
'
```

Gotcha (generalizes beyond SHARP to any gsplat JIT build on this cluster - added
to [[cluster-gpu-and-envs]] too): default login/compute shell gcc is 14.2.0,
too new for nvcc 12.4/12.8's host-compiler check ("gcc versions later than 13
are not supported!") - must `module load gcc-13.4.0` + a matching
`nvidia-cuda-12.8.1` module inside the srun job. Also check for a stray
inherited `TORCH_CUDA_ARCH_LIST` env var (was set to `9.0+PTX` in this shell,
presumably left over from other H200/sm_90 work) - silently makes gsplat JIT
build for the wrong arch; override explicitly to `8.6` for A6000. First build
after fixing both takes ~106s and caches to
`~/.cache/torch_extensions/py313_cu128/gsplat_cuda/` (shared home, so it's a
one-time cost across nodes) - subsequent runs (confirmed via frame60) drop to
~20s wall clock including SLURM overhead, with pure model inference at
1.4-1.9s, matching the paper's <1s claim once the build is warm.
