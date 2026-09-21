# Changelog

## Unreleased

- Click an unlabelled scene object to segment it with a SAM3 point prompt, highlight its
  visible Gaussians, and create an approximate rigid collision body with Make simulatable.
  New selections and collision proxies persist within the viewer session directory.
  Default physical estimates are explicitly unmeasured; existing bodies retain their poses.

- Allow shooting and object simulation before background inpainting. Original observed
  background remains available; unfilled regions are labeled and inpainted views remain gated.

- Make visible objects red with a strong filled mask and a white contour; selected
  objects use a stronger fill and thicker outline. Picking labels remain unchanged.

## 0.2.0 — 2026-09-14

- Integrate native reconstruction, PhiView demo/paper work and BEHAVIOR/DROID studio
  selection while preserving every original branch head and merge history.
- Add a unified CLI with ten discoverable blocks, inspectable plans, sequential local/Slurm
  execution and retained job receipts.
- Add portable config, pinned backend sources and four independent uv lockfiles for CPU
  development, studio, inference and generation.
- Package browser assets, default config, block manifests and the offline gallery/ZIP exporter.
- Document module ownership, calls, contribution flow and release validation; add CPU CI,
  package checks and draft GitHub release automation.

The new H200 studio environment renders all 1,499,998 Gaussians in the checked scene and
passes CUDA/MuJoCo integration checks. This software integration release does not certify
general manipulation or publication quality of the paper campaign.
