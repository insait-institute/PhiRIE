# PhiView demo validation — 2026-09-13

Implementation commit: `3c55cb97c90ca650096fc709e68c952b423a7f4d`.
Branch: `feature/phiview-demo`. The running source and HTML hashes were independently
compared with the manifest and matched. Existing `.claude/` files were preserved.

Live session: job `893677`, step `893677.7`, `sof1-h200-3`, one NVIDIA H200, port `8097`.
The step is scheduled until **2026-09-14 00:34:42 UTC**, subject to scheduler termination.

```bash
ssh -N -L 8097:sof1-h200-3:8097 runyi_yang@hala.slurm.insait.ai
# Open http://localhost:8097
```

Scene `c50d2d1d42_factory` loads all **1,499,998 SH-3 Gaussians** from the original
372,001,036-byte PLY. SHA256:
`7830f4ac1cc646d7cfa67d5cdf6f7d5d769114bbe826852383f7e3e9ec819253`.
There are 18 proposals and 16 constructed rigid bodies. Original rendering was pixel-exact
against a direct full-Gaussian render; native resolution 1752×1168 also passed.

- CPU suite: **126 passed, 13 skipped**, 9.55 s. GPU-dependent skips were not counted as passes.
- Final H200 checks: **9/9 passed**. The robot check requires actual joint motion and visible
  composited pixels; it measured 111,693 visible robot pixels and does not assert task success.
- Chromium: real image click selected `obj_13`; WASD/right-drag, enable/fall/pause/reset and
  clean-selected controls passed. No JavaScript exceptions, no canvas or 3D asset transfers.
- Fresh SAM3 check: on DSC01593.JPG, five prompts returned 9 bottles, 1 keyboard, 1 mouse,
  2 mugs, 1 headphones detection. This is a single-view inference check, not exhaustive
  full-room discovery or an instance-recall measurement.
- Fresh prompted Qwen 2511 edit: three keyboard views edited, `zero_cond_t` supported by
  Diffusers 0.37.0, no cached edit or LaMa fallback accepted; 300 Gaussian fitting iterations
  completed. The new 3D scene loaded and its comparison changed 28,285 pixels by >10 levels.
  That count verifies a changed image, not an inpainting quality metric.

Final evidence directory:
`outputs/phiview-h200-893677-release-v2/` — `manifest.json`, `handoff.json`,
`handoff-health.json`, `cpu-tests.txt`, `validation/checks.json`, `validation/throw.mp4`,
`validation/robot-reach.mp4`, and `browser/browser-check.json` plus screenshots.

Prompt provenance and images remain in
`outputs/phiview-h200-893677-final/builds/1789334496357158750/inpaint/`:
`model-receipt.json`, `prompt-receipt.json`, `edit_meta.json`, edited PNGs, and
`clean_background.ply`. The release restores this exact saved version through its
`active-inpaint.json`. Before/after renders and the comparison are in the preceding run's
`prompt-validation/`. Fresh SAM3 evidence is in its `sam3/sam3-check.json`.

Known limits: inherited clean-all imagery has transparent-object remnants; inherited
collision uses support slabs rather than a verified complete room; two proposals lack
constructed bodies; robot commands use the documented scripted vocabulary. Grasp/placement
success and a general language policy are not established. See [controls and scope](PHIVIEW.md).
The service was left in Original view, simulation paused, with the test arm removed.

Earlier attempts are preserved: job 893627 exposed the projectile broadphase bug; the first
Qwen edit used an incompatible Diffusers version and was stopped and marked failed. Draft
step 893677.0 and superseded release step 893677.6 were stopped after their evidence was saved.
