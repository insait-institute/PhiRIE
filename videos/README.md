# videos/ — drop a phone video, get a simulatable scene

Any `*.mp4` dropped here can be turned into a sim-ready scene with **no
dataset, no GT, no poses**: video → frame extraction → camera poses (COLMAP,
with a feed-forward VGGT fallback) → metric scale + z-up → emulated
ScanNet++-style scene dir → 3DGS training → the automatic pipeline (derived
mesh, SAM3 discovery, TRELLIS assets, physics, MJCF + task suite).

## What to record

- Casual RGB video, **30–120 s**, phone is fine (mp4, any resolution;
  frames are capped to 1168 px wide).
- **Slow, smooth sweep** through the room — walk an arc or two around the
  area of interest, keep objects in view from several directions. Fast pans
  and motion blur break both pose estimation and splat training.
- **Mostly static scene** — no people/pets walking through, nothing moving.
- **Good texture and lighting** — avoid large blank walls as the only
  content, avoid mirrors and big glass/reflective surfaces (they corrupt
  geometry).
- **Show the floor** — metric up-axis alignment RANSACs the floor plane;
  a clip that never sees the floor will abort at the `metricize` stage.

[capture/README.md](../capture/README.md) describes a two-pass filming
procedure and a validator that scores blur, motion and coverage before the
GPU stages run.

## How to launch

On a GPU machine with the three environments set up
([docs/ENVIRONMENTS.md](../docs/ENVIRONMENTS.md)):

```bash
cd /path/to/PhiRIE

# one clip:
bash run/run_video2sim.sh --video videos/myroom.mp4

# every videos/*.mp4, sequentially:
bash run/run_video2sim.sh
```

Knobs: `RESUME=1` re-runs skip stages whose outputs already exist;
`POSE_BACKEND=colmap|omega|vggt` (default `colmap`, with feed-forward
fallback when too few frames register); `MAX_FRAMES` (default 240);
`GS_ITERS` (default 30000); `SIMANY_OUT` overrides the output directory.
Progress is printed to the terminal, and per-stage wall times go to
`outputs/video_<name>/timings.txt`.

## Where results land

Scene name = sanitized video basename (`my room.mp4` → `my_room`).

- `data/recon_scenes/data/<name>/dslr/…` — emulated ScanNet++-style scene
  dir (frames, intrinsics, COLMAP poses) that the whole pipeline reads.
- `data/recon_scenes/splats/<name>.ply` — trained 3DGS splat
  (+ `train_report.json` with holdout PSNR next to it).
- `outputs/video_<name>/`
  - `recon/` — extracted frames, `recon.npz`, `recon_metric.npz`
  - `derived_mesh.ply` — TSDF mesh fused from the splat (no scan needed)
  - `objects/obj_XX/` — per-object TRELLIS mesh, `object.urdf`, physics
  - `report.json`, `crops_sheet.png` — yield + drop-test QA
  - `sim_export/scene.xml` — MuJoCo scene (settle-tested) + Isaac manifest
  - `sim_export/pi05_tasks.json` — pick-and-place task suite for pi0.5

## Limits

- **Rigid objects only** — articulated/deformable things become static or
  single rigid bodies.
- **Metric scale from monodepth** when the feed-forward backend is used,
  roughly ±10 %; don't trust absolute sizes for tight-tolerance manipulation.
- **z-up assumes a visible floor**; scenes without one abort rather than
  guess (rerecord with the floor in frame).
- Objects outside the discovery vocabulary (`agents/core/common.py`
  `VOCAB`) are left in the background shell.
- GT-based evaluation stages (`factory_eval_render`, `eval_vs_gt`) do not
  run — there is no GT for your living room.

Git: `*.mp4` is ignored repo-wide (see `.gitignore`), so clips dropped here
are never committed; this README and `.gitkeep` are tracked.
