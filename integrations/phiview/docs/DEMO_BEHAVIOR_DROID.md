# Demo: BEHAVIOR + DROID in PhysicalView

Two robot datasets served by the same studio: **BEHAVIOR** (PointWorld release of
BEHAVIOR-1K / OmniGibson simulation episodes) and **DROID** (real Franka teleop
scenes). Both were reconstructed by `agents/recon` in the SimAny checkout; this
document is about *viewing* them, not rebuilding them.

Pick the dataset with the Scene tab's **dataset** dropdown (`behavior` / `droid`),
or start the server on one directly:

```bash
PORT=8090 STUDIO_ARGS="--scene behavior_task-0020" sbatch run/studio.sbatch
PORT=8090 STUDIO_ARGS="--scene droid_iprl_sat_oct_14_20_58_47_2023" sbatch run/studio.sbatch
tail -f outputs/logs/studio_<jobid>.log     # prints node + ssh tunnel line
ssh -L 8090:<node>:8090 <login-host>        # then open http://localhost:8090
```

The viewer defaults to **server render**: the GPU rasterizes the browser camera's
view with gsplat and streams JPEG, so a laptop browser never receives the 0.5–3 M
Gaussians of these scenes.

## What is on disk

Result sets live in `$SIMANY_ROOT/outputs/<name>`, scenes and splats in
`$SIMANY_ROOT/data/recon_scenes/{data,splats}`. `objects` counts discovered
instances, `acc` the ones that survived registration into the simulator.

### BEHAVIOR — 6 tasks (sim source, 320×180 release frames upscaled ×4)

| result set | Gaussians | holdout PSNR | cams | objects/acc | scene.xml |
|---|---|---|---|---|---|
| behavior_task-0020 | 2,068,618 | 17.32 dB | 228 | 1 / 0 | yes |
| behavior_task-0045 | 1,128,463 | 15.00 dB | 18 | 0 / 0 | yes |
| behavior_task-0002 | 585,191 | 13.83 dB | 8 | 0 / 0 | yes |
| behavior_task-0027 | 544,265 | 14.13 dB | — | 0 / 0 | yes |
| behavior_task-0023 | 442,997 | 14.70 dB | — | 0 / 0 | yes |
| behavior_task-0011 | 1,325,326 | 10.24 dB | — | 0 / 0 | no |

`_mild` / `_severe` splats also exist under `recon_scenes/splats` — those are the
degradation ablations, not viewer entries (no result-set directory).

### DROID — 9 scenes (real robot, 30 k gsplat iterations)

| result set | Gaussians | holdout PSNR | cams | objects/acc | mesh |
|---|---|---|---|---|---|
| droid_iprl_sat_oct_14_20_58_47_2023 | 1,118,332 | 21.74 dB | 119 | 6 / 5 | yes |
| droid_tri_thu_sep_21_17_26_41_2023 | 2,192,781 | 21.08 dB | — | 0 / 0 | yes |
| droid_pennpal_sat_apr_29_20_28_46_2023 | 551,662 | 20.97 dB | — | 0 / 0 | yes |
| droid_rpl_mon_jun__5_16_11_49_2023 | 625,450 | 20.16 dB | — | 0 / 0 | yes |
| droid_iprl_thu_aug_24_21_29_53_2023 | 1,112,286 | 20.03 dB | — | 0 / 0 | yes |
| droid_rail_tue_oct_24_10_23_40_2023 | 667,270 | 18.70 dB | 100 | 1 / 1 | yes |
| droid_iris_mon_apr_17_16_03_25_2023 | 2,893,808 | 18.38 dB | 240 | 10 / 4 | yes |
| droid_real_wed_jun__7_11_50_12_2023 | 3,005,965 | 17.90 dB | — | 0 / 0 | yes |
| droid_autolab_fri_jul_14_16_55_45_2023 | 2,016,150 | 16.13 dB | — | 0 / 0 | yes |

`droid_clvr_tue_may__9_02_08_20_2023` has no build and no splat; it is not listed
by the viewer.

## What to expect in the viewer

* **BEHAVIOR** reads best. The rooms are full kitchens/living rooms and the
  BEHAVIOR robot is baked into the Gaussians (it is part of the release frames).
  Object yield is near zero, and that is a known data limit rather than a bug:
  SAM3 discovery on 320×180 release frames finds almost nothing, so the Generate
  tab's object table is empty for five of the six tasks. Use these scenes to show reconstruction and the exported
  room, not per-object assets.
* **DROID** carries the object pipeline. `droid_iprl_sat` (5 accepted) and
  `droid_iris` (4 accepted) have registered assets, collision parts and a
  `sim_export/scene.xml`. Its *data* cameras are wrist-mounted close-ups — for an
  overview, orbit the free camera rather than snapping to a camera frame.

## Fly near the capture trajectory, not outside it

Both datasets are captured by cameras looking *inward* from close range — a robot
head/wrist rig, not a room scan — so nothing supervises the Gaussians from
outside. Measured on an A6000 across three camera strategies (jobs 876038 /
876057 / 876067):

| viewpoint | result |
|---|---|
| a data camera, its own K | clean — see `droid_rail` frame 0, sharp down to breadboard holes |
| backed off 0.45 × scene radius along that camera's axis, 72° FOV | centre holds, periphery shatters |
| orbit at 2× scene extent | shards and spikes everywhere, BEHAVIOR rooms become a doll's-house cutaway |

So the usable viewing volume is roughly *the capture trajectory itself*. In the
viewer, snap to a camera frame and make small moves from there; zooming out to
frame the whole cloud will only ever show you floaters. This is a property of the
captures, not of the viewer — and it is the main thing to warn a first-time
visitor about.

## Caveats worth stating out loud

* No `pi05_tasks.json` exists for either dataset, so the Robot tab has no
  authored task to pick; policy rollouts need a task generated first (Generate
  tab → task generation).
* PSNR is holdout PSNR from `<scene>_train_report.json`, measured against the
  same frames that trained the splat — it is a reconstruction-quality number, not
  a novel-view benchmark.
* BEHAVIOR `behavior_task-0011` has no `sim_export/scene.xml`; it loads as a
  splat only.

## Scene-id gotcha (fixed)

`run/run_behavior_recon.sh` writes `outputs/behavior_task-0020` but builds the
scene as `behavior_task0020` (`SCENE_NAME=behavior_${TASK//-/}`). The viewer's
`classify_result_dir` now strips the dash, so scene dir, splat and `SIMANY_SCENE`
all resolve. Before that fix every BEHAVIOR entry showed **no splat**.
