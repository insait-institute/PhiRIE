# Skill: launching every pipeline mode

The exact invocations that work, and the traps around each. All commands run
from the repo root; `run/env.sh` provides `run`/`run_sam3`/`run_gs`/`run_qwen`
(three python envs — see docs/ENVIRONMENTS.md for why they cannot merge).

## Generation pipeline

```bash
SIMANY_SCENE=<id> bash run/run_factory.sh   # GT-driven benchmark mode
SIMANY_SCENE=<id> bash run/run_auto.sh      # fully automatic (SIMANY_AUTO=1)
SIMANY_SCENE=<id> bash run/run_simfoundry.sh  # SimFoundry-repro baseline (row D)
SIMANY_SCENE=<id> bash run/run_inpaint.sh   # removal + background fill
```

- `RESUME=1` makes stages skip themselves when outputs exist.
- Fleets: `run/slurm/fleet_val.sbatch` (array over the 50-scene val split).
- Hybrid V_mesh second pass: `run/slurm/fleet_hybrid.sbatch`. If rerunning
  after a fix, delete stale `obj_XX/trellis/` snapshot dirs first — the
  `.snapshot_done` marker otherwise serves pre-fix alignment verbatim.
- Qwen inpainting: needs the sam3 env (torch ≥ 2.5) and
  `SIMANY_QWEN=force` on A6000 (sequential CPU offload, ~8 min/view);
  per-view `inpainted_k.png` cache makes it resumable.

## Robot closed loop (pi0.5)

```bash
# policy server (A6000): run/pi05_serve.sh — checkpoint picked by
# SIMANY_PI05_CKPT (default pi05_droid_jointpos; sim-co-trained variant
# needs SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim)
sbatch run/slurm/pi05_closedloop.sbatch      # server + eval on one GPU
```

- Composite (photoreal) observations require the scene to have
  `inpaint/clean_background.ply` — only 6 of 50 scenes do (27dd4da69e,
  45b0dac5e3, 578511c8a9, 7b6477cb95, 825d228aec, c50d2d1d42).
- New scenes need `robo.sim.export_mjcf --test` first (also the only source
  of `mujoco_settle.json`, without which excluded-object logic is silently
  empty), then `robo.tasks.pi05_tasks`.
- Action contract: model predicts joint deltas; the server returns absolute
  7-dof jointpos + gripper [0,1], chunks of (15, 8) executed at 15 Hz.
- One scene per process in composite mode (common.py binds the scene at
  import).
- Pass `--time-limit 32` explicitly; the suite default (16 s) is too short.

## Demo rendering (never for benchmark numbers)

`--ext-cam-frame`, `--demo-declutter`, `--render-wh` change what the policy
sees. Traps, all observed on 08-03:

1. The exterior camera materially changes the score (same task: DSC01594
   0.50, DSC01593 0.25, DSC01616 0.00) — pick camera and report together.
2. Declutter exposes inpaint scars other objects were hiding; keep clutter.
3. Judge candidate cameras by the target's margin to the frame edge, not
   its apparent size in a static home-pose frame.

## Viewer

```bash
python -m interface.viewer --outputs-root ./outputs --port 8090
```

Prefer VS Code port-forward of :8090 over the flaky viser share relay; the
browser renderer is SH0-only, so metric renders (gsplat CUDA) look better
than the viewer does.
