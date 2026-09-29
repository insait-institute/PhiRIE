# SimAny Studio — architecture

Interactive, GPU-backed web UI (viser) that drives the **full SimAny pipeline** on one
scene: load a room, discover objects, generate 3D assets with a chosen model, register
them with a chosen registration mode, run physics annotation, inpaint a chosen region
with a text prompt, export the simulator, and drive a Franka arm with a chosen policy on
a chosen task while watching synchronized MuJoCo and photoreal views.

Entry point: `python -m physicalview.app --port 8080` (see `run/launch.sh`).

## Why viser (not Isaac Sim / Newton)

* The cluster has no display; Isaac Sim's RTX renderer segfaults on driver 595.x
  (docs/POLARIS_INTEGRATION.md). Newton is a physics engine, not a UI.
* viser serves a browser client from any GPU node. By default the 3D view is a **server
  render**: gsplat renders the browser camera's view on the GPU and the frame is streamed
  as a JPEG background image (see "Display modes"); the browser holds no splat data. A
  client-side WebGL splat mode remains as a fallback. CUDA is used where it matters:
  gsplat renders (viewer stream, policy observations, snapshots), MuJoCo EGL raster, and
  model inference.
* `interface/viewer.py` and `interface/mujoco_live_viewer.py` already prove the
  splat-per-body + gizmo pattern; Studio generalizes them.

## GPU compatibility (H200 / A100 / RTX PRO 6000 Blackwell / A6000)

| GPU | compute cap | torch 2.4.1+cu124 envs | Studio env (`.envs/studio`, torch cu128) |
|---|---|---|---|
| A6000 (hala) | 8.6 | yes | yes |
| A100-80G | 8.0 | yes | yes |
| H200 | 9.0 | yes | yes |
| RTX PRO 6000 Blackwell | 12.0 | **no** (no sm_120 kernels) | yes (gsplat built for 8.0;8.6;9.0;12.0) |

Consequence: the Studio *process* (viser + gsplat + MuJoCo + policy client) runs on any
of the four. Heavy model stages (TRELLIS, ReconViaGen, SAM3, SAM3D, Qwen-Image-Edit,
LaMa, pi0.5 policy server) keep their existing environments and are **dispatched as
jobs** by `physicalview/jobs.py`: locally when the node's GPU is compatible with that
stage's env, otherwise as a Slurm job pinned to a compatible GPU type. The UI never
imports those models.

## Package layout

```
physicalview/
  app.py            StudioApp: viser server, tab layout, shared Context + event bus
  config.py         StudioConfig (configs/default.yaml): envs, interpreters,
                    partitions per GPU, model registries, defaults
  gpu.py            GPU detection (name, compute capability), env compatibility matrix
  scene_state.py    discover result sets under outputs/, load one scene (splat, mesh,
                    objects, clean background, sim_export, task suite); ObjectRecord
  splats.py         gaussian dict -> viser splat arrays; per-object canonical gaussians
                    posed by 4x4; importance subsampling; caches
  render.py         CUDA gsplat renders: any camera, composite (bg + posed objects),
                    thumbnails, MuJoCo robot mask overlay
  streaming.py      server-render display mode: per-client GPU render -> JPEG background
                    stream (viser camera -> w2c/K, MuJoCo free-camera robot composite)
  jobs.py           Job/JobManager: subprocess or Slurm dispatch, live logs, cancel
  pipeline.py       command builders for every stage + model/mode choices
  ik.py             damped-least-squares IK for the Panda (MuJoCo jacobians)
  robot.py          RobotSession: DroidSimEnv, joint/EE control, tasks, policy episodes
  panels/           one file per UI tab; each exposes build(ctx)
    scene_panel.py  generate_panel.py  inpaint_panel.py  robot_panel.py  jobs_panel.py
configs/default.yaml
run/{setup_env.sh,env.sh,launch.sh,selftest_gpu.py,selftest_all_gpus.sh}
tests/test_studio_*.py    CPU-only unit tests
```

## Shared Context (app.py)

```python
class Context:
    server: viser.ViserServer
    config: StudioConfig
    gpu: GpuInfo
    jobs: JobManager
    scene: SceneState | None      # currently loaded scene (None until loaded)
    robot: RobotSession | None    # created lazily by the robot panel
    selection: Selection          # selected object id / 3D box / camera frame
    events: EventBus              # publish/subscribe by topic (see below)
    display_mode: str             # "server" | "client" (scene panel keeps it current)
    stream: ServerRenderStream | None   # server-render stream (scene panel)
    log(msg) / set_status(msg)
```

Event topics (payload):

| topic | payload | emitted by |
|---|---|---|
| `scene.loaded` | `SceneState` | scene_panel after load |
| `scene.objects_changed` | list of object ids whose artifacts changed | jobs on completion |
| `selection.changed` | `Selection` | scene_panel / inpaint_panel |
| `job.updated` | `Job` | JobManager (state or log change) |
| `robot.tick` | dict(qpos, body poses, obs images) | RobotSession loop |
| `scene.pose_edited` | {object, T} at the end of a gizmo drag | scene_panel |
| `inpaint.version_selected` | clean-background ply path or None | inpaint_panel |
| `display.invalidate` | None — re-render the server stream (coalesced) | any panel (quality/resolution/layers/gizmo) |
| `display.mode_changed` | "server" \| "client" | scene_panel |

Panels only talk to each other through `ctx` and events; no cross-imports of panels.

## Display modes (scene_panel.py / streaming.py)

The Scene tab's **Display** dropdown selects how the 3D view is produced
(`viewer.display_mode` in the config, default `server`):

* **Server render (JPEG stream, recommended).** `ServerRenderStream` keeps one daemon
  render thread per connected browser client. The client's camera (`position`, `wxyz`,
  vertical `fov`, canvas size — viser follows OpenCV conventions) is converted to an
  OpenCV `w2c`/`K`, `render.Renderer` renders background + accepted objects (edited poses
  and live MuJoCo body poses respected) and, when a `RobotSession` exists and "Show robot"
  is on, MuJoCo is rendered from the same view with a free camera (`mjCAMERA_FREE`,
  `model.vis.global_.fovy` = the viewer fov) plus a segmentation render for the robot mask;
  `Renderer.composite_with_robot` puts robot pixels over the splats. The frame is pushed
  with `client.scene.set_background_image(rgb, format="jpeg", jpeg_quality=q)`. Camera
  motion and scene events (`scene.loaded`, `scene.objects_changed`, `scene.pose_edited`,
  `selection.changed`, `robot.tick`, `inpaint.version_selected`, `display.invalidate`)
  only set flags: at most one render is pending per client (frames are dropped, never
  queued), frames are rendered at `moving_scale` while the camera moved within the last
  150 ms and one full-resolution frame follows when it settles, never above `max_fps`.
  Size = min(canvas width, `max_width`) × matching height (720p / 1080p / native in the
  UI). **Why this fixes browser memory:** in the old client mode the browser received
  every Gaussian (up to 1.5 M background splats + per-object splats as float32 centers,
  3×3 covariances, colours and opacities — hundreds of MB of typed arrays plus WebGL
  buffers) and rendered them itself; laptops ran out of memory. In server mode the
  browser holds zero splat data — only a JPEG (~100-300 KB per frame) and lightweight
  helpers: a small axes frame per object (`/helpers/obj_XX`), the highlight box, the
  transform gizmo, the selected camera's frustum and the optional mesh/collision layers.
  Everything heavy runs on the server; viser is display only. The Robot tab creates no
  robot geom meshes in this mode (the robot is in the stream). Without a GPU renderer the
  stream logs once and stays idle (the panel falls back to client mode on CPU-only nodes).
* **Client splats (WebGL, high memory).** The previous behaviour: importance-subsampled
  splat arrays capped by `viewer.max_splats_background` / `max_splats_object` are uploaded
  as `/background` and `/objects/obj_XX` nodes and rendered in the browser. Handles are
  replaced, never accumulated, on reload / `scene.objects_changed` / `robot.tick`.

Config: `viewer.display_mode: server`, `viewer.stream: {max_width: 1280, jpeg_quality: 80,
max_fps: 15, moving_scale: 0.5}`. Switching at runtime tears the other mode down and
publishes `display.mode_changed`. Headless check: `physicalview.smoke` step
`server_render_stream` renders a scene camera and an orbit camera through
`ServerRenderStream.render_for_camera` and asserts the robot composite differs.

## Stage → environment routing (pipeline.py / jobs.py)

| stage | module | interpreter key | GPU needed |
|---|---|---|---|
| discover (auto) | `agents.discover.auto_segment` | `sam3` | yes (cu128 env: any GPU) |
| discover (gt) / crops | `agents.discover.factory_prepare` | `main` | no |
| refine masks | `agents.discover.factory_refine_masks` | `sam3` | yes |
| generate: trellis | `models.s4_trellis` | `main` | yes (cu124: not Blackwell) |
| generate: reconviagen | `models.s4_reconviagen --objects ...` | `main` | yes (cu124) |
| generate: sam3d | `models.s4_sam3d --objects ...` | `sam3d` | yes (cu121) |
| generate: hybrid | `agents.assets.factory_hybrid --objects ...` | `main` | yes (cu124) |
| register | `agents.assets.factory_align` / `s5_align` | `main` | no |
| physics | `agents.assets.s6_physics` | `main` | yes (Qwen text; fallback CPU) |
| report / drop test | `agents.eval.factory_report` | `main` | no |
| export MJCF | `robo.sim.export_mjcf --test --collision-mode {room,shim}` | `main` | no |
| tasks | `robo.tasks.pi05_tasks --out-dir ...` | `main` | no |
| inpaint prepare | `agents.edit.inpaint_prepare` | `main` | no |
| inpaint masks | `agents.edit.inpaint_masks` | `sam3` | yes |
| inpaint edit | `agents.edit.inpaint_qwen` (qwen or lama backend) | `sam3` / `main` | yes / cpu |
| inpaint fill | `agents.edit.inpaint_fill` | `gsplat` | yes (cu124) |
| policy server | `run/pi05_serve.sh` | `openpi` | yes (JAX; A6000/H200) |

`jobs.py` decides local vs Slurm from `gpu.py`'s compatibility matrix and the config's
`partitions` table; every job records command, env, node, start/end, exit code, and a
log file under `outputs/studio/jobs/<job_id>/`.

## Inpainting with region + prompt

Selection modes: (a) **object** — the object's removal set from `inpaint_prepare`; (b)
**3D box** — a `viser` transform-controlled box; Gaussians inside the box form the
removal set, the box is projected into the related camera views to build 2D masks. The
text prompt is passed through to the editor (`--prompt`, default keeps the legacy
"remove the <label> completely..." wording) and `inpaint_fill` re-fills only the
selected set, writing `inpaint/clean_background.ply` plus a versioned copy so the
before/after toggle in the UI is exact.

## Robot

`RobotSession` wraps `robo.envs.pi05_env.DroidSimEnv` built from the scene's
`pi05_tasks.json` (or a task authored in the UI: target object + receptacle/region).
Manual control: seven joint sliders, or drag the end-effector gizmo → `ik.py` solves
joint targets → `apply_action` at 15 Hz. Policy control: pick a registry policy
(`pi05_droid_jointpos`, `droid_pi05_jointpos_with_web_and_sim`, `scripted_sinusoid`),
start/attach a policy server, run an episode; each tick publishes `robot.tick` so the
scene panel re-poses object splats and the robot panel shows the raster and photoreal
(gsplat composite) observation images and the grasp/lift/hover/place stage bar.

## Non-goals / honesty

Studio is a tool, not an experiment producer: it never writes into
`outputs/icra2027/` freezes, and every number it displays is read from pipeline
artifacts (aligned.json, hybrid.json, physics.json, results.json), never computed ad hoc.
