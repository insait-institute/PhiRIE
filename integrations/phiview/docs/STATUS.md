# SimAny Studio — status (2026-09-04)

Branch `feature/studio-ui` (worktree `/group/worldcept/code/SimAny-wt/studio`).

## Verified end to end (headless smoke `physicalview/smoke.py`, scene c50d2d1d42_factory)

| node / GPU | compute cap | env | result | notes |
|---|---|---|---|---|
| hala / RTX A6000 | 8.6 | `.envs/studio` (torch 2.9.1+cu128) | **ALL PASS (9/9)**, job 822673 | renderer 27.9 dB vs photo (raw bg), composite ok, robot IK 0.7 mm, scripted episode 45 ticks, all 5 tabs built |
| hala / RTX A6000 | 8.6 | mini-viewer (torch 2.4.1+cu124) | 7/9 | robot step needs `pydantic` (absent there); sam3 mount missing -> builders now tag instead of fail |
| gcp-eu1-rtx6000-vz3w / RTX PRO 6000 Blackwell | 12.0 | `.envs/studio` | **ALL PASS (9/9)**, job 822672 | first-touch cephfs latency (discover 137 s, load 91 s); cu124 envs cannot run here |
| gcp-eu1-a100-80g-qrfh / A100-80GB | 8.0 | `.envs/studio` | **PASS** self-test (`run/selftest_all_gpus.sh`) | gsplat 640x360 render 3.3 ms, MuJoCo EGL ok; cold cephfs import ~3 min on first touch |
| sof1-h200-5 / H200 | 9.0 | `.envs/studio` | **PASS** self-test | gsplat render 1.4 ms, MuJoCo EGL ok, 8 s total |
| (all four) | | | | identical render output across cards; gsplat_cuda.so holds sm_80/86/90/120 cubins (cuobjdump), reused with no rebuild on every node |

Unit tests on the branch: `PYTHONPATH=$PWD .venv/bin/python -m pytest -q tests` -> 427 passed, 2 skipped.

## Run it

```bash
# GPU node (any of the four types)
cd /group/worldcept/code/SimAny-wt/studio
bash run/launch.sh --gpu a6000            # or h200 | a100 | rtx6000 ; interactive srun
# or manually inside an allocation:
source run/env.sh && $STUDIO_PY -m physicalview.app --port 8080
# from your laptop
ssh -L 8080:<node>:8080 <login-host>   ->  http://localhost:8080
```

## Tabs

* **Scene** — pick any of the 369 result sets (factory/auto/full/droid/video/behavior), layers (raw/clean background, mesh, object splats, collision), object table + selection + gizmos, camera snap, photoreal snapshot.
* **Generate** — discovery model (GT segments / SAM3 auto), generation model (TRELLIS, ReconViaGen, SAM 3D Objects, hybrid), registration mode (yaw-sweep ICP, signed / alternative source-up), per-object generate/register/physics, report, export MJCF (room/shim collision), task generation, full pipeline; proposal cards with construction-time evidence and an "Accept proposal" action.
* **Inpaint** — region by selected object or a draggable 3D box, text prompt + negative prompt, backend (Qwen-Image-Edit or LaMa), refine iterations, versioned clean backgrounds with before/after.
* **Robot** — task dropdown or authored task (target + receptacle/region + instruction), policy (pi0.5 DROID, pi0.5 sim-cotrained, scripted), policy-server start/check, raster or photoreal-composite observations, run/stop episode with grasp/lift/hover/place stage bar, 7 joint sliders + gripper, end-effector gizmo driven by IK, robot geoms live in the 3D view.
* **Jobs** — every stage runs as a job (local when the node's GPU supports that stage's env, otherwise `srun` to a compatible GPU type), live log tail, cancel, GPU compatibility matrix.

## Known limitations

* `/group/streetsplat/worldcept/.envs/sam3` (SAM3 + Qwen-Image-Edit env) was unmounted on the login node and hala during testing; affected stages are tagged `interpreter_missing` and dispatched remotely. Long-term fix: host SAM3/Qwen in the cu128 studio env.
* TRELLIS / ReconViaGen / SAM3D / pi0.5 server envs are cu124/cu121 builds: on the Blackwell node they are dispatched to A6000/H200 via Slurm automatically.
* No pi0.5 server was started during the smoke (12 GB checkpoint, needs 100 GB RAM job); the scripted policy exercised the episode loop.
* `physics` stage has no per-object flag upstream (annotates all non-rejected objects).

## 2026-09-05 — server render display mode (browser memory fix)

The Scene tab now defaults to **server render**: `physicalview/streaming.py` renders the browser
camera's view on the GPU (gsplat, robot composited from MuJoCo via a free camera) and streams JPEG
frames as the viewer background; the browser holds no gaussian data (client WebGL splats remain as a
fallback mode). Verified on hala / A6000 (job 832089): `physicalview.smoke` **ALL PASS (10/10)** with
the new `server_render_stream` step (viser-camera conversion 43.3 dB vs `Renderer.render_camera`,
robot composite changes 2.3 % of pixels, mask 2.4 %) and the `viser_server` step loading the scene
through the real panel: 0 splat nodes in server mode (17 helper frames + frustum), 17 splat nodes in
client mode, clean switch back. CPU tests: 118 passed, 13 skipped (`tests/test_streaming.py` added).
