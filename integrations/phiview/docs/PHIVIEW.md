# PhiView

PhiView runs Gaussian rendering, object picking, highlights, camera motion, model stages,
and MuJoCo physics on the server. The browser displays JPEG frames in one HTML image
and sends input events. There is no canvas, WebGL, mesh, Gaussian array, or client physics.

## Run the H200 demo

```bash
cd /group/worldcept/code/PhysicalView
sbatch run/phiview.sbatch
```

The job runs real-GPU checks before starting port 8095. The log prints the SSH tunnel
command. Default scene: `c50d2d1d42_factory`; full original 1,499,998 Gaussians, SH degree 3.
The stream defaults to 1920×1080; Native resolution uses 1752×1168 for this capture.
No Gaussians are subsampled at either resolution. JPEG compression is used for transport.

```bash
source run/env.sh
"$STUDIO_PY" -m physicalview.phiview --demo --selftest \
  --scene c50d2d1d42_factory --out outputs/my-phiview-run --port 8095
```

Run the command inside a GPU allocation. A session shares one scene, selection and camera
among viewers. Job duration defaults to four hours. Files persist after the job ends.

## Controls

- The mouse toolbar has **View**, **Select object**, and **Move object** modes.
  View uses left-drag to orbit, right-drag (or Shift-left-drag) to pan, wheel to zoom,
  and middle-drag to dolly. Select object uses a click or left-drag box without moving
  the camera. Move object drags the selected simulatable body horizontally at its
  current height, with physics paused. Reset restores its starting pose.
- WASD moves relative to the view; Q/E moves vertically. Shift triples speed. F focuses
  the selected object and sets its orbit center. Navigation remains on the server.
  Fixed robot camera views do not move; choose Free view to navigate. Inputs stop on
  window blur or a 350 ms heartbeat timeout.
- In Select object mode, click any reconstructed object, or choose one in the object list. Known
  regions select immediately. An unlabelled region runs a SAM3 positive point prompt on
  the displayed frame, then lifts its visible surface to unassigned original Gaussians.
  It appears as a new clicked object with a red mask and white outline. Wait for selection
  to finish, then click **Make simulatable**. Expired frames and invalid surfaces are rejected.
- **Deselect** or **Esc** clears the selected mask and cancels pending point/box selection.
  It preserves the object's physics and generated assets. Escape also ends a drag.
- For ScanNet++ `fb5a96b1a2`, `--demo` automatically prepares the green spray bottle:
  a curated box at source view `DSC03413.JPG` prompts fresh SAM3 inference, selects the
  bottle in red, and enables its collision proxy with simulation paused. It combines
  earlier unprepared click fragments covered by the mask, retaining their files as
  provenance. Existing physical bodies and other proposals are not reassigned.
  **Prepare green bottle** selects the persisted object on subsequent uses without
  duplicating it. Use `--no-demo-prepare` to opt out. This is a scene preset, not a
  general color detector; selection/build failures are shown in the demo status.
- Original displays the original splats. Simulatable uses an inpainted background when
  available, otherwise the observed original background with exposed regions left unfilled.
  Native restores the capture resolution.
- Make simulatable enables an existing body or builds a convex collision proxy from
  the selected Gaussian centers. New proxies use default density (500 kg/m³) and friction,
  explicitly marked unmeasured in the physical parameters. The proxy approximates a
  partial observed surface; it does not reconstruct hidden geometry or infer a material.
  Existing bodies keep their current poses and reset poses. Create new bodies before
  placing a robot. Original proposals still support the full Build physics pipeline;
  clicked objects use the interactive proxy builder.
- Physical parameters show the active MuJoCo mass, inertia and friction alongside the source
  estimate. Friction edits affect the selected body's collision geoms. Fall raises the body
  0.3 m and releases it; Friction gives it 0.7 m/s horizontal velocity; Throw gives it the
  chosen camera-directed velocity plus upward velocity. Run/Pause/Reset control simulation.
- Shooting sends a real sphere along the clicked camera ray. The server simulates contacts,
  gravity and friction. Eight projectiles are reused. A projectile's speed is bounded at
  20 m/s. This is a discrete rigid-body simulation, not a ballistic penetration model.
- Generated alternatives need a valid registration and mesh. Choosing one rebuilds its
  physical collision hull and inertia as well as its visual Gaussians. Physics resets and
  the arm is removed when collision geometry changes. Collision uses a convex hull for
  these interactive switches; Build physics provides the pipeline's collision decomposition.
- Inpainted selected/all hides the respective objects using cached 3D clean backgrounds.
  Enter a prompt to run Qwen-Image-Edit-2511 and fit replacement background Gaussians.
  Each edit uses a new writable build. Existing masks can be reused; previously edited
  images are deleted only in the new copy, so the new prompt must execute. A guard rejects
  skipped/cached edits and LaMa fallback. Completed prompt versions are keyed by the exact
  removed-object set. The panel reports queued/running/failed/succeeded state and job paths.
- The control-policy selector chooses scripted IK or the installed π0.5
  checkpoint variants. Viewer hardware stays fixed to DROID Franka/Robotiq. The
  third-person exterior and wrist views use actual simulated camera poses.
  See [selection, robot and camera contracts](PHIVIEW_SELECTION_ROBOT.md) for
  mouse gestures, model loading, image preprocessing and control semantics.
  All selection highlights are hidden during policy loading and execution, including
  scripted IK, then restored on finish, pause, or failure. Selection and the highlight
  preference are preserved. Motion completion is not labeled as grasp or task success.

## Environment

Click discovery uses the configured `sam3` interpreter and a locally cached
`facebook/sam3` checkpoint. Set `PHIVIEW_SAM3_CHECKPOINT` to use an explicit checkpoint.
Inference runs in a separate server process; the browser continues receiving images.
The session output stores `clicks/` (actual input RGB, depth, camera, mask and inference
receipt) and `interactive_objects/` (Gaussian indices, selection metadata and collision
proxy). Reopening the same scene and output directory restores these objects. Source
scene artifacts are never edited. New object discovery invalidates prior background
completion for the changed removal set. Asset generation for clicked objects is not
enabled until generation inputs are prepared.

`configs/phiview.yaml` uses the shared Studio CUDA environment for rendering, the original
SimAny environments for generation, and `run/phiview_inference_python.sh` for SAM3 and Qwen.
The latter adds isolated packages under `.envs/phiview-inference` and official SAM3 source
under `.envs/phiview-sam3-source`; it does not modify the shared Studio environment.
`run/setup_phiview_inference.sh` restores these dependencies. Available shared weights:

- SAM3: `/group/worldcept/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`
- Qwen 2511: `/group/worldcept/hf_cache/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`

The Qwen path can be overridden with `PHIVIEW_QWEN_MODEL`. Existing TRELLIS, ReconViaGen,
SAM3D and physics model availability is checked by their real pipeline stages. A failed
stage remains failed; it is not replaced with a dummy asset.

## Evidence and boundaries

Every run writes `manifest.json`, `actions.jsonl`, snapshots and `validation/checks.json`.
Checks cover pixel-exact original rendering, native resolution, real image picking,
selected/all removal, fall/friction, throw video, projectile contacts, generated alternatives
and robot command motion. Failed attempts remain in their own output directories.

The default scene contains 18 existing proposals and 16 constructed bodies. Two proposals
use approximate bounding-region selection; 16 use cached removal masks. Proposal metadata
includes GT references. Automatic highlighting is not evidence of exhaustive new discovery;
Discover with SAM3 invokes a fresh discovery into an empty build and invalidates prior
object/collision associations.

The inherited collision scene has support slabs, not independently verified full-room
collision. The inherited clean background leaves visible remnants around transparent
objects. Prompt editing is an additional construction step; image quality is not guaranteed
by a passing endpoint or mask test. Fresh selected-object prompt versions have no regenerated
3D object labels, so selection in that view uses the object list until switching back.

The scripted robot controller supports the listed command vocabulary. It is not a general
language policy, and a successful grasp or placement must be established from the actual
simulation, not inferred from a completed trajectory. These demo artifacts are separate
from paper evaluation results.

Rendering uses [gsplat's rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html)
for full SH color, depth and object-membership passes; physics uses MuJoCo.

## Capture a downloadable green-bottle demo

Run `tools/demos/capture_green_bottle.py --session SESSION --config CONFIG --out NEW_DIRECTORY`
in the compatible GPU environment. It requires a session with the prepared green bottle.
The output includes original and moved GS images, a contact-driven scripted robot push,
a shooting MP4, camera/physics receipts, and a ZIP. It checks robot displacement and
projectile-bottle contacts, and never overwrites the source session. The scene/bottle
remain Gaussian renders; robot/projectile meshes are composited using depth. The
robot capture uses scripted IK and is not a learned-policy success claim.

The viewer serves capture files placed in its session's `downloads/` directory.
`green-bottle-demo.zip` enables its Download demo images + video link.
