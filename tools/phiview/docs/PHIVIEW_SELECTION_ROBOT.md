# Image selection and robot cameras

PhiView keeps rendering, segmentation, collision construction, policy inference,
and physics on the server. The browser displays JPEGs and sends input events.

* **View**: left-drag orbits, right-drag/Shift-left-drag pans, wheel zooms, and
  middle-drag dollies. Clicks in this mode never select or segment objects.
* **Select object**: click selects a known object or requests a SAM3 point mask.
* **Move object**: drag the selected simulatable body across its horizontal plane.
  Physics is paused; Esc ends the drag and deselects, and Reset restores the initial pose.
* In Select object mode, left-drag draws a bounding box. A dominant existing object is selected directly;
  otherwise SAM3 receives the box on the exact displayed frame. Disconnected
  distractors are removed from the prompted mask. Dragging pauses
  physics and pins that frame for up to 60 seconds. Selected objects stay bright red with a white outline when other
  highlights are disabled.
* **Make simulatable** creates a convex collision proxy from the selected visible
  Gaussian surface. Hidden geometry is unknown; mass and friction are unmeasured
  defaults. Object records and proxies are scoped to their source build so IDs
  from rediscovery cannot inherit an unrelated collision body. Existing primitive
  floor/table supports survive rediscovery; no new support geometry is inferred.
* **Deselect** or **Esc** clears selection and invalidates any pending SAM3 result.
  Physical bodies and saved assets remain available. The `fb5a96b1a2` demo prepares
  the green spray bottle automatically, using a curated source-camera box and fresh
  SAM3 inference. It restores that object on restart; **Prepare green bottle** can
  select it again. Only overlapping unprepared click fragments are consolidated,
  with their original receipts/files preserved. Use `--no-demo-prepare` to skip it.

The viewer selector chooses the control policy only: base π0.5 DROID, the
sim co-trained π0.5 variant, or explicitly labeled scripted IK. Robot hardware is
fixed to the DROID Franka with Robotiq 2F-85 preset. The backend retains its modular
rig adapters for pipeline tools, but the viewer does not offer a hardware selector.

The third-person view is an actual fixed exterior MuJoCo camera, positioned
at `[0.05, 0.57, 0.66]` m in the DROID base frame and aimed at
`[0.55, 0, 0.10]` m, with a 68-degree vertical field of view. The wrist camera is
attached to the gripper and uses the existing DROID rig's 58-degree field of view.
These are simulated cameras following the existing π0.5 rig preset; they are not
calibration measurements from a physical camera in the scanned room. Choose
**Free view** to navigate; WASDQE and right-drag do not move a fixed camera.

The two π0.5 checkpoint choices come from PhiRoom's policy registry. Executing a
command starts one private, session-owned checkpoint server using the configured
`openpi` interpreter. It does not connect to arbitrary existing policy servers.
The installed base π0.5 checkpoint is the default control model. First loading and JIT compilation can take minutes. Status shows loading,
inference, execution, or failure. Checkpoint and response receipts are stored in
the session's `policies/` directory.

Policy inputs are unhighlighted exterior and wrist images rendered at 1280×720,
resized with padding to 224×224, seven joint positions, gripper position, and the
user's command. The same physical camera transforms drive the viewer's camera
choices. Actions are chunks of 15 absolute 8D targets (seven joints and gripper),
executed at 15 Hz of simulation time with 40 physics substeps and a 0.2 rad joint
delta cap. A default command runs 150 ticks. This is not a promise of 15 FPS wall
time or manipulation success. Pause/reset cancel execution, and late inference
responses cannot restart it. All object highlights are hidden while a learned policy
loads/executes or scripted IK is running, then restored without losing selection.
The simulation has no physical robot connection.

Modules: `phiview_box.py` handles rectangle selection; `phiview_selection.py`
handles discovery and persistence; `phiview_rigs.py` handles hardware and camera
contracts; `phiview_policy.py` handles observations and asynchronous control;
`phiview_policy_server.py` loads checkpoints in their separate environment;
`phiview_demo.py` handles scene-specific automatic preparation; `phiview_navigation.py`
and `phiview_drag.py` handle camera gestures and manual object placement. The small
`web/phiview_mouse.js` module routes browser input for the three mouse modes.

## Validation on 2026-09-14

On scene `fb5a96b1a2_factory`, implementation `27a6b47` passed 128 CPU tests
(50 explicit prerequisite skips), 31 focused checks in A6000 allocation `898843`,
and initial Chromium checks for click selection, reversed box discovery, red highlighting
with other masks disabled, Make simulatable, hardware switching, fixed exterior
and wrist cameras, and free navigation. The browser transferred no 3D assets and
created no canvas. A fresh SAM3 box mask isolated the visible chair surface; its
convex proxy parameters remain unmeasured defaults.

The base π0.5 checkpoint executed two observation/action chunks (30 ticks, 2 seconds
of simulation time) on this implementation, with server metadata confirming the
checkpoint, absolute action convention, and A6000 inference. The sim co-trained
variant executed 15 ticks and pause cancellation passed during the preceding
implementation check. These checks establish runtime integration, not task success.

Evidence is under PhiRoom's ignored output directory
`outputs/demo-fb5a96b1a2-redmask/live/`: `selection-camera-check-27a6b47/`,
`session-selection-final-test/policies/`, and `robot-camera-check/`.
The original v2.0.0 release media is unchanged. After user clarification, the
hardware dropdown was removed; the viewer exposes only the policy selector and
keeps DROID hardware fixed.
