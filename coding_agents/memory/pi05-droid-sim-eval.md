---
name: pi05-droid-sim-eval
description: "pi0.5 closed-loop deployment in PhiRoom MuJoCo scenes: files, serve recipe, action contract, gotchas, RoboLab/PolaRiS reference points"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ffa1f61-2c37-429f-8c36-7da10dbd323d
  modified: 2026-08-04T04:41:05.763Z
---

Interaction closed loop for PhiRoom (2026-07-22): pi0.5 (openpi) deployed in our
exported ScanNet++ MuJoCo scenes, RoboLab-style pick-and-place task suite.

**Code** (all in /group/worldcept/code/SimFoundry/simfoundry/, flat scripts):
`pi05_rig.py` (MjSpec: Menagerie panda_nohand + robotiq_2f85 attached at
attachment_site with SERL -90deg z mount quat; tabletop box; scan-frame or
synthetic ext cam; wrist cam on 2f85 base), `pi05_env.py` (DroidSimEnv: 15 Hz,
dt=1/600 x40 substeps, delta clamp 0.2 rad/tick, obs = DROID keys),
`pi05_tasks.py` (suite generator + staged TaskScorer grasp/lift/hover/place
0.25 each; robot base = grid search on tabletop maximizing reachable
graspables), `pi05_render.py` (CompositeObs: clean_background.ply + per-object
trellis_gs.ply posed from sim + MuJoCo robot masked in via segmentation ->
photoreal obs, ~0.28 s per 2-cam tick at 640x360), `pi05_eval.py` (websocket
client, warmup + reconnect-retry), `pi05_serve.sh`, `pi05_closedloop.sbatch`
(server+eval one A6000; SCENES/OBS/EXTRA env knobs). Suites at
outputs/<scene>_factory/sim_export/pi05_tasks.json (6 scenes, 16 tasks).

**Serve**: checkpoint gs://openpi-assets-simeval/pi05_droid_jointpos predownloaded
to /group/worldcept/openpi_cache/openpi-assets-simeval/pi05_droid_jointpos
(12.4GB). openpi = github.com/xuningy/openpi fork (has pi05_droid_jointpos
config; mainline does NOT). Action contract: model predicts joint DELTAS,
server-side AbsoluteActions transform returns ABSOLUTE 7 jointpos + gripper
[0,1], chunk (15,8), execute all 15 at 15 Hz then requery (RoboLab default).
DROID reset pose [0,-pi/5,0,-4pi/5,0,3pi/5,0]. Gripper: ctrl 0-255, binarize
policy dim 7 at 0.5.

**Why:** RoboLab (NVlabs, arXiv:2604.09860, Isaac-Lab) and PolaRiS
(arXiv:2512.16881, sim-evals repo) are the references; PolaRiS: visual gap
costs ~25% success, dynamics ~0 — our photoreal composite is the differentiator.
pi0.5 gets only 28% on RoboLab-120, so low absolute numbers are normal.

**How to apply / gotchas:**
- First infer = JAX jit (minutes) -> websockets 20s keepalive KILLS the
  connection; client must reconnect+retry (RoboLab does the same). Warmup once
  after connect.
- Ext camera MUST be a real scan-trajectory pose (_pick_scan_camera): synthetic
  DROID-offset viewpoints render as blurry gaussian mush (outside training
  views); scan-frame poses are photoreal (the 28dB regime).
- Composite mode only for scenes with inpaint/clean_background.ply (c50d2d1d42,
  578511c8a9, 45b0dac5e3, 27dd4da69e) — raw splat has objects baked in (ghosting).
- common.py binds SIMF_SCENE/SIMF_OUT at import -> composite eval = one scene
  per process.
- Exported scene.xml has NO furniture collision (only micro-slabs) — rig builder
  adds a solid tabletop box from the support cluster; targets filtered to
  settle drift <0.10 m, drift >0.25 m bodies deleted from sim.
- Raster (gray mesh) zero-shot: pi0.5 reaches toward objects but doesn't grasp
  (score 0) — expected OOD; use composite for real numbers.

**First full results (2026-07-22, outputs/pi05_runs/)**: full2_raster 0/16,
full2_*_composite 0/10, dbg1 instrumented run: policy DOES engage — gripper
closes 48-94% of ticks, ee-to-target min 0.10 m (mug) but closes early at the
wrong spot and wanders; classic last-15cm visual-servo failure. Diagnosis from
policy-input dumps: (a) wrist near-field = beige splat mush (scene splat never
trained at 10-30 cm; TRELLIS object splats themselves look fine up close),
(b) ext scan-frame camera leaves the ROBOT small at frame edge — DROID ext
cams are over-the-shoulder 0.4-0.9 m from the BASE. Next levers, ranked:
(1) try gs://openpi-assets-simeval/droid_pi05_jointpos_with_web_and_sim
(sim-co-trained variant, PolaRiS says co-training is the fix), (2) constrain
_pick_scan_camera to over-the-shoulder poses near the base, (3) depth-composite
MuJoCo table into wrist view to replace near-field mush, (4) 32 s time limit
(16 s too short) + episodes>1 with jitter. Also: sbatch needs unique --port per
job (localhost collision, tyro wants --port BEFORE policy:checkpoint) and the
job's first infer takes ~2 reconnects (jit) — normal.
## 2026-07-27: loop was DEAD for 5 days on two independent infra bugs
Paths all moved — see [[simany-rename-2026-07-27]] (pi05_* now in
`simany/robot/`, serve script in `scripts/`, sbatch in `scripts/slurm/`).
Between 07-22 and 07-27 the closed loop never once started; three jobs
(645181, 645906, 645991) all died before the policy server was up:
1. **orbax restore against CephFS hangs silently.** Last log line is
   `Restoring checkpoint from /group/worldcept/openpi_cache/...` and then
   nothing for 63 min — no error, no progress. A plain rsync of the same 12 GB
   path finishes in 8.7 s, so it is the many-small-scattered-reads restore
   pattern specifically, not a filesystem outage. Fix: `pi05_serve.sh` prefers
   a node-local `/scratch/runyi_yang/openpi_cache_local/pi05_droid_jointpos`
   copy gated on a `.copy_complete` marker -> **restore drops to 15.4 s**.
   Caveat: /scratch is NODE-LOCAL, so the copy only helps on the node that
   made it (hala today).
2. **The sbatch never had `#SBATCH --mem`.** `--mem=100G` was only ever typed
   on the 07-22 command line; without it the cgroup default is `ReqMem=2G` and
   the policy server is OOM-killed mid-restore (645991, ExitCode 0:125,
   `OUT_OF_MEMORY`). This is host RAM, not GPU. Now baked into
   `scripts/slurm/pi05_closedloop.sbatch`.
Honest score baseline to beat: **all 6 runs on disk are 0.0** — 32 episodes,
zero successes, `mean_score=0.0` everywhere (full2_raster n=16,
full2_*_composite n=2/3/5, dbg1 n=5, smoke4 n=1).
Videos DO exist and are worth showing: 32 mp4 / 236 MB in `outputs/pi05_runs/`,
of which 15 are photoreal composite. Composite mp4s are **1280x360 = two
640x360 panes** (ext | wrist), 480 frames @ 15 fps = 32 s. The ext pane is
genuinely photoreal with the Panda+2f85 cleanly composited; the wrist pane is
the known near-field splat mush. Cropped/upscaled ext-only 720p clips in
`outputs/pi05_runs/robot_ext_clips/`.

### First green run after the fixes (job 646895, 17m34s, COMPLETED)
`fix2_c50d2d1d42_composite`: **0/5, mean 0.000**, all four stages False on all
5 tasks, 240 ticks each. Apples-to-apples vs `full2_c50d2d1d42_composite`
(same 240 ticks, same 5 tasks, OLD wrist cam) which was also exactly 0.000 ->
**the 07-26 wrist-camera move to the gripper centerline changed nothing
measurable.** Visual check of the wrist pane across 4 matched ticks: the
near-field splat mush is present in BOTH old and new, and the new camera makes
the gripper jaws occupy noticeably more of the frame, i.e. less visible scene.
So the change addressed neither diagnosed root cause. Of the four levers ranked
on 07-22, the two highest-value ones are still UNTRIED: the sim-co-trained
checkpoint `droid_pi05_jointpos_with_web_and_sim`, and depth-compositing the
MuJoCo table into the wrist view to replace the mush. Also note dbg1's 480-tick
(32 s) episodes were never reproduced — the suite default is 16 s, so the "32 s
time limit" lever silently reverted; pass `--time-limit 32` explicitly.
Operational note: first infer needs **3** websocket reconnects (not ~2) for the
jit; the server-side handshake tracebacks during that window are noise. The
retry cap is 5 — do not lower it.

## Demo-render recipe (2026-08-03) — and three traps
Deliverable: `outputs/pi05_runs/demo_final/c50_mug_grasp_lift_720p.mp4`
(grasp+lift, score 0.50, ee_dist_min 4.0 cm, no wrong grasps).
Recipe: `--task-filter obj_09 --ext-cam-frame DSC01594 --render-wh 1280 720
--time-limit 32` (NO declutter), then crop the ext pane in post:
`crop=1280:720:0:0,crop=820:461:230:10,scale=1280:720`.
New DEMO-ONLY flags in pi05_eval (they change what the policy sees, so never
use for benchmark numbers): `--ext-cam-frame`, `--demo-declutter`,
`--render-wh`. `pi05_tasks.ext_cam_from_frame(frame)` pins any scan frame.

**Trap 1 — the ext camera materially changes the score.** Same task, same
everything else, 720p: DSC01594 = 0.50 (grasp+lift), DSC01593 = 0.25,
DSC01616 = 0.00. The policy is very sensitive to exterior viewpoint, so a
camera picked for looks can silently destroy the result. Pick camera and
report score together.
**Trap 2 — declutter and higher resolution fight each other.** Objects were
hiding the inpaint scars left where OTHER objects were removed; dropping them
exposes big pale patches, and 720p magnifies them. Keep full clutter for
demos: it both looks more honest and covers the residue.
**Trap 3 — a static home-pose frame does not tell you the framing.** DSC01616
looked ideal (no chair, big mug) but the mug sits on the bottom edge, so the
actual grasp happened half out of frame. Judge candidate cameras by the
target's MARGIN to the frame edge, not its size.
Also: `_pick_scan_camera`'s score (align/distance/height only) is blind to what
fills the frame — on c50 it picks DSC01593 where the office chair eats ~25% of
the image. The chair/monitors/papers are FURNITURE, absent from the object
table and baked into clean_background.ply, so they cannot be removed — only
cropped out or dodged by camera choice. Survey tooling (throwaway, in the
session scratchpad): cam_survey.py renders all 23 candidate poses,
framing_survey.py renders the shortlist at 720p through the production
CompositeObs path.
Unrelated defect found while doing this: with ALL objects excluded, several
transparent bottles STILL appear in clean_background.ply — removal failed for
them, matching the known "diffuse gaussian cloud far from geometry" case.

## 2026-08-03: "use a less cluttered scene" — tried, and why it failed
**Only 4 of 50 `outputs/*_factory` scenes have `inpaint/clean_background.ply`**
(27dd4da69e, 45b0dac5e3, 578511c8a9, c50d2d1d42), and composite REQUIRES it.
That 4/50 pool — not scene availability — is what constrains every demo
choice. Composable object counts: 45b=7, 27dd=17, c50=18, 578=25.
Picked **45b0dac5e3** (bathroom sink, cleanest, recon PSNR 34.1 dB = #4/50;
sim holds just cup obj_01 + 4 bottles). It genuinely looks far tidier than the
c50 desk — and it still cannot carry a demo:
1. **Policy fails outright: 10/10 episodes 0.000** across 4 ext cameras
   (DSC05695/05698/05699/05735) and BOTH tasks, 2 eps each with jitter, 32 s.
   Best approach 6.4 cm vs the 4.0 cm that grasped in c50.
2. **No admissible scan pose frames a counter-mounted arm.** All 55 candidates
   rendered at 720p: back off (1.31-1.34 m) and the whole arm fits but the
   target is too small for the policy; move in (1.17-1.21 m) and the wall
   cabinet occludes the arm AND it gets smaller (base is further from cam).
   The room is small and the scan looked at the sink, not at where an arm
   would stand.
3. **Cropping cannot rescue it** (the trick that saved c50): zooming in fills
   the frame with the MIRROR, which 3DGS reconstructs as a large dark smear,
   and the arm sits right against it.
Corroborated regularity: policy performance tracks the target's apparent size
in the exterior image, while good composition needs distance — directly
opposed. Also explains c50's 0.50 -> 0.00 swing across cameras.
Suspected extra factor here: a grey-white cup (0.086x0.088x0.097 m) on a WHITE
counter is very low contrast, vs c50's mug (0.15x0.10x0.089) on beige.
**Actionable**: to get clean scene + real success, widen the composite pool by
running `run_inpaint.sh` on more of the 50 scenes, then pick a tidy one with
good scan coverage. That is the root unblock, not more camera search.

## 2026-08-04: pool widened to 6 — and scene-hunting is a DEAD END
Ran `run_inpaint.sh` on 825d228aec (2 h 52 m, 15 views) and 7b6477cb95
(4 h 10 m, 29 views) via the new `scripts/slurm/inpaint_scene.sbatch`
(`-p batch` because Qwen exceeds the 4 h debug wall; `-w hala` because it is
the only a6000/sm_86 and the gsplat JIT cache holds sm_86/sm_90 only — an A100
would silently rebuild it; Qwen needs the sam3 env + `SIMANY_QWEN=force`).
Composite pool: 27dd4da69e, 45b0dac5e3, 578511c8a9, 7b6477cb95, 825d228aec,
c50d2d1d42. Hole PSNR: 825d trained 42.2 / held 25.3 dB; 7b64 30.3 / 25.7 dB.
Two prerequisites new scenes need first: `export_mjcf --test` (no scene had
`sim_export/` outside the original 6; `--test` is also the only source of
`mujoco_settle.json`, without which `exclude_objects` is silently always empty),
then `pi05_tasks`.
Fixed a real blocker while doing it: `export_mjcf.py`'s degenerate-hull filter
only rejected `<4 vertices`, missing MuJoCo's *separate* "mesh volume is too
small" rejection (flat sliver with >=4 verts) — 7b6477cb95 obj_12_p12 failed
compile. Now also drops hulls whose convex-hull volume < 1e-9 m^3. This will
recur on any of the other 44 scenes.

**RESULT: the new scenes score 0/36 in composite.** Post-geometry-fix summary
(episodes / nonzero / lifts / best ee):
| scene | mode | eps | nz | lifts | ee_best |
|---|---|---|---|---|---|
| c50d2d1d42 | composite | 10 | **4** | **2** | 0.027 |
| d755b3d9d8 | raster | 10 | 5 | 0 | 0.010 |
| f3d64c30f8 | raster | 15 | 3 | 1 | 0.014 |
| c50d2d1d42 | raster | 15 | 2 | 2 | 0.016 |
| 45b0dac5e3 | composite | 18 | 0 | 0 | 0.041 |
| 825d228aec | composite | 18 | 0 | 0 | **0.078** |

**My scene-selection criteria were falsified.** 825d228aec was best-in-class on
all three things I pre-validated — jaw fit (3 staplers, 35x50x107 mm, all well
inside the 85 mm stroke), bearing (all targets within +-6.2 deg), inpaint
quality (target views 41-54 dB) — and still went 0/18, never even reaching
within 7.8 cm. So those criteria do NOT predict policy success; do not reuse
them as if they do.
Post-hoc explanation (not predictive, found after the fact): **"the gripper can
close on it" and "the policy can see it" pull in opposite directions.** A
stapler is low, dark and small on a WHITE bench; c50's mug is too wide to
envelop (89x100 mm vs 85 mm stroke) yet big and high-contrast on a beige desk,
and the policy grabs its handle. Optimising jaw fit alone made targets
invisible.
**Conclusion: stop hunting scenes.** c50d2d1d42 composite (4/10 nonzero, 2
lifts) is the only configuration that works and is a favourable outlier, not
the typical case. For a presentable clean-scene clip the realistic path is the
scripted-IK driver (DLS IK waypoints + weld at grasp, no policy server, works
in any scene, 825d228aec's clean bench is ideal) — labelled as a scripted
demo, with policy numbers kept separate and honest.

Related: [[simfoundry-repro-status]], [[cluster-gpu-and-envs]],
[[simany-rename-2026-07-27]], [[pi05-zero-score-root-cause]].
