---
name: pi05-zero-score-root-cause
description: "Why every pi0.5 PhiRoom/SimAny episode scores 0.000: base-placement geometry bug (isotropic reachability, no frontal cone) — plus what is ruled out"
metadata: 
  node_type: memory
  type: project
  originSessionId: 557e99ab-bcd8-44e5-8f1b-22d3105d0258
  modified: 2026-08-04T15:40:13.633Z
---

Diagnosed 2026-07-28 on the 0.000-everywhere pi0.5 closed-loop results
(see [[pi05-droid-sim-eval]]). The user's hunch ("it's geometry") is correct.

## The bug
`simany/robot/pi05_tasks.py`:
- `_place_robot()` picks base XY by maximizing `n_reach(p)` (L120-122) =
  count of graspables with `norm(center_xy - p) <= REACH_M - 0.1`. **Pure
  isotropic radius, no angular term.** Maximizing a count-within-radius over a
  cluster lands the base in the MIDDLE of the clutter — so objects end up at
  every bearing, including behind the arm. Real DROID desks mount the arm at
  the EDGE facing in.
- L126 then sets `yaw = arctan2(cen - base_xy)` where `cen` (L103) is the
  centroid of **all** table `members`, a different set than the graspables the
  position was optimized for. Position objective and yaw objective disagree.
- `reachable()` (L215-217), used to admit task targets, is the same isotropic
  radius test. So a target 0.675 m directly BEHIND the arm passes.

## Evidence (scene c50d2d1d42, base yaw 103.5 deg)
Bearing relative to base heading vs closest approach:
| obj | dist_base | yaw_off | dist from home EE | ee_dist_min dbg1/fix2 | grasped |
|---|---|---|---|---|---|
| obj_04 | 0.418 | +18.2 | 0.300 | - | YES (wrong) x2 |
| obj_08 | 0.501 | -19.0 | 0.304 | - | YES (wrong) x3 |
| obj_03 | 0.687 | +41.7 | 0.537 | 0.392/0.299 | no |
| obj_09 | 0.546 | -49.3 | 0.485 | 0.100/0.190 | no |
| obj_14 | 0.656 | -60.5 | 0.587 | 0.179/0.300 | no |
| obj_17 | 0.398 | -65.5 | 0.444 | 0.486/0.401 | no |
| obj_02 | 0.675 | **+143.2** | 1.026 | **1.023/1.023** | no |

- obj_02's `ee_dist_min` is `1.023152293668803` in BOTH dbg1 (480 ticks) and
  fix2 (240 ticks) — identical to 16 sig figs, and equal to its distance from
  the reset pose. The arm never moved toward it at all, in either run.
- The only two objects ever grasped are exactly the two nearest the home-pose
  gripper axis (+-19 deg, 0.30 m). **Neither is ever a commanded target.**
- Whole suite: **11/16 targets outside +-30 deg, 8/16 outside +-60, 5/16
  outside +-90 (literally behind).**

## Ruled out (do not re-litigate)
- **Render realism.** `full2_raster` (n=16, NO splat at all, plain MuJoCo
  raster) and every photoreal composite run both score exactly 0.000. The
  largest possible realism swing produces zero difference. So 3DGRUT/ArtiFixer
  cannot be the lever — they bet on a second blocker behind the first.
- **Target visibility.** 15/16 targets project INSIDE the 640x360 exterior
  camera image (only d755b3d9d8 obj_18 is out of frame, u=-46.6 px), depths
  0.49-1.71 m.
- **Gripper convention inversion.** obs `0=open..1=closed` matches openpi
  `droid_policy.py`; `grip_first_close` varies (1,15,29,32,53,58,75) rather
  than pinning at 0/1 as an inversion would.
- **The scorer.** `TaskScorer.update()` L332 uses `env.grasped(t)` — the SAME
  predicate as the `wrong_grasps` diagnostic at L350, which demonstrably fires.

## NOT explained by geometry alone
45b0dac5e3's 2 targets (-20.3, +18.0 deg) and f3d64c30f8's 3 targets (+22.1,
+18.2, -30.2) are well-aimed, in frame, 0.72-1.02 m deep — and still 0/5.
So at least one more blocker exists. Next suspects: arm mounted amid clutter so
approach paths are blocked; CoACD hull vs 2f85 jaw width; approach-pose
feasibility.

## 2026-08-04: object placement / "the mesh looks broken" — 3 SEPARATE bugs
The user reported the target mug was placed wrong. My first hypothesis — objects
free-fall onto a too-low collision table — was **REFUTED by causal test**; do
not repeat it. `corr(drop-onto-box, settle drift) = +0.032` (none) vs
`corr(t=0 penetration, drift) = +0.467`; deleting `support_11..15` cut obj_09
drift 27.0 -> 4.4 mm, while deleting the neighbouring *bodies* left it unchanged.
1. **P0, the real cause of drift.** `simany/sim/export_mjcf.py:139-148`
   (`SLAB=0.02` at `:24`, aabb source `:126`, pad `:141`) emits one private
   static 2 cm slab per object with its AABB **padded 0.10 m per side**. In the
   bottle/mug cluster these overlap and impale neighbours: obj_09 starts
   **21.8 mm inside three foreign bottle slabs** and is ejected upward by
   exactly that, total drift 42.9 mm. scene.xml has NO background collision at
   all (`grep -c background` = 0) despite the module docstring. 6/16 objects
   also start inside their OWN slab because the slab height comes from a
   ~500-vertex strided subsample (`:123`) of a ~20k-vertex mesh, not the
   collision hulls. Fix: one static box per detected support level
   (`cluster_heights()` at `:27` exists and is unused), or at minimum shrink the
   0.10 m pad and skip slab-vs-other-object overlap.
2. **P1, separate bug, different symptom.** `pi05_tasks.py:195`
   `top_z = min(bottom_z) - 0.02` = 0.7435, which is **35-62 mm below the real
   desk** (background z-mode 0.775-0.795; member-bottom median 0.7927) and is
   set by obj_11, which sits on a genuinely lower adjacent surface. Consequences:
   the arm is bolted 4.2 cm under the desk (`:197`), and `region.zlo` (`:351`)
   lets a placed object sink into the visible desk yet still score.
3. **P2.** `simany/assets/factory_align.py` lifts assets above their own local
   surface: bottles +27..48 mm, mug +12.5 mm (the mug's *detected* bottom is
   within 0.7 mm of the surface, so this is alignment, not detection).

### Relocating one object safely (recipe that worked)
Use a **variant output dir**, never edit canonical data: `factory_hybrid`'s
frozen `trellis/aligned.json` + `.snapshot_done` silently reverts in-place edits
(omit `trellis/` and `trellis_mesh.ply` from the variant and that branch is
skipped). Real copies are only ~110 KB (JSON/urdf + collision parts); symlink
`clean_background.ply`, `trellis_gs.ply`, `mesh_sim.*`.
**Both files must change, for different consumers:** `objects/obj_XX/aligned.json`
`T` drives the MuJoCo body pose, mesh scale, slab z and the composite splat pose;
`objects/objects.json` `aabb` drives the slab **xy** and ALL pi05 geometry
(`_pick_table`, `_place_robot`, `reachable`, `_qualifier`, region). Editing `T`
alone leaves base pose/reach/region/slab at the old spot. Then
`export_mjcf --test` and `pi05_tasks`, both with **SIMANY_SCENE exported
explicitly** — `pi05_tasks.py:253-255` setdefaults it from the dir name, so a
variant dir yields a nonexistent scene and silently falls back to a synthetic
camera. `scripts/slurm/pi05_closedloop.sbatch` hardcodes `outputs/${s}_factory`
and cannot drive a variant; call `pi05_eval` directly.
CompositeObs needs no change: it poses objects from the LIVE MuJoCo body pose
(`pi05_render.py:86` -> `pi05_env.py:96-98`); only scale comes from aligned.json.

### RESULT: mug moved onto the mouse pad — best demo so far
The mouse pad is GT `segments_anno.json` objectId 74 (`mousepad`), **absent from
objects.json**, so it exists only in the background — meaning `pi05_tasks` can
never emit "put it on the pad" as a *goal region*. obj_03 (black Fujitsu) is the
mouse on the pad; obj_02 is on bare desk.
`T` [4.6345, 2.6555, 0.8188] -> **[3.7136, 2.83, 0.82336]**. obj_09 settle drift
**42.9 -> 4.41 mm**; obj_03 14.6 -> 11.1 mm (no regression). Suite regenerates to
base [3.868, 2.217] yaw 103.3 deg, ext_cam DSC01618, tasks {obj_03, obj_09}, and
the mug's bearing improves +6.6 -> **+0.8 deg** at d=0.632 m.
**2/2 episodes grasp+lift (0.50 each)**, ee_dist_min 4.57/5.06 cm, no wrong
grasps — the first reproducible success (c50 canonical was 4/10 nonzero).
Clip: `outputs/pi05_runs/demo_final/c50_mug_on_mousepad_720p.mp4`
(crop `768:432:270:15` of the 720p ext pane, upscaled).
Two predicted artefacts did NOT materialise at this camera: obj_03's fill is the
scene's worst hole (901 gaussians, median luminance 0.737 on a 0.193 pad, 4.74 dB
on DSC01614) but reads dark here and is largely covered by obj_03's own mesh; and
the feared ~11 mm float (GT pad top 0.8055 vs gsplat pad median 0.7942) is not
visible. Note c50's shipped `sim_export/scene.xml` (Jul 22) is STALE — pre-rename
`meshdir=.../SimFoundry/...` and an older floor z; regenerate before trusting it.

## 2026-08-04: support-shim fix LANDED (export_mjcf.py) — physics win, score-neutral
Replaced the per-object slab scheme with (a) heights AND footprints from the
**CoACD collision hulls at full vertex resolution** and (b) **private collision
channels**: objects carry bit 0 + their channel bit, a shim carries ONLY the
channel bit, so by MuJoCo's `(contype1 & conaffinity2) || (contype2 &
conaffinity1)` rule a shim is invisible to every other object and to the robot.
Greedy colouring guarantees two objects sharing a channel are >= SAFE=0.05 m
apart; c50d2d1d42 needs 7 channels, worst case over all exported scenes is 11,
well inside 31 mask bits. Deleted the dead `cluster_heights()` (a duplicate of
`pi05_tasks._cluster`; its tol=0.06 single-link chaining merges a 76 mm spread
into one "level" and would drop objects up to 74 mm — measured stable 7/16 ->
3/16, so per-level boxes are NOT the answer here).
Why hulls, not more visual verts: CoACD inflates hulls past the input surface —
7 of obj_06's parts sit 76-93 mm BELOW its visual mesh bottom. The old code read
a ~500-point strided subsample of the ~20k-vertex VISUAL mesh; all 20k visual
verts would still leave obj_06 92 mm out.
**Measured on c50d2d1d42:** median settle drift 42.6 -> **23.8 mm**, stable<3cm
7/16 -> **9/16**, max t=0 penetration 97.8 -> **8.5 mm**, objects penetrating
10 -> **2**, obj_09 mug **42.9 -> 4.8 mm**.
Regressions to know: obj_05 (a floor-standing box misassigned to the table
cluster, bottom 0.4747) 329 -> 931 mm; obj_08 5.5 -> 21.6; obj_04 7.3 -> 14.8;
obj_17 46.5 -> 57.6. `exclude_objects` changes `['obj_05','obj_06','obj_13']` ->
`['obj_05']`, so obj_13 (a bottle 20 cm from the mug) and obj_06 become
physically present — a real occlusion/clutter change for every episode.

### Policy effect: NEUTRAL, and beware the n=1 baseline trap
A first A/B looked like a regression (0.125 vs the recorded 0.500) — **that was
an artifact**: the 0.500 baseline `hi_DSC01594` is **n=1 with no jitter**, while
the A/B ran 2 episodes with `--jitter 0.02`. Matched 3-way at 2 eps / jitter
0.02 / 32 s / DSC01594 pinned:
| condition | mean | ep0 / ep1 |
|---|---|---|
| unpatched | 0.125 | grasp / none |
| patched, obj_06+13 forced out (isolates seating) | 0.125 | grasp / none |
| patched, obj_06+13 present | 0.125 | grasp / none |
Identical. The fix costs nothing and the clutter change costs nothing; ep0 ee
even improved 0.0744 -> 0.0393. **Always match --episodes/--jitter before
calling an A/B**, and pin the camera (this scene spans 0.00-0.50 on camera
choice alone).
### THE SAMPLE-SIZE TRAP — read this before running any A/B here
Pooled over **all 47 c50 composite episodes** after the geometry fix: nonzero
rate **13/47 = 0.28**, mean staged score **0.106**, and per-run means span the
whole range **0.000 … 0.500**. Episode-to-episode variance dominates every
effect measured in this project so far. Required sample, normal approx, 80%
power, alpha 0.05, from a 0.28 base rate:
| effect to detect (abs. change in nonzero rate) | n per arm |
|---|---|
| +0.15 | **158** |
| +0.25 | **60** |
| +0.40 | **24** |
So **n=2 is pure noise and n=5 is barely a hint.** Three claims made in this
session off n<=2 were later falsified by n=5:
1. "mug-on-mousepad is jitter-robust, 2/2 grasp+lift, 4x better" — at n=5 it is
   **0.150 (2/5 nonzero)** vs the ORIGINAL position's **0.200 (2/5)**, Fisher
   p = 1.000. Relocating the mug bought **nothing** measurable. The 2/2 was luck.
2. "the slab fix caused a regression (0.125 vs 0.500)" — the 0.500 baseline was
   n=1 without jitter; matched, all three conditions are 0.125.
3. "success is a position x camera interaction" — built on the same n=2 cells;
   it does not survive n=5 either.
What IS solid (measured on physics, not on policy score, so no sampling issue):
the shim fix's penetration/drift numbers, and the geometry fix's
`corr(t=0 penetration, drift)` causal tests. Prefer physics metrics for
verifying sim changes; treat policy score as needing n>=24 even for large
effects. The mug-on-mousepad **clip is still a real, valid clip** — that episode
happened — it just is not evidence that the relocation helps.

## Fix (LANDED 2026-08-01, simany/robot/pi05_tasks.py)
`_bearing_off`/`_in_workspace` (radius 0.25-0.80 m AND +-60 deg frontal cone),
`best_yaw()` searches headings per base candidate, `_workspace_quality()`
prefers the 0.35-0.65 m sweet annulus (first version without it pushed targets
to 0.74-0.78 m near full extension — count-based scoring can't tell "barely"
from "comfortably" reachable). Old suites backed up in
outputs/_pi05_suites_backup_20260722/. New suites: 13 tasks (was 16), 0/13
beyond +-60 deg or 0.70 m, 9/13 in the sweet spot. Suite regen:
`python -m simany.robot.pi05_tasks --out-dir outputs/<scene>_factory` per scene.

## RESULT: the fix works
- geofix_raster (664106, 13 eps): first nonzero episode ever (d755 cup,
  grasp=True, 0.25, ee_dist_min 5.6 cm). Distribution shift is the real
  signal: median ee_dist_min 0.346 -> 0.129 m, best 0.018 m vs prior-best
  0.100 m, Mann-Whitney p=0.0043.
- geofix composite (664121, 3 eps c50): **first LIFT ever** — obj_09 mug
  grasp+lift score 0.50, ee_dist_min 2.7 cm, in the PHOTOREAL composite.
  Video: outputs/pi05_runs/geofix_c50d2d1d42_composite/
  c50d2d1d42_factory__obj_09_to_region_ep0.mp4.
- Co-trained ckpt droid_pi05_jointpos_with_web_and_sim/80000 downloaded
  (12 GB, anonymous GCS HTTPS; gsutil needs reauth). norm_stats lives at
  assets/physical-intelligence/droid — symlinked to assets/droid.
  pi05_serve.sh now takes SIMANY_PI05_CKPT=<subpath> and self-manages the
  /scratch copy. Bucket also has _4x/_40x variants and _nocotrain control.
- 2026-08-01 infra: env.sh hardcoded g++-12 (absent on hala; now probes),
  and ~/.bashrc's TORCH_CUDA_ARCH_LIST=9.0+PTX leaks through --export=ALL
  changing the torch-extension hash -> gsplat rebuild; pi05_closedloop.sbatch
  now pins "8.6;9.0+PTX" unconditionally.
