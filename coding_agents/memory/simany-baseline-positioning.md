---
name: simany-baseline-positioning
description: "SimAny vs SimFoundry/WANDA/Lumera: verified facts that change our claims, plus two citation errors found"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0b4246c3-7101-4e38-8834-10129cb6e3ad
  modified: 2026-08-01T00:13:41.326Z
---

Verified 2026-08-01 against the full text of arXiv:2606.28276 (kept at
`docs/related/simfoundry_arxiv_2606.28276.txt` in the repo).

**SimFoundry (2606.28276) — directly comparable, our closest competitor.**
- Their headline geometry column is `SimFoundry Tuned (3 min/Obj)`
  (F1 0.99/0.97/0.93 easy/med/hard); zero-shot is 0.92/0.87/0.81. The text
  concedes robotics scenes were "tuned with a human operator" (line 1200) and
  Appendix K is a drag-onto-pointcloud GUI. **Always compare against their
  zero-shot column** — this is the best support our annotation-free claim has.
- Their F1 is **@10 mm on 12 staged YCB tabletops with FoundationPose
  quasi-GT**. Ours is @20 mm on ScanNet++ submeshes. NOT comparable; never
  put them in one table.
- Their automatic inpaint+retrain background **beats** their own physically
  cleared capture on all 7 metrics (15.29 dB / 0.605 / 0.749 vs 12.91 / 0.497
  / 0.549). So the criticism of that route is **cost** (~90 min/scene) and
  **granularity** (no per-object removal sets), NOT quality. Our intro
  currently implies quality — needs re-scoping.
- Timing 19.98-67.13 min/scene, ~5 min/object on RTX 3090. Our abstract's
  "29x faster than the closest prior system" is measured against **HoloScene**
  (8.3 h/scene) and must name it; vs SimFoundry the honest figure is ~6.5x
  per object on different GPUs.
- No code released.

**WANDA (2607.13154, CMU)** — cite/contrast only, needs an AgiBot G1 + one
teleop RGBD demo + 5-30 min human labelling per task; render-only (no URDF,
no dynamics). Second system showing inpaint-then-reconstruct works.

**Lumera (2607.20889)** — orthogonal. Single game-engine image -> editable
UE5/Blender scene with parametric lights. Cite for our "lighting is baked in"
limitation; their own light F1 is 0.209 @0.5 m, so the problem is open.

**Two citation errors found and fixed in paper/refs.bib:**
1. Two different papers are called "SAM3D". SimFoundry's baseline is Meta's
   *SAM 3D: 3Dfy Anything in Images* (2511.16624), an image-to-3D generator —
   NOT the 2023 *SAM3D: Segment Anything in 3D Scenes* (2306.03908) we cite
   for discovery. Now `sam3dgen` and `sam3d` respectively.
2. `behavior1k` cited "CoRL 2024" — no such version. Preliminary is CoRL 2022;
   extended work is arXiv:2403.09227.

**Best free win available**: our 77.2% drop-test stability is essentially
PhyRecon's 78.16% Stability Ratio on the *same dataset* (ScanNet++). Re-anchor
the metric to their published protocol. But PhyRecon and HoloScene disagree by
>50 points on the same baseline (ObjSDF++ 25.28 vs 81.6), so state simulator +
threshold + duration and report a sweep.

Full tiered survey in `docs/BASELINES.md`; exact paper edits with before/after
text in `docs/PAPER_REVISIONS.md`. Related: [[simany-rename-2026-07-27]],
[[simfoundry-repro-status]].
