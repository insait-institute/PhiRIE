# SimAny — what the contribution actually is

> Naming note: this project was developed under the internal codename
> *SimFoundry* (it began as a reproduction of arXiv:2606.28276) and briefly
> as *PhiRoom*. Both names are retired. **SimFoundry now refers only to the
> prior NVIDIA/Stanford system we cite and compare against**; SimAny is ours.

## One sentence

Given only posed RGB images of a real indoor scene plus its off-the-shelf
reconstruction (a mesh and a 3D Gaussian splat), SimAny produces a
photorealistic, editable, simulation-ready digital twin **fully
automatically** — no semantic annotations, no per-object prompts, no clicks —
and **verifies its own output without any ground truth**.

## The three claims we defend

### 1. Dual-use discovery: one training-free stage feeds both branches

The load-bearing architectural idea. A single open-vocabulary instance
discovery stage (SAM3 masks → occlusion-aware raycasting onto the scene mesh →
voxel-IoU union-find merging across views) produces mesh vertex sets that are
consumed twice:

- a **physics branch** that generates, registers, and physically annotates one
  rigid asset per object;
- an **appearance branch** that *erases the same object* from the background
  splat.

Why the second consumer matters: prior systems either leave the original
object baked into the background (so the twin double-renders as soon as an
object moves) or re-train the background from inpainted video at large cost.
Discovery emits the same contract that GT annotations would, which is what
lets every downstream stage run byte-identically in benchmark (GT-driven) and
automatic modes — that is what makes the ablation ladder below meaningful
rather than a comparison of two different pipelines.

### 2. Transparency-robust Gaussian-native removal

A glass bottle's splat is a diffuse, low-opacity cloud whose Gaussian centers
sit far from any physical surface, so both geometric selectors miss it. The
removal set unions three selectors, and the third is the contribution:

1. Gaussians within 3 cm of the instance surface;
2. Gaussians within 2.4 cm of the *registered asset* surface (covers volume
   the scan never captured);
3. **multi-view mask vote** — a Gaussian in the dilated bounding region is
   removed if its center projects inside the object's 2D mask in ≥2 of its
   views. Opacity-agnostic by construction.

Holes are then filled on a MAD-trimmed support plane with normal-oriented
Gaussian disks (5 mm grid), colour-initialized from the object's primary
inpainted view, and refined by differentiable rasterization against
**per-frame composited** targets. Two details are load-bearing and were
learned the hard way: supervising on primary views only (cross-frame diffusion
outputs are mutually inconsistent; mixing them drives the optimum to a blotchy
mean) and compositing every object's patch into one target per frame rather
than last-writer-wins (otherwise neighbouring objects get baked in).

### 3. Supervision-independent verification

Because a fully automatic system will sometimes fail silently, we ship a
battery that scores the *edited* scene with no annotations at all: alpha
coverage inside the removal mask, support-plane depth residual, residual
re-detection with SAM3 using the object's own prompt, and held-out-view
consistency. Failures become a precise re-processing worklist rather than
silent corruption. On the pilot scene it independently rediscovers exactly the
failure modes the literature also finds hardest (transparent bottles) plus one
we did not anticipate (box-scale holes outgrowing the support-plane fill).

## Headline numbers (all 50 ScanNet++ v2 validation scenes)

| | GT-driven | Automatic (no GT) |
|---|---|---|
| instances | 789 | 1,082 |
| yield (tier A+B) | 57.9% (76%\*) | 62.0% |
| geometry F1@20 mm | **0.783** (hybrid) / 0.708 single-view | 0.582 (vs matched GT) |
| drop-test stable | 77.2% | 75.4% |
| per scene / per object | 11.6 min / 44 s | 17.0 min / 47 s |

\*excluding one library outlier scene (200 shelved books, 189 correctly rejected).

- **Cost**: 9.7 A6000-hours for 50 scenes GT-driven, 14.1 automatic. Against
  HoloScene's reported 8.3 h/scene that is 43× / 29× faster (their GPU model
  is unconfirmed, so treat the multiplier as indicative).
- **Appearance**, official DSLR test split (frames the splat never trained on):
  composite twin 28.32 dB vs SceneSplat background 28.89 dB — a 0.57 dB gap;
  automatic twin 27.52 dB. On the pilot scene, compositing onto the *cleaned*
  background reads 28.39 dB vs 27.55 clean-only.
- **Discovery vs MaskClustering**, same SAM3 masks, matched 28-frame budget:
  F1@IoU0.25 0.605 vs 0.571, F1@IoU0.5 **0.419 vs 0.286** (1.5×). The honest
  reading: at the loose threshold density matching closes most of the gap; what
  survives is localization precision. Single scene.
- **Hybrid generation**: residual-gated choice between single-view (TRELLIS)
  and multi-view (ReconViaGen) lifts pooled F1@20 mm 0.708 → 0.783 over 457
  objects, picks multi-view for 53%, and catches **all 18** multi-view
  collapses (F1 < 0.1) with no category rules — within 0.006 of the oracle
  that always picks the better asset.

## The result we did not expect (and the most interesting one)

The ablation ladder removes ground truth one rung at a time:

| | instances | yield | F1@20 mm |
|---|---|---|---|
| A: GT segments + GT mesh | 789/50 | 58.0% | 0.708 |
| B: SAM3 + GT mesh | 1,082/50 | 62.0% | 0.582 |
| C_GT: GT segments + splat-derived mesh | 678/49 | 56.0% | 0.693 |
| C_SAM3: SAM3 + splat-derived mesh | 930/49 | 59.0% | 0.553 |
| D: single-image zero-shot | 389/40 | **17.2%** | 0.309 |

Replacing ScanNet++'s laser-scan mesh with one TSDF-fused from nothing but the
trained splat's own rendered depth costs **2 points of yield and 0.015 F1**
(C_GT vs A). The dataset's high-fidelity mesh was never load-bearing. The
cliff is row D: dropping the reconstruction *entirely* collapses yield to
17.2%. The requirement is "have a reconstruction at all", not "have a
laser scan" — which is what makes the method applicable beyond scan datasets.

This also produced a methodological finding worth stating: scoring discovered
instances against *their own extraction* (the only annotation-free option)
reads the C_SAM3-vs-B comparison **backwards** (0.667 vs 0.630) relative to
matched-GT scoring (0.553 vs 0.582). Own-extraction scoring credits a pipeline
against its own errors; we report both and trust only the matched one.

## What we deliberately do not claim

Kept explicit so reviewers do not have to find it:

- **No sim-to-real / policy correlation of our own.** Cited from prior work,
  not re-validated on hardware. We have no robot.
- **Verification battery and the MaskClustering/FlashSplat baselines are
  single-pilot-scene**, not fleet-scale.
- **No articulation**; deformables treated as rigid; lighting baked into the
  splats, so moved objects cast no shadows.
- **Furniture is an opt-in extension**, not in the main protocol: large
  partially-observed desks defeat single-view generation and furniture-scale
  holes outgrow the support-plane fill.
- **Registration robustness** was stress-tested on the partial-cloud zero-shot
  path; benchmark numbers register against complete instance submeshes, which
  is better conditioned.
- **BEHAVIOR-1K coverage is a static feasibility check** (2.3% of 1,015
  activities vocabulary-complete, 0.6% fully groundable in objects we have
  actually generated), not a simulator run.
- **MJCF export approximates the background** with per-object static support
  slabs rather than the full scan mesh, so cross-object and object–room
  contacts differ from the PyBullet export, which does collide against the
  carved scan.

See [BASELINES.md](BASELINES.md) for how this positions against concurrent
systems, and which comparisons are still owed.
