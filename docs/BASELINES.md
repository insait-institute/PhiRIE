# Baselines: what we have, what we owe, what we should cite

Setting: ScanNet++ v2 `nvs_sem_val`, 50 scenes. Resources: A6000-class GPUs.
**No real robot, no DROID/YAM/AgiBot hardware, no LiDAR, no panoramic rig,
no re-scans of the same room.** That constraint decides most verdicts below.

SimAny spans seven sub-problems, each with its own literature:

| | sub-problem |
|---|---|
| A | real-to-sim / image-to-sim scene generation |
| B | open-vocabulary 3D instance discovery in real scans |
| C | image-to-3D asset generation |
| D | Sim(3) registration of a generated asset into a metric scene |
| E | object removal + inpainting in 3D Gaussian splats |
| F | physical parameters, collision geometry, plausibility evaluation |
| G | simulator export + photoreal rendering of simulator state |

---

## 0. Two corrections to the previous version of this document

**(1) The "SAM3D" conflation.** The old plan listed *"SAM3D
(segment-anything-3D objects) — SimFoundry paper's own baseline (F1
0.66–0.71)"*. Those are two different papers:

- `\cite{sam3d}` = **SAM3D: Segment Anything in 3D Scenes**
  ([2306.03908](https://arxiv.org/abs/2306.03908), 2023) — 2D-mask lifting,
  correctly cited in our Related Work as a **discovery (B)** ancestor.
- SimFoundry's actual baseline is **SAM 3D: 3Dfy Anything in Images**
  ([2511.16624](https://arxiv.org/abs/2511.16624), Meta) — an **image-to-3D
  generator (C)**. Verified at line 1048 of the prior-work text
  (fetched arXiv source, not included in this repository).

`refs.bib` now carries both, as `sam3d` and `sam3dgen`.

**(2) Those numbers are not reusable.** SimFoundry's F1 is **@10 mm on 12
staged YCB tabletop scenes with FoundationPose quasi-GT** (verified: *"F1-Score
(with threshold 0.01 meters)"*). Ours is **F1@20 mm on ScanNet++ instance
submeshes**. Different threshold, different scenes, different reference. Never
put them in one table.

---

## 1. Already measured (in-tree, on our own data)

| sub-problem | baseline | result | where |
|---|---|---|---|
| B | MaskClustering, same SAM3 masks, matched 28-frame budget | F1@0.25 **0.605** vs 0.571; F1@0.5 **0.419** vs 0.286 | `agents/baselines/maskclustering.py` |
| C | TRELLIS vs ReconViaGen, identical registration | 0.708 → **0.783** pooled F1@20 mm via residual gate; all 18 collapses caught | `agents/assets/factory_hybrid.py` |
| E | FlashSplat optimal mask assignment | union IoU **0.649** vs our final removal set (recall 0.87) | `agents/baselines/flashsplat.py` |
| E | LaMa vs Qwen-Image-Edit as fill supervisor | keyboard-desk **34.3 vs 23.9 dB**; elsewhere 19.4 vs 20.5 | `agents/edit/inpaint_qwen.py` |
| G | SceneSplat background (reconstruction ceiling) | twin 28.32 dB vs bg 28.89 dB on the official DSLR test split | `agents/eval/factory_eval_render.py` |
| A | ablation ladder A/B/C_GT/C_SAM3/D | see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | `run/slurm/ablation_rowC_*.sbatch` |

Caveat carried from the paper: the MaskClustering and FlashSplat comparisons
and the verification battery are **single-pilot-scene**.

---

## 2. The three papers under review

### 2.1 SimFoundry — [2606.28276](https://arxiv.org/abs/2606.28276), NVIDIA/Stanford/GaTech/UT/Toronto, Jun 2026
**Verdict: (a) directly comparable. Our closest competitor, and the name collision that prompted the rename.**

Single smartphone video → per-object mesh/scale/pose, **articulation (part seg
+ joints + URDF)**, CoACD, VLM mass/friction, PyBullet + Isaac Lab export, 3DGS
background, plus object/scene/task **digital cousins** and MimicGen demos.

What matters for our positioning, all verified against the full text of
arXiv:2606.28276:

- **Their headline geometry column is human-tuned.** Table L.2 reports
  `SimFoundry Tuned (3 min/Obj)` at F1 0.99/0.97/0.93 (easy/med/hard) vs
  zero-shot 0.92/0.87/0.81, and the text concedes robotics scenes were
  *"tuned with a human operator, requiring a few minutes worth of iteration"*
  (line 1200). Appendix K is a drag-object-onto-pointcloud GUI. **This is the
  strongest available support for our annotation-free / interaction-free
  claim** — provided we compare against their *zero-shot* column, not the
  headline.
- **Their automatic background beats their manual one.** Table L.6, 5 dorm
  scenes: manual physical clearing 12.91 dB / 0.497 SSIM / 0.549 NCC;
  automatic two-pass video inpaint + splat retrain **15.29 / 0.605 / 0.749**.
  Our intro currently implies inpaint-and-retrain is a *quality* compromise.
  It is not — it is a **cost** compromise (~90 min/scene by their own number)
  and it yields **no per-object removal sets**, so nothing can be individually
  moved. Re-scope the criticism accordingly.
- **Timing**: 19.98–67.13 min/scene, ~5 min/object, RTX 3090. The honest
  per-object comparison is therefore ~6.5× (their 5 min vs our 44 s), across
  different GPUs — *not* the 29×/43× figure, which is against HoloScene.
- **Policy**: Pearson 0.911, MMRV 0.018, >0.59 above PolaRiS. We cannot
  reproduce this, and must scope our borrowed citation to *tabletop* twins.
- Tabletop-focused by their own stated limitations; objects enumerated by
  Gemini on one frame. **CORRECTED 2026-08-31: an official code release now
  exists** -- github.com/NVlabs/SimFoundry, created 2026-07-15 (after this
  document's original audit), Apache-2.0, "Initial open-source release: V0
  rigid-body and articulation generation" pushed 2026-08-14, example
  scenes/assets added 2026-08-26, 281 stars/19 forks as of this check, not
  archived, actively maintained. This changes the correct `status` for
  SimFoundry-reproduction in baselines/release_status.yaml from a
  from-description reimplementation toward "run the real thing" -- see
  that file and docs/ICRA_RESEARCH_CONTRACT.md for the current call on
  whether/how far integration has actually gone. What the release does
  NOT include (per its own README, "Not yet released"): the sim-to-real
  policy training / data-generation pipeline behind its headline
  Pearson-0.911-vs-PolaRiS policy numbers above -- so that comparison
  claim in this document still cannot be independently reproduced even
  now; only the scene-reconstruction pipeline (their "Pipeline A/B/C") is
  public. Requirements per their README: Linux+CUDA GPU, ~250GB disk, HF
  account with gated access to facebook/sam3 (this repo already has this),
  facebook/dinov3-vitl16-pretrain-lvd1689m, and briaai/RMBG-2.0 (neither
  confirmed for this account yet), plus a GCP project with Vertex AI or a
  Gemini API key for their VLM stages (this account's gcloud has no active
  project set as of this check).

**Owed experiment (highest value in this document):** reimplement their
*automatic background route* — two-pass video inpainting → DepthAnything3 poses
→ depth-supervised splat retrain — on 1–2 ScanNet++ scenes and score it on the
**official DSLR test split**. That is the direct competitor to our
Gaussian-native removal, measured on *our* protocol. ~90 min/scene by their own
measurement, so roughly 3 GPU-hours for a two-scene comparison.

### 2.2 WANDA / "Worlds in One Demo" — [2607.13154](https://arxiv.org/abs/2607.13154), CMU, Jul 2026
**Verdict: (b) cite and qualitatively contrast. Not benchmarkable here.**

One teleoperated RGBD demo per task (AgiBot G1 + PICO VR) + **5–30 min of human
XMem clicking and contact/segment labelling per task** → MAtCha 2DGS splats,
BundleSDF object meshes with tracked 6-DoF, Marble-generated worlds,
factorized-rendered observations. **Explicitly render-only: no dynamics, no
URDF, no collision, no engine export.**

Numbers: BEHAVIOR π0.5 Q-score 16.67 from 1 demo (1,360 generated) vs
3.33 / 11.11 / 62.22 for 20 / 50 / 200 teleop demos (~50× sample efficiency);
real AgiBot, 5 tasks × 10 trials, **54.8%** vs **0.0%** from one real demo, and
15.7% without Corrective State Expansion; cross-embodiment Linearbot zero-shot
40%; 3.04 GPU-h per data-hour (6.4× less than MoMaGen). Code "coming soon",
data released on HuggingFace.

Why it matters despite being unrunnable: (i) it is the **second** system
showing inpaint-then-reconstruct backgrounds work well, reinforcing the
re-scope above; (ii) it obtains real robot transfer from **single-photo
generated worlds**, which a reviewer could read as contradicting our row-D
cliff. It does not — they need a *navigable, renderable* world, we need a
*metric replica* — but the paper should say so explicitly rather than leave it
to the reader.

### 2.3 Lumera — [2607.20889](https://arxiv.org/abs/2607.20889), Tsinghua et al., Jul 2026
**Verdict: (c) orthogonal, must cite.**

Single **game-engine** image → oriented boxes + labels + **engine-native
parametric lights** (x, y, z, r, g, b, I) + SAM-3D per-object meshes → editable
Blender/UE5 scene. Dataset Lumera-2K: 2,513 UE5 projects, 63 M instances,
102.6 K lights. **No physics, no collision, no robot export, no splats.**

Two uses for us. It turns our *"lighting is baked into the splats"* limitation
into a benchmarked target with concrete numbers — light P/R/F1
0.218 / 0.201 / 0.209 @0.5 m, intensity log₁₀ MAE 0.431 (≈2.7×), Pearson
0.628, i.e. **the problem is far from solved**, which is a fair way to cite a
limitation. And its weak single-image box numbers (mAP 0.1141; zero-shot
SpatialLM at 0.0000) independently corroborate our row-D finding that one
image is not enough.

---

## 3. Newly found baselines, by sub-problem

Verification legend: ✅ metadata confirmed on the arXiv abstract page;
⚠️ ID is real but metadata was not cross-checked — **verify before citing**.

### A. Real-to-sim scene generation

| system | id | verdict |
|---|---|---|
| **SimRecon** ✅ | [2603.02133](https://arxiv.org/abs/2603.02133), CVPR 2026 | **Closest new competitor.** Compositional real-video → SimReady assets, and it **already reports ScanNet numbers**. Watch for code; if released, a Tier-1 head-to-head. |
| **HoloScene** ✅ | [2510.05560](https://arxiv.org/abs/2510.05560), NeurIPS 2025 | Already our efficiency reference (8.3 h/scene). Also our **physics-metric** reference — see F. Single-video input, so ScanNet++ DSLR works. |
| **MetaScenes** ✅ | [2505.02388](https://arxiv.org/abs/2505.02388), CVPR 2025 | Already cited. Sharpen the contrast: it *does* annotate mass/material/bounciness from foundation models, but uses **human ranking and human asset placement**, and **reports no stability metric**. Precisely our wedge. |
| **Image2Sim** ✅ | [2607.05765](https://arxiv.org/abs/2607.05765), Jul 2026 | Posed **RGB-D** sequences → 20 K interactive scenes, feed-forward. Input shape matches ours exactly. Release status unknown. |
| **EmbodiedGen V2** ✅ | [2607.07459](https://arxiv.org/abs/2607.07459), Jul 2026 | Generative multi-room worlds, not scan twinning. Cite as the generative alternative. |
| **Video2Game** ✅ | [2404.09833](https://arxiv.org/abs/2404.09833), CVPR 2024 | **Public code**, posed images, indoor. Cheap "mesh export" contrast. |
| **SimuScene** ✅ | [2606.03994](https://arxiv.org/abs/2606.03994), Jun 2026 | Single image → sim-ready, with physics inside the reconstruction loop. Feasible row-D-style comparison, no robot needed. |
| GASE ✅ | [2606.17520](https://arxiv.org/abs/2606.17520) | Already cited. Now known to need **panoramic camera arrays** → not runnable here. |

### B. Instance discovery
The MaskClustering comparison stands. Worth adding as *cited* alternatives:
OpenMask3D, Open3DIS, SAI3D, Any3DIS (already cited). Cheap and useful:
**zero-shot SpatialLM** as a single-image proposal baseline for row D — Lumera
measures it at mAP 0.0000 on their data, so it is a weak but legitimate
reference point.

### C. Asset generation
Add **SAM 3D** ([2511.16624](https://arxiv.org/abs/2511.16624)) to the
`V_mesh` slot — it is SimFoundry's own generator baseline, so running it in our
hybrid harness makes our slot-selection result directly legible to their
readership. Hunyuan3D 2.1 remains the other obvious slot-in.

### D. Registration
No new external baseline found; the synthetic partial-view suite
(`tests/test_align_synthetic.py`) remains the right instrument. It
reproduces post-refactor: **zero 180° flips**, misses confined to yaw-grid
local minima on cones/capsules.

### E. 3DGS removal and inpainting
Already cited: GaussianEditor, FlashSplat, InFusion, RoboPearls, TRAN-D.
Add for completeness: GScream, Gaussian Grouping, SPIn-NeRF.
**Newly relevant:** *DiffusionHarmonizer*
([2602.24096](https://arxiv.org/abs/2602.24096), NVIDIA) — a single-step
enhancer that harmonizes inserted objects into imperfect 3DGS renders. Exactly
the artifact class our composite twin exhibits; cite as complementary and as
future work for closing our 0.57 dB gap.

### F. Physics — **the weakest-covered area, and the cheapest to strengthen**

Our design (CoACD + VLM mass/friction + settle) is *method-identical* to
SimFoundry's and *property-identical* to MetaScenes'. **Neither reports a
physics metric.** So the claim is not the mechanism — it is that we
**measure** it, at 50 scenes / ~1000 instances. Two published protocols exist,
and both are on ScanNet++:

- **PhyRecon** ✅ [2404.16666](https://arxiv.org/abs/2404.16666), NeurIPS 2024 —
  *Stability Ratio*, drop simulation in **Isaac Gym**. ScanNet++: RICO 26.43,
  ObjectSDF++ 25.28, **PhyRecon 78.16**.
- **HoloScene** ✅ [2510.05560](https://arxiv.org/abs/2510.05560) — *Stable%* in
  **Isaac Sim**, plus a penetration-failure count. ScanNet++: ObjSDF++ 81.6,
  PhyRecon 67.3, DP-Recon 20.0, **HoloScene 93.9**; penetrations 76 / 33 / 102 / **6**.

**Our 77.2% lands right on PhyRecon's 78.16% on the same dataset** — a free,
immediately legible positioning sentence. But the two papers disagree by
>50 points on the *same* baseline (ObjSDF++ 25.28 vs 81.6), which proves the
protocol is not standardized. **We must state simulator, threshold and duration
explicitly and report a threshold sweep**, or the comparison is attackable.

Feasible, non-LLM physics baselines (static input, no video, no robot):

| system | id | why |
|---|---|---|
| **VoMP** ⚠️ | [2510.22975](https://arxiv.org/abs/2510.22975), NVIDIA | Predicts **density** per voxel from any renderable 3D representation → `mass = ∫ρ dV` is directly comparable to our LLM mass. Pretrained weights. **The single best runnable F-baseline.** |
| **PhysQuantAgent** ⚠️ | [2603.16958](https://arxiv.org/abs/2603.16958), IROS 2026 | Direct competitor to our LLM mass annotator, and ships **VisPhysQuant** (real objects, measured GT mass). Their finding that *scale prompting* is what helps independently validates our "category + metric dimensions" prompt. |
| **PhysSplat** ⚠️ | [2411.12789](https://arxiv.org/abs/2411.12789), ICCV 2025 | The other static-input, MLLM-driven, scene-level property estimator. Closest in spirit to ours. |
| **Pixie** ⚠️ | [2508.17437](https://arxiv.org/abs/2508.17437) | Static multi-view → material fields, video-free. Complementary axis (elastic moduli, not mass). |
| **PhysGaussian** ✅ | [2311.12198](https://arxiv.org/abs/2311.12198) | **Not an estimator** — params are hand-set in a JSON. Cite as motivation: physics-on-3DGS presupposes exactly what we produce. |

Collision geometry: **V-HACD** is the mandatory one-afternoon ablation against
CoACD. Newly relevant: *Learning Convex Decomposition via Feature Fields* ⚠️
([2603.09285](https://arxiv.org/abs/2603.09285), NVIDIA, Mar 2026) —
feed-forward, **consumes Gaussian splats directly**, and benchmarks against
V-HACD *and* CoACD. A plug-in baseline if weights are released.

Scene-level plausibility: **PhyScene** ✅
([2404.09465](https://arxiv.org/abs/2404.09465), CVPR 2024 Highlight) defines
`Col_obj`, `Col_scene`, `R_out`, `R_reach`, `R_walkable`. All five are
computable from what we already have, and would turn 50 scenes into a
scene-level result rather than only a per-object one.

### G. Simulator export + GS rendering

| system | id | verdict |
|---|---|---|
| **3DGRUT / NuRec** ✅ | [2412.12507](https://arxiv.org/abs/2412.12507), CVPR 2025 oral | **Apache-2.0, ships a native ScanNet++ dataloader, exports USD ParticleField.** Lowest-friction path to an Isaac-Sim-renderable asset; best immediate win in this section. |
| **DISCOVERSE** ✅ | [2507.21981](https://arxiv.org/abs/2507.21981), IROS 2025 | MIT, 3DGS + MuJoCo, splats as interactive scene nodes. Closest architectural match to our render path. |
| **SplatSim** ✅ | [2409.10161](https://arxiv.org/abs/2409.10161), ICRA 2025 | Already cited. Render half reproducible; sim2real half is not (no UR5). |
| **GSWorld** ✅ | [2510.20813](https://arxiv.org/abs/2510.20813), ICRA 2026 | Public code + HF assets, but tabletop, and its real2sim step needs **manual ICP alignment** — another annotation-free contrast point. |
| **NavGSim** ✅ | [2603.15186](https://arxiv.org/abs/2603.15186) | Hundreds of m² indoor GS + mesh-free collision extracted from Gaussians. Closest published "whole indoor scene GS simulator". No code → cite only. |
| **RoboSnap** ✅ | [2607.06699](https://arxiv.org/abs/2607.06699), Jul 2026 | Single RGB → collision-aware assets + 3DGS visual layer, fully automated. Strongest 2026 single-image comparison, and evaluable without a robot. |

Ruled out for lack of hardware: GASE (panoramic rig), Ψ-Map (LiDAR), IGFuse
(needs rearranged re-scans), ExoGS (AirExo-3), LEGS (Unitree G1), GaussTwin
(Franka), Real2Sim-Eval (real rollouts), ReaDy-Go / RoboGSim / Robo-GS
(code release unverified).

### Articulation (we do not model it — cite and move on)
URDFormer ✅ [2405.11656](https://arxiv.org/abs/2405.11656) (RSS 2024) ·
Articulate-Anything ✅ [2410.13882](https://arxiv.org/abs/2410.13882) (ICLR 2025) ·
Real2Code ✅ [2406.08474](https://arxiv.org/abs/2406.08474) (**venue disputed —
evidence points to ICLR 2025, not CoRL 2024; verify**) ·
AutoURDF ✅ [2412.05507](https://arxiv.org/abs/2412.05507) (CVPR 2025,
annotation-free, closest in spirit) ·
URDF-Anything ✅ [2511.00940](https://arxiv.org/abs/2511.00940) (NeurIPS 2025) ·
ArtGS ✅ [2502.19459](https://arxiv.org/abs/2502.19459) (3DGS-native).

---

## 4. Prioritized worklist

**Tier 1 — run these; feasible with what we have**

1. **Re-anchor stability as Stability Ratio** against PhyRecon's and
   HoloScene's published ScanNet++ numbers. *Zero GPU cost — we already have
   77.2%.*
2. **SR threshold + duration sweep**, plus a **rotational** drift criterion
   (both published protocols omit rotation, and it catches tip-overs that
   translate < 3 cm). Pre-empts the strongest available attack. *~2 GPU-h.*
3. **Penetration count and depth** (object–object, object–floor; at spawn and
   at rest), matching HoloScene's failure taxonomy. *Cheap in PyBullet.*
4. **V-HACD vs CoACD** ablation: part count, volume inflation, surface error.
   *~1 GPU-h.*
5. **Density sanity check** — ρ̂ = m̂ / V against a material-density table;
   report the fraction of physically impossible densities. *CPU-only, and no
   competitor reports it.*
6. **SimFoundry's automatic background route** on 1–2 scenes, scored on our
   official-split protocol. *~3 GPU-h. Highest scientific value.*
7. **SAM 3D in the `V_mesh` slot**, so our hybrid result speaks directly to
   SimFoundry's readership. *~1 GPU-h on the pilot scene.*
8. **VoMP** as a learned, non-LLM mass baseline. *Pretrained, single GPU.*

**Tier 2 — cite and contrast, do not run**
SimFoundry (policy correlation), WANDA, SimRecon (until code lands),
MetaScenes, HoloScene, PhysQuantAgent, PhysSplat, NavGSim, RoboSnap, GSWorld,
PhyScene, DiffusionHarmonizer, the articulation cluster.

**Tier 3 — mention only**
Lumera, EmbodiedGen, PhysGaussian, Vid2Sim (**two different papers share this
name** — check which you mean), driving/outdoor GS systems.

---

## 5. Efficiency table (corrected)

| system | input | per-scene cost | notes |
|---|---|---|---|
| **SimAny (GT-driven)** | posed RGB + mesh + splat + GT instances | **11.6 min** (9.7 A6000-h / 50) | 44 s/object |
| **SimAny (automatic)** | posed RGB + mesh + splat | **17.0 min** (14.1 A6000-h / 50) | 47 s/object |
| SimFoundry | one RGB video | 19.98–67.13 min (RTX 3090) | ~5 min/object; **+ ~90 min** for the automatic background |
| HoloScene | one video | 8.3 h | GPU model unconfirmed |
| SceneSplat | posed RGB | splat training only | no sim-ready assets |

The **29× / 43×** multipliers in the abstract are against **HoloScene** and
must name it. Against SimFoundry the honest figure is **~6.5× per object**,
across different GPUs.
