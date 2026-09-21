# Paper revisions owed after reading the July-2026 concurrent work

Source: close reading of SimFoundry ([2606.28276](https://arxiv.org/abs/2606.28276)),
WANDA ([2607.13154](https://arxiv.org/abs/2607.13154)), Lumera
([2607.20889](https://arxiv.org/abs/2607.20889)). Full analysis and the
verified quotes behind each item are in `docs/BASELINES.md`.

> **Status 2026-08-01: all P1 and P2 items below have now been APPLIED** during
> the paper rewrite. This file is kept as the record of *why* each claim reads
> the way it does, and of the exact wording that was replaced — so the
> reasoning survives if a co-author questions a sentence. The "suggested"
> wording below is, in each case, approximately what is now in `root.tex`.

Priority 1 is "a reviewer who has read SimFoundry will call this out".

---

## P1.1 — Abstract: the speed multiplier does not name its referent

**Current** (`docs/paper/root.tex`, abstract):
> the fully automatic mode runs in 17 minutes per scene on a single A6000---29$\times$ faster than the **closest prior system** (11.6\,min / 43$\times$ when ground-truth segments replace discovery)

**Problem.** 29×/43× is computed against **HoloScene** (8.3 h/scene), which is
not the closest prior system. SimFoundry runs **19.98–67.13 min/scene**
(~5 min/object, RTX 3090) — comparable to us per scene, ~6.5× slower per
object. A reader who assumes "closest prior system" means SimFoundry will
conclude we inflated the number by ~5×.

**Suggested.** Name HoloScene explicitly, and give the SimFoundry comparison
separately and honestly:
> …runs in 17 minutes per scene on a single A6000---29$\times$ faster than
> HoloScene's reported 8.3\,h per scene (11.6\,min / 43$\times$ under
> ground-truth segments), and roughly 6.5$\times$ faster per object than
> SimFoundry's reported 5\,min/object, though on different GPUs.

---

## P1.2 — Introduction: the criticism of inpaint-and-retrain is factually backwards

**Current** (line ~94):
> prior systems either keep the original object baked into the background (double rendering once objects move) or re-train the background from inpainted video **at large cost**~\cite{simfoundry}.

**Problem.** As written this reads as a quality criticism. SimFoundry's
Table L.6 (5 dorm scenes) shows their **automatic inpaint-and-retrain route
beats their own manually-cleared capture on all seven metrics** —
15.29 dB / 0.605 SSIM / 0.749 NCC vs 12.91 / 0.497 / 0.549. WANDA independently
does the same thing successfully. The criticism that survives is **cost**
(~90 min/scene, their number) and **granularity** (a retrained background has
no per-object removal sets, so objects cannot be individually moved or
restored).

**Suggested.**
> …or re-train the background from inpainted video~\cite{simfoundry,wanda}.
> That route produces good backgrounds---SimFoundry reports it beating a
> physically cleared re-capture---but costs roughly 90 minutes per scene and
> yields a single monolithic background, with no per-object removal sets to
> edit, restore, or verify.

---

## P1.3 — Related Work: the novelty claim rests on a four-way conjunction

**Current** (lines ~148–151):
> \name\ is, to our knowledge, the **first system that is simultaneously** annotation-free, interaction-free, dataset-scale (50 scenes / $>$1k instances), and verified without ground truth.

**Problem.** Four-way "simultaneously first" claims invite exactly one reviewer
response. It is also now weaker than it looks: SimRecon
([2603.02133](https://arxiv.org/abs/2603.02133), CVPR 2026) is automatic and
already reports ScanNet numbers.

**Suggested.** Claim the mechanisms, which are defensible individually:
> Relative to these systems, \name\ contributes three mechanisms: a discovery
> stage whose output is consumed by both the physics and appearance branches;
> transparency-robust removal directly in the Gaussian representation; and a
> verification battery that requires no annotations. It is run at dataset
> scale (50 scenes, $>$1k instances) with no human in the loop at any stage.

---

## P1.4 — Experiments: the 28.32 dB number needs a protocol footnote

**Current** (Sec. Main Results): composite twin **28.32 dB** vs SceneSplat
background 28.89 dB.

**Problem.** SimFoundry's background numbers are **15.29 dB**. Those are
render-vs-**real-video** comparisons on unposed dorm captures; ours is
novel-view synthesis on the **official held-out DSLR test split**. Left
unqualified, this reads as a 13 dB win that we did not earn.

**Suggested footnote.**
> Not comparable to the background PSNR reported by~\cite{simfoundry} (15.29\,dB):
> that protocol scores renders against a real video of the cleared scene, while
> ours is novel-view synthesis against held-out frames of the official DSLR
> test split.

---

## P2.1 — Related Work understates SimFoundry's scope

**Current** (line ~134): described as the "modular segment–generate–register–annotate
recipe from a single video … targets tabletop workspaces and re-trains its
background splat from inpainted video."

**Missing:** articulation (part segmentation + joint prediction → URDF),
object/scene/task **digital cousins**, MimicGen demo generation, policy
evaluation (Pearson 0.911 vs real, >0.59 above PolaRiS), and **two** background
routes. Understating a competitor is a credibility risk when the reviewer is
likely to be one of its 18 authors.

**Also add the strongest honest differentiator we now have:** their headline
geometry column is `Tuned (3 min/Obj)` and the text concedes robotics scenes
were *"tuned with a human operator"* (line 1200 of the full text). Compare
against their **zero-shot** column (F1 0.92/0.87/0.81), and note that our
protocol has no tuned mode at all.

---

## P2.2 — Scope the borrowed policy-correlation citation

**Current** (Limitations, line ~632):
> Sim-to-real policy correlation is cited from~\cite{simfoundry} rather than re-validated on hardware.

**Add:** that correlation was measured on **tabletop** manipulation twins, not
room-scale scanned scenes, so it transfers to our setting only as an
indication.

---

## P2.3 — The Gaussian-splatting-in-simulators paragraph should include WANDA

The SplatSim paragraph should add WANDA's factorized rendering (compositing
rendered robot/object meshes over Gaussian backgrounds) as the most recent
instance of the pattern we adopt as infrastructure.

---

## P2.4 — Lighting limitation can now cite a benchmark

**Current** (Limitations): "lighting is baked into the splats, so moved objects
cast no shadows."

**Add** Lumera as the concurrent attempt to recover engine-native parametric
lights, *and* its numbers, which show the problem is open: light detection
P/R/F1 0.218 / 0.201 / 0.209 at 0.5 m, intensity log₁₀ MAE 0.431 (≈2.7×
error). This cites a competitor while making clear the gap is not ours alone.

---

## P2.5 — Generator-version caveat

SimFoundry's FAQ notes TRELLIS.2 claims transparency support. We use TRELLIS
v1. Sec. V-D should say so, since transparency is the failure mode we report
most.

---

## Already applied

| fix | where |
|---|---|
| `\name`: PhiRoom → SimAny, with a comment warning that "SimFoundry" in this paper always means the cited prior work | `docs/paper/root.tex:17` |
| `behavior1k` bib entry said `booktitle={CoRL}, year={2024}` — **no such version exists**; the preliminary version is CoRL 2022, the extended work is arXiv:2403.09227 | `docs/paper/refs.bib` |
| Added `sam3dgen` (Meta's *SAM 3D: 3Dfy Anything in Images*, 2511.16624) so it is no longer conflated with the 2023 *SAM3D: Segment Anything in 3D Scenes* we cite for discovery | `docs/paper/refs.bib` |
| `docs/BASELINES.md` rewritten: removed the claim that SimFoundry's F1 0.66–0.71 / 0.81–0.92 numbers are reusable (they are **F1@10 mm on staged YCB tabletops with FoundationPose quasi-GT**) | `docs/BASELINES.md` |

## Unverified — check before citing

- **Real2Code** venue: our note says CoRL 2024; evidence points to **ICLR 2025**.
- arXiv IDs marked ⚠️ in `docs/BASELINES.md` (VoMP, PhysQuantAgent, PhysSplat,
  Pixie, feature-field convex decomposition) — IDs are real, metadata was not
  individually confirmed.
- **"Vid2Sim" names two different CVPR 2025 papers.** Check which one you mean.

---

## What the 2026-08-01 rewrite added beyond these fixes

- **New Sec. IV "Experimental Setup"** — every metric is now defined in the
  paper before it is used (discovery F1 matching rule, geometry F1 pooling and
  reference choice, stability ratio with simulator/threshold/duration, the
  official-test-split appearance protocol).
- **New Sec. V-F "Physical Plausibility"** — anchors our 77.2% against
  PhyRecon (78.16%) and HoloScene (93.9%) on ScanNet++, *and* states why the
  three numbers are not directly comparable.
- **Method restructured** to specify the pipeline completely: the two-mode
  (automatic / benchmark) structure sharing one interface, best-view selection,
  the size-sanity and tier gates, the LLM fallback path, and the removal
  selectors with their radii.
- **Related Work** gained SimRecon, WANDA, Lumera, PhyRecon, DISCOVERSE,
  3DGRUT, the articulation cluster, and a paragraph disambiguating the two
  "SAM3D" papers.
- **`supplementary.tex` (new, 5 pp.)** — full hyperparameter tables per stage,
  formal metric definitions, the three-environment explanation, baseline
  reimplementation details, failure analysis, and the engineering findings
  (CoACD-before-torch, COM-vs-link frames, composited targets, stale caches).

## Paper build (updated 2026-08-01)

`bash docs/paper/build.sh` produces two PDFs from one source:

| file | pages | what |
|---|---|---|
| `root.pdf` | 16 | **extended / arXiv version**, appendices included |
| `conference.pdf` | 10 | **ICRA cut**, no appendices |

Shared sources: `preamble.tex`, `body.tex`, `fig_pipeline.tex`, `refs.bib`;
`appendix.tex` is extended-only. The split is driven by `\ifextended`, set in
each wrapper. In `body.tex`, `\ifextended ... \fi` blocks and the `\ext{}` /
`\conf{}` macros gate content: the systems-comparison table, notation table,
both algorithms, the per-category subsection, the hybrid and stage-timing
tables, the Discussion section, and two figures are extended-only.

**ICRA has no supplementary track.** Verified against the ICRA 2026 CFP: the
limit is 8 pages *including references*, supplementary material must fit inside
those 8 pages, and only a video attachment is accepted. So nothing a reviewer
must read may live in `appendix.tex`.

### Known issue

`conference.pdf` is **10 pages against an 8-page limit** — about two columns
must still go. Everything cheap has already been cut. Remaining candidates,
in increasing order of cost to the paper:

1. Sec. V-G furniture extension (~18 lines) — an opt-in probe on one scene.
2. Sec. V-I BEHAVIOR-1K (~16 lines) — could become a footnote.
3. Sec. V-C MaskClustering prose (~10 lines) — the table carries most of it.
4. The SR sweep table (~8 lines) — but it is a headline result now.
5. Related Work compression — currently the longest non-experimental section.

This is an editorial call about which result to demote, so it is left open.
Gate a block by wrapping it in `\ifextended ... \fi`; it will stay in
`root.pdf` and vanish from `conference.pdf`.
