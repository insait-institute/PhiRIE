# Experiment design — SimAny paper (audited 2026-08-10)

Governing rule: every table answers one falsifiable question, every metric has
a stated reason to exist, and every number is regenerated from the outputs
tree by `agents/eval/make_paper_tables.py` (or is verified prose). A cell that
cannot be traced to a file on disk does not go in the paper.

## 1. Claim → experiment → metric matrix (main paper)

| # | Claim | Experiment | Metric & why it is meaningful | Evidence status | Action |
|---|-------|-----------|-------------------------------|-----------------|--------|
| C1 | Full automation costs little vs GT-driven | tab:main, 50 ScanNet++ val scenes, GT vs SIMANY_AUTO; stages 2–7 byte-identical | **Yield (tier A+B)**: fraction of objects passing geometry gates — the "usable asset" rate, not a soft mean. **F1@20mm pooled per-object**: pooled, not per-scene macro, because 5 zero-yield scenes drag macro to 0.636 without describing asset quality; **matched-GT scoring** (vertex-IoU≥0.25 match, 381/671) because own-extraction scoring (0.630) lets the pipeline grade its own homework | STRONG (post-audit) | none |
| C2 | GT dependencies are individually removable | tab:ladder, rungs A/B/C_GT/C_SAM3/D; one variable changes per rung | Same yield/F1 pair as C1 so rungs are comparable; footnotes carry matched-vs-own both ways because C_SAM3-vs-B *reverses sign* under own-extraction (0.667 vs 0.630) — reporting both is what makes the ladder honest | STRONG; stale "15%" prose fixed → 17.2% (2026-08-10) | none |
| C3 | Discovery beats MaskClustering given identical masks | tab:mc — same SAM3 masks, same prompts, matched 28-frame budget | F1 at vertex-IoU 0.25 **and** 0.5: 0.25 measures "found it", 0.5 measures "delineated it"; the 1.5× claim is made only at 0.5 where it holds | WEAK: single pilot scene (admitted in Limitations) | Extend to ≥5 scenes; commands below (§3.1) |
| C4 | Residual-gated hybrid generation is safe + better | tab:hybridcat, 453 dual-scored objects across 45 scenes | Per-generator mean F1 + **collapse count** (F1<0.1): the gate's job is catastrophe prevention, so collapses caught (18/18) is the headline; oracle gap (0.006) bounds what a better gate could add | STRONG | none |
| C5 | Gaussian-native removal + completion works | Removal battery (α-residual, re-detection, hole PSNR), FlashSplat baseline, Qwen-vs-LaMa | Re-detection rate = does the object *functionally* disappear (detector-level, not pixel-level); hole PSNR min-across-views (one bad frame = visible blotch); union IoU vs FlashSplat isolates the transparency-vote contribution | WEAK: pilot scene (16 objects) for the battery; render side now 4-scene (mean 29.73 vs 29.34 dB, filled 2026-08-10) | 2 more scenes queued (job 687728) → update prose to 6-scene means when done; removal battery extension to the 6 clean-bg scenes is the next cheapest strengthening |
| C6 | Twins are physically plausible | tab:srsweep threshold sweep 1–10 cm, both modes | SR at 3 cm is a *convention*; publishing the whole curve + drift p50/p90 is the defensible move because published protocols (PhyRecon vs HoloScene) disagree by >50 pts on a shared baseline. Negative result kept: hybrid raises F1 but lowers SR (79.9→76.6) | STRONG | none |
| C7 | Cost is dominated by classical geometry, not generation | tab:timing stage breakdown | A6000-hours by stage; the 68%-in-registration+decomposition number is what justifies "43×/29× faster than HoloScene" being architectural, not hardware luck | STRONG | none |
| C8 | Task-level validation is bounded, not solved | BEHAVIOR-1K coverage protocol | Vocabulary-complete %, groundable count, coverage % — deliberately a *bound* (2.3%/0.6%) framed as negative result | STRONG (honest) | none |

Prose-consistency fixes applied 2026-08-10: row-D "surviving 15%" → "17.2%"
(matches 67/389 in tab:ladder); clean-background composite extended from
pilot-only to the 4-scene means (verified from render_metrics_v2.json:
twin-on-clean ≥ clean-only on all four; 29.73 vs 29.34 dB). Discovery
accounting (336 of 844, +522 unmatched) re-verified from
`outputs/*_auto/eval_vs_gt.json`: 560 matched proposals cover 336 distinct GT
instances (matching is many-to-one), 844−508 missed = 336 ✓. Overfull boxes in
tab:mc and tab:srsweep fixed (tabcolsep). Both PDFs rebuild clean
(root 16 pp / conference 10 pp).

## 2. Open page-budget decision (needs author call)

conference.tex declares "8 pages including references"; the current build is
10 (body ends p.9, references spill to p.10). Candidate cuts, in order of least
damage: (i) compress V-E removal prose (the extended version keeps the full
battery), (ii) fold V-G furniture into Limitations (it is already labeled
preliminary), (iii) drop fig:removal to single-column. Not applied — cutting
content is an author decision.

## 3. Designed-but-unfilled experiments (each with its run recipe)

### 3.1 Multi-scene MaskClustering (strengthens C3)
5 scenes: c50d2d1d42 (done) + 578511c8a9, 9071e139d9, 3db0a1c8f3, d755b3d9d8
(high-instance-count val scenes). Per scene:
```
run agents.baselines.maskclustering prepare --scene <S> --stride 12
run agents.baselines.maskclustering srun --scene <S>   # prints GPU cmds (their env)
run agents.baselines.maskclustering convert --scene <S>
run agents.eval.eval_instances --pred outputs/<S>_mc/auto_instances.npz
```
~1 GPU-h/scene. Table change: tab:mc gains a "5-scene mean" row; caption drops
"(pilot scene)".

### 3.2 Removal battery on all 6 clean-bg scenes (strengthens C5)
Scenes with inpaint/clean_background.ply: 27dd4da69e, 45b0dac5e3, 578511c8a9,
7b6477cb95, 825d228aec, c50d2d1d42. Re-run the verification stage
(agents.assets verify + re-detection) per scene; extend the α-residual /
re-detection counts from 16 objects to ~76.

### 3.3 Policy-deployment axis (gap study; extended version or next paper)
Four-track grid per docs/GAP_STUDY.md — the only fully apples-to-apples policy
cell is **LIBERO round-trip** (native MuJoCo → video → video2sim twin → pi0.5
on both arms, same tasks). Protocol locked before any number is quoted:
- episodes ≥ 5 with pose jitter on *all* episodes; `--time-limit 32` explicit;
  identical, reported camera per arm; observation mode fixed (raster|composite);
  demo-only flags banned.
- Metric: per-stage funnel (grasp/lift/hover/place rates) with Wilson CIs, not
  bare success — the score landscape is 0-heavy and success alone has no
  gradient (power analysis: +0.15 nonzero-rate needs ~158 eps/arm; +0.25 → 60;
  +0.40 → 24).
- Recon/generation cells for the video tracks report holdout PSNR and
  **fit@20/40mm** (never "F1" without independent GT).
Status: video2sim pipeline works end-to-end (pilot a29cccc784: VGGT 12.3 cm /
0.5°, splat 20.98 dB vs 34.5 dB COLMAP reference); zeroshot-100 fleet
partially complete; LIBERO native env not yet installed (the blocking item for
the round trip).

## 4. Metric definitions guardrails (why-not notes)

- **No single "gap number"** across tracks: axes are deliberately separate
  because the comparability grid differs per track (GAP_STUDY §1).
- **fit@ vs F1@**: any registration score against the pipeline's own partial
  observation is labeled fit@ (sharp_hybrid precedent) — F1 implies an
  independent reference.
- **Never mix SimFoundry's published numbers into our tables**: their F1@10mm
  is on 12 staged YCB scenes with FoundationPose quasi-GT and a human-tuned
  headline column; compare zero-shot column only, in prose.
- **PSNR ≠ removal quality**: background PSNR says nothing about hole quality
  (scene 27dd4da69e ranks #1 on bg PSNR with 10/12 objects failing the 15 dB
  hole gate) — hole-fill PSNR stays min-across-views, per-object.
