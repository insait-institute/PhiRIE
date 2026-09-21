---
name: simany-paper-state
description: "SimAny ICRA paper: current structure, what the 2026-08-01 rewrite changed, supplementary, and the open page-limit issue"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0b4246c3-7101-4e38-8834-10129cb6e3ad
  modified: 2026-08-01T01:46:09.054Z
---

**ICRA HAS NO SUPPLEMENTARY TRACK** (verified against the ICRA 2026 CFP):
8 pages *including references*, supplementary must fit inside those 8 pages,
only a video attachment is accepted. So the appendix material can only ship in
an arXiv/extended version.

`bash paper/build.sh` (TeX Live 2026 module on PATH) builds **two PDFs from one
source**: `root.pdf` (extended/arXiv, 16 pp, appendices included) and
`conference.pdf` (ICRA cut, 10 pp, no appendices). Shared: `preamble.tex`,
`body.tex`, `fig_pipeline.tex`, `refs.bib`; `appendix.tex` is extended-only.
Driven by `\ifextended` set in each wrapper; gate a block with
`\ifextended ... \fi` or the `\ext{}` / `\conf{}` macros. `\name` = SimAny.

**All paper data tables are generated**, never hand-typed:
`python -m simany.eval.make_paper_tables` reads `outputs/` and writes
`paper/tables/*.tex` (per-category, per-scene, stage timing, hybrid
per-category, SR sweep). Re-run it after any re-computation.

**Main paper structure**: Intro / Related Work / Method (discovery, best-view +
generation, registration, tiers+rejection, physics+export, removal+completion,
verification) / **Experimental Setup** (new — defines every metric before use) /
Experiments (main, ablation ladder, MaskClustering, single-vs-multi-view,
removal+verification, **physical plausibility** (new), furniture, efficiency,
BEHAVIOR-1K) / Limitations / Conclusion. Figures: eval_strip, cmp_bottles_v3,
qwen_vs_lama, mj_photoreal_grid (factory_sheet was dropped for space).

**Supplementary** = full hyperparameter tables per stage (values read out of
the code, not invented), formal metric definitions, three-interpreter
explanation, baseline reimplementation details, failure analysis, engineering
findings, BEHAVIOR-1K protocol, registration stress-suite detail.

**The 2026-08-01 rewrite applied every P1/P2 item in `docs/PAPER_REVISIONS.md`**
— chiefly: the abstract's 29x/43x now names HoloScene (and adds ~6.5x/object vs
SimFoundry); the intro no longer implies inpaint-and-retrain is a *quality*
compromise (it is cost + no per-object granularity — SimFoundry's Table L.6
shows it beating their cleared capture); the "first system that is
simultaneously..." four-way novelty claim became three mechanism claims; the
28.32 dB number carries a footnote distinguishing it from SimFoundry's 15.29 dB
render-vs-real-video protocol. New Sec. V-F anchors our 77.2% drop-test
stability against PhyRecon's 78.16% and HoloScene's 93.9% on ScanNet++ while
stating why the three are not comparable.

**Three findings that came out of generating the tables from real data**
(these corrected earlier beliefs — do not revert them):
1. `report.json` still holds the ORIGINAL drop test (pre-hybrid assets, COM
   frame). The corrected one is `outputs/<scene>/drop_v2.json` from
   `simany.sim.redrop`. Always read drop_v2.
2. The 79.9%->77.2% stability change is **-3.3 pts from the hybrid asset swap**
   and **+0.7 pts from the COM->link frame fix** — the frame fix is small and
   *raises* stability. Earlier notes blamed the frame fix for ~3 points.
3. **The hybrid raises geometry F1 (0.708->0.783) but slightly lowers
   stability (79.9->76.6, frame held fixed).** Better surfaces != better
   physics. Now reported in Sec. V-F as an honest negative.
Also measured: SR sweep 71.3/74.2/77.2/79.9/86.9 at 1/2/3/5/10 cm, drift
p50 0.03 mm vs p90 138 mm — stability is bimodal, so the threshold the
literature disagrees over is not what separates these systems.

**OPEN ISSUE: conference.pdf is 10 pages against ICRA's 8.** Everything cheap
is already gated to extended-only. Remaining candidates ranked in
`docs/PAPER_REVISIONS.md` ("Known issue"): furniture subsection, BEHAVIOR-1K,
MaskClustering prose, Related Work compression. Left open — it is an editorial
call about which result to demote.

**Refactor regression found and fixed 2026-08-01**: `simany/sim/redrop.py` had
`parents[1]` (correct in the old flat layout, wrong after the reorg). All
`parents[N]` in the package are now consistent with file depth; an import
smoke test cannot catch this class of bug.

Verify before citing: `Real2Code` venue (likely ICLR 2025, not CoRL 2024) and
the ⚠️-marked arXiv IDs in `docs/BASELINES.md`.

Related: [[simany-rename-2026-07-27]], [[simany-baseline-positioning]],
[[simfoundry-repro-status]].
