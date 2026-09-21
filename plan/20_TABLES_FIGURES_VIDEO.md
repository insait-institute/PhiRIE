# Task 20 — Automatic Tables, Figures, and Three-Minute Video

**Priority:** P0 near freeze  
**Suggested owner:** paper artifact / visualization engineer  
**Depends on:** Tasks 01, 10, 14–19  
**Blocks:** Task 21

## Objective
Generate every quantitative paper artifact directly from frozen metric files and create a concise video that demonstrates the phone-scan intuition, policy evaluation, correct abstention, and repair.

## Outputs
- `agents/eval/make_icra_tables.py`
- `agents/eval/make_icra_figures.py`
- `paper_artifacts/<freeze_id>/tables/*.tex`
- `paper_artifacts/<freeze_id>/figures/*`
- `video/storyboard.md`, `video/render_video.py`

## Required figures
1. phone capture → metric/object-factorized twin → policy → certificate/repair;
2. real vs simulated policy matrix/scatter with within-task markers;
3. risk-coverage and calibration plot;
4. oracle error-attribution plot;
5. three phone rooms with one success, one repair, one correct abstention.

## Video structure
0–25 s: problem and phone scan. 25–65 s: automatic factorization/export. 65–115 s: same frozen policies in real/sim. 115–150 s: certificate and repair/abstain. 150–175 s: scale, oracle, and failure evidence. Final seconds: limitations, not marketing claims.

## Implementation steps
1. Read only frozen manifests/metrics; fail on manually edited cells.
2. Include CIs, sample counts, coverage, and `N/A` handling.
3. Generate data provenance JSON beside every table/figure.
4. Use predetermined qualitative selection rules and disclose demo-only camera changes.
5. Verify video length/size/anonymity.

## Acceptance criteria
- [ ] Re-running one command recreates all artifacts byte-stably where practical.
- [ ] Every numeric cell maps to run IDs.
- [ ] No placeholder/TBD remains after freeze; before freeze, placeholders stay visibly red.
- [ ] Video shows at least one failure and one abstention.

## Paper artifact unlocked
All final visuals and the submission video.
