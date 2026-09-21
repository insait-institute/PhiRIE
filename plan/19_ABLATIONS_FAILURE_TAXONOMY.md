# Task 19 — Construction/Capture/Certificate Ablations and Failure Taxonomy

**Priority:** P1  
**Suggested owner:** experiment integration researcher  
**Depends on:** Tasks 09–18  
**Blocks:** final paper tables

## Objective
Run a minimal, causally interpretable ablation suite and label failures consistently from synchronized videos, states, contacts, and build reports.

## Required construction ablations
- single-view generator only;
- no residual candidate gate;
- one-way registration objective;
- no collision decomposition;
- generic mass/friction;
- legacy support shims;
- no Gaussian object removal;
- raw reconstruction/no factorization.

## Required capture ablations
One image, 8 frames, 32 frames, full RGB video, RGB+phone depth, pose noise, blur/missing-view corruption. Derive all from the same full capture when possible.

## Certificate ablations
Global PSNR, global F1, drop stability, task visual+pose, +support/contact, +dynamics, +reachability, full+repair.

## Failure labels
build/pose, object discovery, asset geometry, registration/scale, appearance ghosting, collision/support, physical response, camera/control mismatch, reachability/safety, policy semantic failure, ambiguous rubric. Assign the earliest causal failure plus secondary contributors.

## Outputs
- `configs/ablations/*.yaml`
- `agents/eval/failure_taxonomy.py`
- `agents/eval/ablation_matrix.py`
- `docs/FAILURE_LABELING.md`

## Acceptance criteria
- [ ] Each row changes one declared variable.
- [ ] Episode/reset IDs and policies remain paired.
- [ ] Two annotators label a subset and agreement is reported.
- [ ] Ablations report both static and policy metrics, plus coverage.

## Paper artifact unlocked
Three ablation tables and an evidence-backed limitations/failure section.
