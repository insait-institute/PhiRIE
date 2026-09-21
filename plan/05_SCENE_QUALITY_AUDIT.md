# Task 05 — Automatic Build Audit and Task-Agnostic Quality Report

**Priority:** P1  
**Suggested owner:** evaluation engineer  
**Depends on:** Tasks 01, 03–04  
**Blocks:** Tasks 12–13, 17

## Objective
Produce a ground-truth-free build audit that catches silent reconstruction, factorization, rendering, and physics failures before task-specific certification.

## Existing code
- `agents/eval/factory_report.py`
- `agents/eval/factory_eval_render.py`
- Gaussian removal verification under `agents/edit/` and `agents/eval/`
- `robo/sim/redrop.py`

## Outputs
- `agents/eval/build_audit.py`
- `docs/BUILD_AUDIT_SCHEMA.md`
- `tests/test_build_audit.py`

## Required checks
- pose/reconstruction coverage and held-out rendering;
- duplicate/missing instance indicators;
- registration residual and candidate disagreement;
- visual-to-collision surface distance;
- removal alpha/depth residual and residual re-detection;
- penetration, support overlap, settle drift, drop response;
- room-collision coverage and disconnected floating components;
- metric scale and robot-alignment status.

## Implementation steps
1. Consolidate existing reports into one versioned JSON schema.
2. Distinguish `pass`, `warning`, `fail`, and `not_applicable`; never collapse missing evidence into pass.
3. Produce per-object and scene-level summaries with links to diagnostic renders.
4. Add deterministic thresholds from Task 00 and store raw measurements.
5. Queue failed objects/stages for Task 13 without modifying outputs in place.

## Tests
- Synthetic ghost object, penetration, missing support, bad scale, and transparent-object residue trigger expected checks.
- Audit is deterministic across reruns.
- A failed check cannot be removed by averaging with unrelated room regions.

## Acceptance criteria
- [ ] Every build receives one audit JSON and human-readable report.
- [ ] Audit runs without ScanNet++ GT.
- [ ] All features required by the task certificate are exposed through stable APIs.

## Paper artifact unlocked
Automatic verification claim and global-vs-task-local certificate ablation.
