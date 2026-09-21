# Task 17 — End-to-End Phone Room Collection

**Priority:** P0  
**Suggested owner:** user/robot lab lead with data-support agent  
**Depends on:** Tasks 02–06, 11–13  
**Blocks:** Task 18 and final title claim

## Objective
Collect and process at least three genuinely new rooms/workspaces using the user-facing phone protocol, before task-specific tuning, to validate capture-to-simulator deployment.

## Room design
Select visually and physically diverse rooms: different support materials, clutter, lighting, object scales, transparent/thin objects, and obstacle layouts. Each room should support at least two measurable manipulation tasks and contain both easy and certificate-challenging cases.

## Required capture package
- raw phone RGB video; optional depth/AR pose logs;
- calibration marker/base observations;
- privacy/release consent and redaction status;
- surveyed check distances not used by construction, reserved for evaluation;
- task definitions and predeclared reset distributions.

## Implementation steps
1. Freeze rooms/tasks and capture order before looking at SimAny outcomes.
2. Run capture validation and permit at most one protocol-defined recapture.
3. Execute full automatic pipeline; log all build failures and repairs.
4. Measure independent scale, alignment, object pose/support, capture/build time, and certificate coverage.
5. Archive raw and processed data with hashes and access controls.
6. Select qualitative examples only after quantitative freeze, using predefined criteria.

## Acceptance criteria
- [ ] Three rooms reach reconstruction; failures remain counted.
- [ ] At least six tasks have task graphs and certificates.
- [ ] No per-object clicks or manual placement enter the automatic build.
- [ ] Capture and total wall-clock times are reported honestly.

## Paper artifact unlocked
The central phone-scan evidence, deployment table, teaser, and video narrative.
