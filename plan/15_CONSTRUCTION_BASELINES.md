# Task 15 — Fair Construction Baseline Adapters

**Priority:** P1  
**Suggested owner:** baseline/reproduction team  
**Depends on:** Tasks 00–01, 03–06, 14  
**Blocks:** final oracle table

## Objective
Run the strongest available scene-construction baselines under common input and output contracts without overstating incomplete releases.

## Baselines
- raw fused reconstruction / no object factorization;
- SimRecon official released modules plus explicitly documented glue;
- ReplicateAnyScene released stages;
- official PolaRiS manual construction as a human-effort reference;
- RoboSnap only if an official runnable release supports the frozen protocol;
- SimFoundry remains literature-only unless an official release appears.

## Outputs
- `baselines/simrecon_adapter.py`
- `baselines/replicate_adapter.py`
- `baselines/raw_reconstruction.py`
- `baselines/release_status.yaml`
- `docs/BASELINE_REPRODUCTION.md`

## Implementation steps
1. Pin commit, model weights, license, and release date.
2. Feed identical capture frames/poses/depth and forbid hidden oracle inputs.
3. Convert outputs to the engine-neutral manifest; record every added glue step.
4. Measure build success, runtime, GPU/CPU, and human intervention.
5. Run common static/oracle metrics and policy outcomes where output supports them.
6. Mark unsupported cells `N/A`, never zero.

## Acceptance criteria
- [ ] Every numeric row is reproducible from official released code plus audited glue.
- [ ] Partial and literature-only methods are visually distinguished.
- [ ] Metric thresholds and references match across methods.
- [ ] Failure to run is documented with logs and does not become an unfair zero.

## Paper artifact unlocked
Construction baseline rows and defensible related-work positioning.
