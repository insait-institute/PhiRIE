# Task 13 — Certificate-Guided Targeted Repair and Abstention

**Priority:** P1  
**Suggested owner:** pipeline integration engineer  
**Depends on:** Tasks 05, 11–12  
**Blocks:** final phone/paired claims

## Objective
Convert certificate explanations into bounded, reproducible repair actions and demonstrate that repair improves downstream paired metrics without manual object placement.

## Repair routes
- visual ghosting → expand/revote removal, strengthen completion, re-render audit;
- pose/scale uncertainty → alternate generator candidate, registration restart, extra views;
- collision/contact failure → regenerate/decimate collision, adjust support, bounded physics search;
- robot alignment/reachability → recalibrate transform, never move the target by hand;
- unresolved uncertainty → abstain.

## Outputs
- `robo/certification/repair.py`
- `configs/repair/default.yaml`
- `run/run_repair.sh`
- `tests/test_repair_routing.py`

## Implementation steps
1. Define a finite repair action registry with allowed inputs and maximum attempts/cost.
2. Create a new immutable build ID for every repair; retain parent lineage.
3. Re-run only invalidated stages using stage hashes.
4. Recompute audit/certificate before any policy rerun.
5. Compare repair to equal-compute random repair and no-repair controls.
6. Track coverage, human time, compute, and whether failure migrates to another stage.

## Acceptance criteria
- [ ] Every repair is automatically selected from measured evidence.
- [ ] No manual pose/asset placement is counted as automatic.
- [ ] Repair improves held-out paired MAE/MMRV or increases honest abstention coverage trade-off.
- [ ] Failed repairs remain visible in the report.

## Paper artifact unlocked
“Repair or abstain” claim, final main-table row, and compelling before/after video.
