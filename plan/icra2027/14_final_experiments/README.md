# Final SimAnyRoom experiments: converge, execute, release

This supersedes **unexecuted quotas** in campaign 13. It does not supersede
historical data or relabel old experiments. No new datasets, foundation models,
policies or learned modules are required for this submission.

## Fixed scientific scope

SimAnyRoom constructs simulation assets from a Gaussian reconstruction or RGB
video with declared public metric calibration. Shared object identity connects
geometry, appearance, collision and state-dependent observations. MuJoCo/Isaac
are execution backends, not reconstruction baselines. Game-engine portability,
universal room conversion and real-world transfer are not established here.

The final required evidence is: **official-system construction comparison**, a
**Gaussian observation/Harmonizer closed loop**, and a **replacement-scope check**.
The main system table is target-only/common-importer, not full-room recovery.
The scope table explicitly tests target+destination. Keep remaining native
context disclosed. Do not call a rendered room with one replaced target a fully
reconstructed physical environment.

## Final budget (not results)

| Suite | Independent instances | Resets | Arms | Planned units |
|---|---:|---:|---:|---:|
| System comparison | 24 in 6 layouts, 2 tasks | 10 | 4 | 960 |
| Observation / Harmonizer | 12 source-selected instances | 10 | 6 | 720 |
| Target to destination scope | the same 12-instance subset | 10 | 3 | 360 |
| **Total** | subsets overlap intentionally | | | **2,040** |

There are **48 per-instance paired blocks**, not 2,040 independent scenes.
Fresh references are already included in every suite. Each block uses one
policy process, and cannot be resumed by mixing old reference outcomes with
new process outcomes. Actual rollouts can be fewer than planned units because
of recorded method failures. Unavailable integration is missing, not zero success.

Tasks: `PickPlaceCounterToSink` (600 ticks), `PickPlaceSinkToCounter` (900 ticks),
under the already pinned RoboCasa version. Cabinet is omitted prospectively
from this **new** two-task study, not erased from previous three-task results.
This narrows the scope and avoids starting a new articulation integration.

Choose source families before constructor outcomes. Record prior exposure.
A new run on an already observed scene is a follow-up, not an unseen test.
Do not select the 24 instances from successful reconstructions. Source failures
remain visible; recipe changes after observing TEST require a separate study.

## What stays, what stops

Reuse the COMPLETE prior F1 closure: REF385, B073, B1131, B2118, B3128 out of480.
Do not rerun it or claim B3 outperforms B1. Keep the 50-room real-scan study, its
appearance losses, and old B4/BM/L1 negative results in their original cohorts.

Close a small, fixed set of component comparisons from existing captures (see
03_components). Do not continue 432/720 proposal campaigns merely to fill every
old quota. Do not train StateResidual16 or add RVG-v0.5, more editors, a second
policy, a new backend, or new HSSD/Hypersim/BEHAVIOR datasets. Existing official
PolaRiS6/100 remains a separate Isaac/DROID diagnostic, not a RoboCasa baseline.

Chorus is tested on a bounded query set. Empty predictions stay empty. Without
positive independent evidence it remains an evaluated front-end, not a headline
open-vocabulary contribution. Harmonizer needs the raw-GS/official/robot-restore/
state-constrained comparison, not only a pleasing screenshot.

## Code implemented in this revision

- `robo.campaign.finalize`: deterministic selection, immutable complete roster,
  actual DEV admission receipts, native-worker binding, global multi-bundle
  dispatch cap, canonical-ledger crosswalk and full-denominator collection.
- `robo.eval.final_release`: validated source/result joins; standard success,
  coverage, per-task tables and existing hierarchical paired bootstrap.
- `robo.rendering.final_observation`: atomic multi-camera state checks, temporal
  history isolation, no silent fallback, exact visible robot-core restoration,
  and a wrapper at the existing native policy-observation entry point.
- `run/campaign/finalize.sh`, fixed protocol, recipe template and CPU tests.

This is execution/validation code, not a new physics engine. The official asset
imports, renderer callbacks and final arm names still need to be connected in
the cluster's admitted native worker. This revision does not magically convert
the old DEV-only worker into every final backend. Task00 spells out the exact
integration work; tests here cannot establish GPU-model/native performance.

## Owners and sequence

00_integrate: bind existing working source branches and native final contracts.
01_official_comparison: real SF/PolaRiS adapted assets, matched robot blocks.
02_gaussian_observation: correct raw GS, same-state observation bridge, H arms.
03_components: close frozen generator/edit/semantic evidence, select on DEV only.
04_scope: preserve target exactly; replace declared destination only.
05_release: complete denominators, contrasts, claims and compact handoff.

Start00/01/03 in parallel. Start02 with the first valid calibrated scene. Freeze
one recipe after DEV, then run all registered blocks regardless of effect sign.
Do not turn this plan into another open-ended development campaign.

Read `RUNBOOK.md`, `SOURCE_INTEGRATION.md`, then `AGENT_PROMPT.md`.
