# SimAnyRoom: Gaussian-centric video-to-simulation campaign

Date: 2026-09-08. Code: `RunyiYang/PhiRoom`; paper: `RunyiYang/SimAnyRoom`.
This is the active **new system-development campaign**, not a replacement of any
historical numerical result. Start with `AGENT_PROMPT.md`, `RUNBOOK.md`, and
`PAPER_SCOPE.md`. The user's approved SimAnyRoom system title stays unchanged.

## Objective

One RGB video -> estimated cameras/metric-scale calibration -> persistent Gaussian
scene -> frozen Chorus semantic/visual encoding -> open-vocabulary instances ->
observed appearance plus generated hidden geometry -> registered physics proxies
-> actual-state Gaussian observations -> bounded state-conditioned harmonization
-> frozen-policy evaluation.

Compare official SimFoundry and PolaRiS. Do NOT rename our fixed TRELLIS pipeline
as SimFoundry. Video input, Gaussian rendering, generic image editing, Chorus, and
NVIDIA DiffusionHarmonizer are not individually new inventions. The prospective
contribution is their shared 3D identity/state and measured system-level effects.

## Readiness: implemented versus needing native admission

Implemented in this revision:
- frozen scene-family inventory, capability registry, source-only HTML gallery;
- finite experiment-matrix generation, immutable task/parameter identities,
  dependency handling, ordinary Slurm submission and component collection;
- actual TRELLIS, TRELLIS.2 and official ReconViaGen-v0.2 inference/export adapters;
- actual Telea, SDXL inpainting, Qwen-Image-Edit-2511 and Gemini image-edit adapters;
- explicit paid-API permission/call/cost reservations and exact outside-mask copy;
- frozen Chorus package inference, persistent source IDs, sparse 3D instance grouping;
- rigid-motion/depth/instance validated temporal warping, protected bounded residuals;
- NVIDIA-model sequence adapter with emitted-output history and state correction;
- small trainable state-conditioned residual module and same-state training loop;
- official SimFoundry execution, real PolaRiS-bundle validation and method-neutral
  asset export into the existing native importer contract;
- existing-harness command bridge, not a second robot rollout ledger.

Not claimed completed by CPU tests: model downloads/access, upstream API/runtime
smokes, full RGB-video camera/scale recovery on new data, baseline native-role and
policy integration, correct GS robot occlusion, trained visual-adapter quality,
new dataset-specific robot/task adapters, or any full new scientific result.
PolaRiS's official composition workflow includes human effort. Its bundle validator
is NOT an invented fully automatic PolaRiS constructor. RVG-v0.5 is an optional
separate upstream adapter, NOT an alias of v0.2. No model is silently substituted.

## Task owners and order

| Task | Owner | First real artifact | Depends on |
|---|---|---|---|
| P00 | prior-results owner | close F1 missing units, preserve F2/F3 limitations | existing outputs |
| P01 | data owner | verified inventory + source-only gallery + public/private split | none |
| P02 | official-baseline owner | actual SF output and composed PolaRiS scene + effort log | P01 |
| P03 | RGB/GS owner | non-GT video camera reconstruction + frame/occlusion controls | P01 |
| P04 | semantics owner | input-aligned Chorus IDs + independent instances | P03 |
| P05 | generator owner | shared-input three-generator comparison | P01/P04 |
| P06 | inpainting owner | exact-mask model comparison + multi-view reveal test | P01/P03 |
| P07 | observation owner | same-state official-vs-constrained H sequence + latency | P03/P04 |
| P08 | policy owner | fresh reference/baseline/ours native paired block | P02/P05/P07 |
| P09 | release owner | source-bound tables, results archive, honest 90-second demo | all measured |

P00, P01, P02 setup and P09 table scaffolding start together. P05/P06 can use frozen
existing captures without waiting for all new RGB-video rooms. P07 cannot hide an
incorrect raw renderer. API or dataset access blocks only the affected branch.

## Finite full experiment targets, not already available data

- Existing 48-instance RoboCasa follow-up: close 40 unmeasured B1/B2 units using
  its own same-process/retry contract. Never pool this with the new cohort.
- New RoboCasa main: 12 layout groups x 3 admitted rigid task families x 2 target
  instances = 72 canonical instances; 10 paired resets; 4 asset-system arms =
  2,880 planned units. Freeze exact task IDs from the installed benchmark.
- BEHAVIOR extension: 8 room groups x 3 admitted rigid task instances = 24;
  10 resets x 4 arms = 960 planned units, ONLY after an appropriate policy and
  task-state adapter pass. Otherwise publish reconstruction/physics-only data.
- Observation/Chorus interventions: predeclared 36-instance RoboCasa subset;
  10 resets. Enumerate six observation/semantic arms, including official H and
  current Option C. These are not a generator x inpainter x policy Cartesian grid.
  Fresh reference/native controls are required when starting a new policy process.
- Generator study: 240 declared object jobs, three generator families, one shared
  primary seed = 720 proposals. A separate DEV subset of 48 objects x 3 models x
  3 seeds diagnoses stochasticity; it never selects TEST best-of-N outputs.
- Background completion: 24 DEV and 96 TEST same-state masked cases across scene
  families. Telea/SDXL/Qwen/Nano Banana are separate versions; Nano Banana 2 optional.
  Reuse each model's output across all downstream rendering evaluations.
- Reconstruction/appearance extension: HSSD 30 TEST rooms, ReplicaCAD 24 layouts,
  Hypersim 40 TEST scenes, ScanNet++ historical50 plus 20 new RGB-video scenes
  subject to rights and pretraining-overlap review; optional Replica/HM3D/phone.

These are prospective caps, not invented scene identifiers or availability claims.
`DATASETS.md` defines capability and access gates. Full means the complete frozen
eligible roster, not downloading every multi-terabyte collection and applying an
inapplicable robot metric to it. Do not remove failed cases after outcomes.

## Fast launch

```bash
bash run/campaign/run.sh catalog --out /NEW/catalog
bash run/campaign/run.sh freeze --inventory /RESOLVED/inventory.jsonl --out /NEW/cohort
bash run/campaign/run.sh gallery --inventory /NEW/cohort/scenes.jsonl --out /NEW/gallery
bash run/campaign/run.sh matrix --scenes /NEW/cohort/scenes.jsonl --models /RESOLVED/models.yaml --out /NEW/matrix
bash run/campaign/run.sh prepare --tasks /NEW/matrix/tasks.jsonl --runtime /RESOLVED/runtime.yaml --out /NEW/run
bash run/campaign/run.sh launch --bundle /NEW/run --max-jobs 1
bash run/campaign/run.sh launch --bundle /NEW/run --max-jobs 1 --submit
bash run/campaign/run.sh collect --bundle /NEW/run --out /NEW/collection-001
```

`matrix/unbound.json` is a mandatory report, not something to delete. Generic
component completion is NOT manipulation success. Use the existing native ledger
and `robo.eval.paper_pipeline` for scientific counts, pairing and intervals.
