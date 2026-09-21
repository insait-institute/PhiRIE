# E1 current-source admission and missing-unit plan

Status: **protocol review required; no construction/metric execution authorized by
this audit**. This is a reference inventory, not a replacement five-row result.
Owner `trellis2_environment`, branch `agent/icra-e1-current-source-audit`.

## Current evidence and scope

The full E3 source is genuinely in the automatic-discovery/splat-fused **input
family**: fresh official-TRAIN-only Gaussian rendering, TSDF fusion, SAM3 instance
discovery and construction-surface registration. Its entire discovered roster,
including preparation failures, is retained. The normalized job ledger is a
controller outcome; it does not establish completed simulator construction.

A direct E1 row adapter is not scientifically admissible under the current fixed
producer contract. `construction_metrics` names `accepted_instances` build yield,
`stable_instances/tested_instances` attempted drop tests, and runtime scene
construction time; its table drops any extra scope columns. Feeding differently
scoped values into those fields would lose the distinction in the final table.
The metadata inventory therefore leaves **all E1 measurement fields null**,
retains all 250 planned scene/regime cells, and carries actual E3 controller
counts only in a distinctly scoped evidence object. It never renders a table.

Differences that prevent combining old and new rows:

- Legacy E1 enumerated/prepared whitelist assets and used tier A/B acceptance.
  New E3 uses FULL discovery vocabulary, every discovered job, two initial
  tools, bounded registration retry, and A4 accept/reject/abstain gates.
- E3 physical evidence uses a single convex hull, fixed category-independent
  mass/friction, corrected initial collision AABB, explicit solver/contact
  settings and link-frame drift. Legacy `factory_report.drop_test` uses the
  exported URDF, starts its base at 5 mm, and a different sinking predicate.
  The shared two-second settling/free phases do not make those tests identical.
- E4 `--collision-mode room` exports paired carved room collision and records
  MuJoCo settle diagnostics. Those are neither the legacy isolated drop test
  nor manipulation success. Sealed population metadata binds exported bytes;
  a materialization manifest alone does not certify the physics/export tail.
- E3 runtime covers attributed initial generation, registration, isolated
  physics and retry. It omits Gaussian training, discovery/fusion/preparation,
  full-room export and appearance cleanup. Summing overlapping E3 policy rows
  would also double-count shared initial proposals and controller work.
- E4's task protocol intentionally excludes scenes without semantic queries;
  E1's construction population still contains those scenes. E4 completion
  cannot supply a 50-scene E1 completion denominator by itself.

## Prospective common measurement contract

Before launching any E1 job, review and freeze one shared constructor/proposal
policy, vocabulary, input enumeration, physical parameters, collision backend,
acceptance definition and metric scope across all five input regimes. Replacing
only the automatic+splat row while keeping historical single-TRELLIS tier-A/B
rows changes several treatment axes. The historical counts remain diagnostics.

Recommended next admissible measurement is **CPU-only export/drop closure of
the already frozen automatic+splat assets**, with independently named scopes in
the existing producer. It requires a small explicit scope extension to
`construction_metrics` and its renderer before publication, rather than silently
mapping controller/probe values. The exact prospective metric fields must be
reviewed before observing additional drop results. Either retain the historical
E1 drop implementation on newly exported matched collision assets, or explicitly
revise the common five-regime physical protocol; never relabel one as the other.

Runtime should expose input-reconstruction time separately from constructor-tail
time, preserving every stage's source/runtime receipt and retry history. Count
serial stage durations and GPU/CPU resource time separately; do not infer end-to-
end latency from a sum of parallel stages or substitute E4 job wall time for a
per-scene constructor runtime. Missing components remain null.

## Missing/rerun matrix

| Fixed input regime | Existing reusable candidate | Required new work before admissible row |
|---|---|---|
| GT segments + scan mesh | Public TRAIN RGB/cameras and declared input scan/segments; legacy output is reference only | Fresh source-bound enumeration and common constructor. Distinguish declared input scan from evaluation-only geometry; no same-surface score called held-out fidelity. |
| Auto discovery + scan mesh | TRAIN RGB/cameras, SAM3 source/checkpoint/runtime | Fresh automatic discovery using the declared scan geometry; common constructor, independent matching, export/drop/runtime closure. Auto+splat crops/proposals cannot be reused unless every input/config/identity hash matches. |
| GT segments + splat-fused mesh | Fresh TRAIN Gaussian and derived mesh | Explicit annotation-to-derived-surface association producer. Current `load_gt_instances` reads scan vertex coordinates/centroids; setting `SIMANY_MESH_SRC=derived` alone does not remove this leak. Keep construction registration targets derived-only. |
| Auto discovery + splat-fused mesh | Full E3 discovery/proposals/registration/control and independent geometry evaluation; complete sealed E4 exports where exact identities match | Freeze common acceptance/drop/runtime scope; authenticate current export assets, finish missing tails including no-query scenes, run only missing declared CPU drop units. No generation rerun merely for a new publication directory. |
| Single RGB + metric depth | Existing DA3 and SAM3 runtimes/checkpoints and RGB source frames | Predeclare first lexicographic official TRAIN frame (or another public-only fixed rule) for every scene before outcomes; remove GT-dependent selection/lifting calls through existing producers, retain all detected/preparation-failed instances, common constructor and held-out evaluation. Existing `run_simfoundry.sh` is not a safe launcher without these guards. |

The two GT-input rows may use explicitly declared discovery annotations, but
must not import evaluation labels or held-out geometry into selection/registration.
Where an independent reference cannot be separated from the declared scan input,
withhold independent-fidelity claims; do not silently use input reconstruction
residual as F1. The public-only automatic/single-RGB rows have no such annotation
access permission.

## Reuse gate and execution tiers

Every admitted reuse must bind original source commit, exact object IDs, inputs,
config, checkpoint, camera, controller, metric implementation and treatment.
Current snapshot seals are **candidate references only**; this cheap audit checks
metadata bytes and original saved seal hashes, without rerunning the original
geometry validators. Admission still requires the corresponding original-source
validator and unchanged measurement-affecting fields. In-flight or incomplete
E4 exports remain unavailable, never accepted based on file existence alone.

Smoke: canonical schemas and negative scope tests, then the predeclared first
lexicographic two scenes with source-bound exports, preserving missing units.
Pilot: the same declared real subset with final output fields; no alternative
scene selected because its stability/geometry is better. Full: all 50 scenes ×
five regimes, immutable common protocol/source and resumable per-object files.
Use ordinary jobs, no arrays. Separate CPU drops/aggregation from generation.

Cost: reference audit is CPU metadata I/O only, no weights or GPU. A CPU-only
export/drop pilot is the smallest new measurement; runtime must be measured on
that fixed pilot before budgeting a fleet. Four other regimes require fresh
input preparation and likely generation; no credible fleet GPU-hour estimate
exists for their not-yet-frozen object populations. No external credential is
known to block the existing ScanNet++/SAM3/TRELLIS/ReconViaGen/DA3 route; the
present blockers are scientific protocol and missing implementation/evidence.

## Reproducible audit

```bash
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python \
  plan/icra2027/01_construction_scale/audit_current_sources.py \
  --out outputs/e1-current-source-audit/NEW_snapshot.json
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e1_current_source_audit.py tests/test_construction_metrics.py \
  tests/test_construction_inventory.py
```

The snapshot supplies machine-readable per-scene/per-regime missing criteria,
original source references and exact hashes. It preserves the 250-row historical
CSV and never writes `construction_table.*` or paper values.
