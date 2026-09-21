# E3 — Agentic Construction Ablation

**Priority:** P0, highest-value unfinished experiment  
**Paper output:** agentic-construction table and title claim  
**Depends on:** E0 and a frozen pool of object proposals

## Research question

Does feedback-driven tool selection, retry, and abstention improve room-to-sim construction over a fixed one-shot pipeline?

The word **agentic** is admissible only if this task produces a controlled ablation. A diagram of several modules is not evidence.

## Operational definition

For this paper, an agentic constructor:

1. maintains explicit scene/object job state;
2. invokes one or more specialized tools through a common artifact contract;
3. reads measurable evidence returned by those tools;
4. chooses `accept`, `retry`, `reject`, or `abstain`;
5. records every action, proposal, evidence value, dependency invalidation, and terminal status.

Ground-truth evaluation geometry may not enter the controller.

## Read first

- `agents/assets/factory_hybrid.py`
- `agents/assets/factory_align.py`
- `agents/assets/s5_align.py`
- `agents/assets/s6_physics.py`
- `agents/eval/build_audit.py`
- `robo/eval/fidelity_metrics.py`
- `robo/certification/features/geometry.py`
- `robo/certification/features/support_contact.py`
- `robo/certification/features/dynamics_probes.py`
- `plan/23_NVIDIA_HARMONIZER_INTEGRATION.md` only for the generic service pattern, not for object selection

## Fixed ablation rows

Every row starts from the same discovered object jobs and the same initial capture.

### A0 — Fixed single-path constructor

- Generate the existing best-view TRELLIS proposal only.
- Register it once.
- Unconditionally pass it downstream unless the current hard file/schema checks make export impossible.
- No evidence-based alternative, retry, or abstention.

### A1 — Multiple proposals, fixed priority

- Produce the same initial TRELLIS and ReconViaGen proposals used by later rows.
- Choose by the frozen rule `ReconViaGen if available, otherwise TRELLIS`.
- Do not inspect residuals to choose.

### A2 — Evidence-based proposal selection

- Register every initial proposal with the same registration code.
- Select using only construction-time evidence.
- Preliminary v1 uses the exact lexicographic contract below and never reads a
  legacy hybrid winner or score.

### A3 — Verification-triggered retry

- Begin with A2.
- If all initial proposals fail a predeclared gate, invoke at most one bounded
  signed-source-up registration restart of the lexicographically best initial
  raw mesh.  The frozen alternative order is `-z,+x,-x,+y,-y` after the
  initial `+z` attempt.
- The retry records the full triggering failure-reason set and creates a new
  proposal ID, registered artifact, and transform/artifact hash.  Second-view
  generation, reduced ReconViaGen subsets, stricter collision decomposition,
  and support-aware pose restarts are not active in preliminary v1.
- Do not tune the retry action using GT.

### A4 — Full SimAnyRoom

- Begin with A3.
- Accept only proposals that pass the frozen evidence gates.
- Unsupported jobs terminate as `abstain` and remain in build coverage.
- Do not silently retain the original scene object as a successful simulation asset.

## Fairness contract

- A0–A4 use identical discovery outputs, object IDs, input images, metric frame, registration code, physical test definition, evaluation reference, and acceptance threshold definitions.
- A1–A4 share the same initial two-proposal pool. A3/A4 may add only the documented retry proposal.
- Runtime includes all proposals and retries actually invoked.
- Conditional F1 and stability are reported together with planned-job coverage.
- Report a threshold sweep for A4 in the supplement so abstention cannot be optimized to one favorable point.

## Checked-in preliminary implementation contract

The machine-readable source of truth is split between
`configs/experiments/icra2027/agentic_jobs.yaml` and
`configs/experiments/icra2027/agentic_policies.yaml`.  This is schema version 1
and is explicitly a **development/preliminary** contract, not a paper-ready
freeze.  The 50-scene evaluation population may not be used to tune its
thresholds.  A fresh clean smoke-mode E0 contract is sufficient to launch the
full 50-scene run as preliminary evidence; paper-mode E0 is required only
before promoting results into the paper or title claim.

### Population and provenance boundary

- The conditional population contains 457 accepted factory-report object jobs
  from the exact 50-scene `construction_regimes.yaml` roster.  Membership is a
  report row with `tier` A/B and `rejected == null`, sorted by scene ID and
  integer object index.  Scene-level `hybrid_all.json`, a directory glob, or a
  success-only scan is never a membership authority.
- TRELLIS raw meshes exist for all 457 jobs.  ReconViaGen raw meshes exist for
  453; the remaining four jobs are retained in every policy denominator as
  typed unavailable proposals.  A policy therefore always has 457 planned
  jobs and the full A0--A4 ledger has 2,285 policy/object rows.
- Proposal identity is the raw canonical mesh path, byte size, and SHA-256,
  before registration.  Legacy `aligned.json`, `rvg_eval.json`, and
  `hybrid.json` winner/score fields are forbidden controller inputs.
- The cached TRELLIS/ReconViaGen bytes are hash-frozen, but their original
  generator commits, checkpoint identities, and runtime manifests were never
  recorded.  The current controller/registration commit must not be backfilled
  as a generator commit.  This provenance gap independently limits the run to
  preliminary evidence.
- Rendering observations use the source-scene Gaussian plus camera intrinsics
  and COLMAP poses frozen in the same 50 E2 room manifests.  E2 metric images
  and scores are not controller inputs.  E1 outputs are not a runtime
  dependency; E1 contributes only the checked-in 50-scene roster contract.
- The sanitized controller manifests preserve these identities as nested
  records: each resolved-jobs scene has `source_scene_gaussian`,
  `camera_artifacts.{intrinsics,poses}`, and its sanitized `jobs`; each resolved
  proposal has raw artifact maps, typed availability evidence, and explicit
  `raw_generator_provenance`.  The exact key sets are frozen in
  `controller_input_boundary.exact_sanitized_schemas`; unknown keys fail.
- The observation renderer uses `RGB+ED`, immediately discards RGB, and keeps
  expected depth plus Gaussian alpha only.  It intersects the RGBA alpha
  channel (`>=128`) with Gaussian alpha (`>=0.60`) and finite depth in the
  inclusive 0.05--8.0 m range, unprojects at pixel centers (`+0.5`), inverts
  COLMAP world-to-camera to camera-to-world, and stores world-frame float32
  points in metres.
- This is a controller-GT-isolated conditional ablation, not GT-free object
  discovery.  Of the 457 legacy crops, 386 SAM3 masks were selected by overlap
  with a GT projection and 71 are GT-projected fallbacks.  The controller never
  receives that provenance or any evaluation geometry.

### Deterministic policy semantics

- A0 selects TRELLIS without reading evidence for choice.
- A1 uses the frozen priority `ReconViaGen, then TRELLIS`, based only on whether
  a construction candidate completed with its required artifacts and evidence.
  It falls back for typed ReconViaGen unavailability or a ReconViaGen tool crash,
  and never reads a residual or gate value before choosing.
- A2, A3, and A4 use one lexicographic ordering: gate pass first, then fewer
  failed checks, lower symmetric clipped registration residual, smaller
  absolute scale error, lower settle drift, frozen tool order, and finally
  proposal ID.  No weighted opaque score is permitted.
- Initial registration uses only the `+z` source-up hypothesis.  A3/A4 invoke
  at most one real retry only when every available initial proposal fails the
  gate.  Preliminary v1 always retries the lexicographically best initial raw
  mesh with signed-axis registration alternatives `-z,+x,-x,+y,-y`, while
  retaining all triggering failure codes.  Every retry must create a new
  proposal ID, registered artifact, and transform/artifact hash.
- After the retry, A3 still accepts the lexicographically best available
  proposal but labels it `unsupported` when it remains below gate.  A4 instead
  abstains and selects no asset.  Abstentions and failures stay in the planned
  denominator.

The preliminary conjunction requires at least 200 observation points,
registration residual at most 0.04 m, scale ratio in [0.4, 2.5], visible
fraction at least 0.5, absolute support gap at most 0.015 m, support overlap at
least 0.15, valid collision with 1--32 usable convex parts, no settled sinking,
a stable settle with drift at most 0.03 m, and initial penetration at most 0.01
m.  Missing or non-finite evidence fails closed.  The required A4 registration
residual sweep is 0.01, 0.02, 0.03, 0.04, 0.05, and 0.06 m while every other
check and the candidate set remain fixed.

The deterministic construction probe samples 20,000 mesh points with seed 42,
uses the 0.03 m symmetric clip, a 0.01 m support band, and a 0.005 m initial
support gap.  It generates one deterministic convex hull and runs PyBullet at
240 Hz: two seconds with velocity zeroed, then two seconds free.  Sinking means
final AABB minimum z below -0.01 m; stability additionally requires drift below
0.03 m.  Every category uses the same mass 0.3 kg, friction 0.5, and restitution
0.1.  The contact coefficients are explicitly applied to both the object and
plane base links with `changeDynamics`, read back with `getDynamicsInfo`, and
sealed in each probe artifact; a mismatch fails the construction candidate.

E3 has one terminal construction decision per policy/object job; it has no
episode or reset dimension.  Manipulation episodes, reset banks, and task-level
success belong to E4, which consumes the frozen A0 and A4 selected-asset rows.
Every non-null selected asset carries both the raw mesh and raw visual Gaussian
path/byte-size/hash plus its registered transform and physical-probe artifact
closure.  The final aggregate seal authenticates every selected JSON and the
complete policy/scene directory roster, including empty-scene directories.

## Required implementation

Prefer a new package with no duplicated model wrappers:

```text
agents/orchestrator/
  __init__.py
  artifact.py
  job_graph.py
  controller.py
  policies.py
  evidence.py

robo/eval/
  agentic_ablation.py

configs/experiments/icra2027/
  agentic_policies.yaml
  agentic_jobs.yaml
```

### Common artifact contract

Every proposal record must include:

```json
{
  "freeze_id": "...",
  "scene_id": "...",
  "object_id": "...",
  "job_id": "...",
  "proposal_id": "...",
  "tool": "trellis|reconviagen|registration|collision|...",
  "tool_commit": "...",
  "input_hashes": {},
  "artifact_paths": {},
  "evidence": {},
  "decision": "pending|accept|retry|reject|abstain",
  "reason_codes": [],
  "parent_proposal_ids": [],
  "started_utc": "...",
  "finished_utc": "...",
  "wall_s": 0.0
}
```

Evidence values must be raw, named quantities, not one opaque score. At minimum archive:

- symmetric clipped registration residual;
- scale and bounding-box plausibility;
- visible fraction / observation support;
- support gap and overlap;
- initial penetration;
- collision validity / number of usable convex parts;
- isolated settle or drop result;
- missing-evidence flags.

### Decision log

Write append-only:

```text
outputs/icra2027/<freeze_id>/agentic/job_ledger.jsonl
```

A decision is reproducible from the frozen config and evidence. The controller may be deterministic. Do not add an LLM merely to justify the word agentic.

### Evaluation producer

Implement:

```bash
python -m robo.eval.agentic_ablation \
  --jobs configs/experiments/icra2027/agentic_jobs.yaml \
  --policies configs/experiments/icra2027/agentic_policies.yaml \
  --out outputs/icra2027/<freeze_id>/agentic
```

Required outputs:

```text
job_ledger.jsonl
selected_assets/<policy>/<scene>/<object>.json
agentic_ablation.csv
agentic_ablation.json
coverage_fidelity_curve.csv
retry_breakdown.csv
failure_transitions.csv
```

The main CSV must contain:

```text
policy_id
planned_jobs
accepted_jobs
build_coverage
f1_20
cd_cm
catastrophic_collapses
stable_fraction
runtime_minutes_per_scene
retry_count
abstain_count
```

## Selection and retry gates

Store all gates in YAML. The first frozen version should be simple and interpretable. Do not train a high-capacity selector on the evaluation set.

Suggested decision order:

1. schema/frame/unit validity;
2. scale plausibility;
3. registration residual and observation support;
4. support/penetration validity;
5. collision validity;
6. short physical stability test;
7. accept, retry, or abstain.

The final policy should be chosen on a development subset or hidden-oracle training scenes, then frozen before the 50-scene test aggregate.

## Critical negative result

Existing evidence suggests geometry F1 increases from 0.708 to 0.783 while isolated stability falls from 79.9% to 76.6%. Preserve this result. It motivates multi-objective evidence. Do not hide it or redefine stability to reverse it.

A successful A3/A4 should recover some physical validity without losing the geometry gain, or expose unsupported objects through coverage rather than pretending they are valid.

## Tests

- deterministic replay of a job ledger;
- no GT path accessible from controller code;
- fixed-priority row never reads residuals for choice;
- retry creates a new proposal ID and artifact hash;
- retry count is bounded;
- dependency invalidation affects only descendants;
- abstained jobs remain in the denominator;
- duplicate job/proposal IDs fail;
- an injected tool crash becomes a typed job failure;
- policy outputs are invariant to input file ordering.

## Fast smoke

Use 3 objects representing:

1. both proposals valid;
2. multi-view collapse, single-view valid;
3. both initial proposals invalid, retry succeeds or abstains.

The smoke must exercise all four terminal actions and finish in under 5 minutes using cached or synthetic artifacts.

## Slurm execution DAG

Run the full preliminary experiment with
`run/icra2027/submit_e3_noarray.sh`.  It enqueues 154 ordinary jobs in one
invocation and never uses a Slurm array:

1. one CPU E0 contract job and one dependent CPU inventory job;
2. one synthetic three-object smoke on the selected GPU/MVPY profile, used as
   an interpreter and CUDA gate rather than an RGB+ED correctness claim;
3. a complete real RGB+ED `observe` -> CPU `control` -> CPU `evaluate` pilot
   for frozen one-object scene `5748ce6f01`; the pilot control exits nonzero if
   its sealed controller shard has no jobs or no real proposal;
4. the other 49 GPU `observe` jobs, enqueued immediately but eligible to run
   only after the pilot `evaluate` succeeds, each followed by its CPU
   `control` and CPU `evaluate`; and
5. one CPU aggregate gated on all 50 evaluation jobs.

The total is 51 GPU jobs and 103 CPU jobs on `sof1-h200-[0-7]`.  Set
`E3_GPU_PROFILE=hala-a6000` for the exact Hala tuple `hala`, `batch/normal`,
`gpu:a6000:1`, `NVIDIA RTX A6000`, compute capability 8.6.  The default
`gcp-a100` profile retains the exact qrfh A100-80G tuple.  Every accepted job
ID is atomically added to
`outputs/icra2027/submissions/<freeze_id>/jobs.tsv`; logs, temporary files,
and caches also stay below the SimAny checkout.  The submitter prints the
exact read-only `run/icra2027/monitor_e3_noarray.sh --ledger ... --watch`
command after enqueueing the DAG.  It refuses a dirty checkout, a reused
freeze/output root, and all detected Slurm or `SBATCH_ARRAY_INX` array
contexts.

Every `control` job has a four-hour time limit.  The pilot's control and
evaluation IDs are reused in the 50-scene phase collections; the pilot is not
submitted a second time.  `E3_REQUIRE_NONEMPTY_PROPOSALS=1` is exported only
to that pilot control job, so caught tool crashes cannot release the fleet.

## Full-run target

- All object jobs used in the existing 457-object hybrid study.
- Prefer the complete 50-scene benchmark if retry artifacts are computationally feasible.
- Every A0–A4 row evaluated on the same planned object IDs.
- At least 20 genuine retry-triggered jobs before making a retry claim. If fewer occur, report the mechanism as a case study rather than a headline result.

## Claim gate

The agentic claim passes only when A4 provides one of:

- higher conditional fidelity at matched coverage than A0/A1;
- higher coverage at matched fidelity;
- a lower catastrophic/physical-failure rate at comparable fidelity;
- improved downstream manipulation in E4 under the same planned tasks.

Merely spending more compute and matching A2 does not pass.

## Acceptance criteria

- [x] A0–A4 are generated from code, not assembled manually.
- [x] Every decision has evidence and a reason code.
- [x] No runtime decision accesses evaluation GT.
- [x] Coverage and conditional quality are both reported.
- [x] Retry is a real additional tool action.
- [x] A threshold/coverage sweep is archived.
- [x] The paper title claim has a clear pass/fail conclusion.


Completed and independently audited in `20260906-357caca-v1` (source `1b3c516`);
all planned jobs, genuine retries and the threshold sweep are archived. The broad
A4/title improvement gate is **FAIL**; narrow selection and registration-retry
results are reported in paper `18def27`. See `STATUS.md` and the E9 claim ledger.

## Handoff

`STATUS.md` must state whether the title claim passed, list all retry actions and counts, provide the frozen policy YAML hash, and link the exact rows used by E4 and E9.


## Training-view prerequisite for a new automatic proposal pool

The legacy auto-discovery and splat-depth stages subsample the complete COLMAP
trajectory, which may include official held-out cameras. New construction must
filter the official training roster before applying the unchanged frame stride.
The opt-in `--train-split` path does that in the existing producers; it never
reads reference geometry or evaluation metrics. Legacy behavior is retained for
old commands, and existing artifacts are not retroactively certified.

```bash
python -m agents.discover.training_views --config configs/experiments/icra2027/construction_regimes.yaml --dataset-root /data/ScanNetpp --out <fresh-input-audit.json>
python -m agents.discover.derive_mesh_from_splat render --train-split <scene>/dslr/train_test_lists.json
python -m agents.discover.derive_mesh_from_splat fuse --train-split <scene>/dslr/train_test_lists.json
python -m agents.discover.auto_segment --scene-dir <scene> --out-dir <fresh-discovery> --mesh-path <construction-derived-mesh.ply> --train-split <scene>/dslr/train_test_lists.json
```

These are individual stages, not an authorized full-run launcher. Source/config,
input mesh, checkpoint, image hashes and exact-source E0 must still be frozen
before GPU execution. Pass the same training frame set to the existing
`factory_prepare --frame-allowlist` path. Never register on the evaluation
surface. Independent automatic multiview crops and raw generator provenance
remain required; this input audit alone does not close those gaps. Strict
rendering refuses an existing depth directory, discovery refuses an existing
instance output/manifest, and fusion rejects partial/stale/mislabeled depth
frames or a changed split before writing a new mesh.

## Automatic job inventory extension (engineering schema 2)

`agentic_automatic_jobs.yaml` and `agentic_automatic_policies.yaml` declare the
initial one-scene engineering adapter. The latter preserves every policy and
threshold from the existing policy file and changes only the study scope.
`robo.eval.agentic_ablation --inventory` selects the automatic adapter when
`automatic_sources` is declared. It checks the source discovery/pool identities,
retains every discovered ID as `scene_id/obj_<automatic_instance_id>`, and
records source IDs and prepared-output indices only in the audit sidecar.
The controller never receives legacy `gt_object_id`, labels, discovery scores,
or evaluation references.

Resolved inventory schema 2 adds explicit available/unavailable observation
status. Preparation failures have no RGBA, frame, mask, point cloud, or measured
point count. The canonical observe/control stages retain these jobs and emit
A0--A4 reject/abstain terminal rows without invoking a renderer or registering an
invented surface. Both initial tools remain represented for every job.

This adapter does not turn the v21/v28 UNKNOWN Gaussian lineage into a paper
result. A declared unrun initial tool remains `initial_tool_not_run`; v33
preserves that earlier incomplete inventory. The current source config binds
both completed v28 TRELLIS and v31 RVG pools, with all six planned jobs. Evaluation matching and generation-runtime aggregation must be
completed before any automatic-study table promotion. Existing schema-1 legacy
runs and their immutable outputs remain independently reproducible.

Smoke: `python -m pytest -q tests/test_agentic_ablation.py tests/test_agentic_missing_observations.py --basetemp=outputs/icra2027/test-tmp/automatic`.
Real inventory: `python -m robo.eval.agentic_ablation --jobs configs/experiments/icra2027/agentic_automatic_jobs.yaml --policies configs/experiments/icra2027/agentic_automatic_policies.yaml --contract-manifest outputs/icra2027/<freeze_id>/contract/freeze_manifest.json --freeze-id <freeze_id> --out outputs/icra2027/<freeze_id>/agentic --inventory`.

### Complete automatic scene populations

The same canonical inventory command accepts `automatic_sources` as a list of
the existing per-scene source declarations. It requires `population` to bind
`scene_roster_config`, its file SHA-256 (`scene_roster_config_file_sha256`), the
normalized roster SHA-256 (`scene_roster_sha256`), and integer
`planned_scenes`, `planned_jobs_per_policy`, `planned_policy_object_rows`.
The source list must exactly match the frozen roster order; complete per-scene
discovery manifests determine the object count. A zero-object discovery remains
in `resolved_jobs.scenes` and the planned scene denominator. Missing prepared
objects remain typed unavailable jobs with all five policy rows.

Per-scene hashes and tool provenance remain in `inventory_audit.scene_audits`;
the controller receives the same sanitized schema-2 inventory. The global
source-contract digest binds the complete jobs config. Single-scene configs and
their output schema remain compatible. A multi-scene engineering inventory is
not a full experiment or a paper gate: independent evaluation matching, complete
runtime accounting, and all declared treatment outputs are still required.

Smoke: `python -m pytest -q tests/test_agentic_automatic_population.py tests/test_agentic_missing_observations.py --basetemp=.t/automatic-population` (create `.t` first).
