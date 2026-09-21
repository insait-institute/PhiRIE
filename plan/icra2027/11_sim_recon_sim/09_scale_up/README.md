# Native reconstruction scale-up: from one working pair to testable conclusions

**Status: TODO / prospective execution plan, not new results.** Written 2026-09-07 after reading `PhiRoom/e40dbdbf0922d357cf12c4b7c93289d955fb79c0` and `../STATUS.md`. Paper repository: `RunyiYang/SimAnyRoom`. User-facing method names: **Agentic Construction** and **Task-Conditioned System Verification**.

This directory is the active scale-up entry point. It supersedes the *future RoboCasa budget and dispatch order* in the parent plan, not its data-isolation requirements or any executed experiment. Preserve the original ScanNet++ cohorts, first native milestone, prior failures, and source receipts. Do not repeat the first milestone merely to create another report.

## Starting evidence, not a population claim

The recorded DEV milestone used one target-only ideal RGB-D reconstruction, fixed TRELLIS, native uniform-color rendering, and the original room/destination as oracle context. Five native reference episodes ran, four succeeded; only ONE reconstructed pair was measured, with reference/reconstruction success at 392/429 ticks. The 392-action reconstructed replay failed the terminal retreat predicate. The unchanged-native import replay matched all 392 steps. No Gaussian room, B3/B4 benefit, broad preservation, or full-room reconstruction was demonstrated. Authoritative details and source paths: [parent STATUS](../STATUS.md).

## The objective

Obtain a reproducible multi-instance native-policy comparison, establish whether evidence selection and room-context repair improve interaction at declared coverage/cost, and progressively replace oracle context. Success of the work means complete valid measurements, not a forced positive hypothesis.

First executable deliverable: complete the remaining first-pilot reference/reconstructed pairings where compatible canonical assets exist, preserving the failed reference seed. Then run **8 independent DEV builds, 40 paired resets per arm**. Do not wait for BEHAVIOR, Harmonizer, a learned verifier, or an entire room renderer to produce these results.

## Read and dispatch

Read [AGENT_PROMPT.md](AGENT_PROMPT.md) and [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md), then assign one worktree per owner:

| ID | Owner/task | Main output | Dependencies |
|---|---|---|---|
| N0 | [Protocol, identities and scorer](00_protocol_and_identity/README.md) | Versioned cohort schema, canonical/reset bindings, U0/U1 and scorer contract | Existing native milestone |
| N1 | [Cohort capture and batch execution](01_cohort_and_batch/README.md) | Canonical instances, sealed captures, ordinary-job dispatcher | N0; coding may start immediately |
| N2 | [Shared candidate construction](02_shared_candidates/README.md) | Reusable B0/B3 candidate artifacts and observed-surface baseline | N1 captures |
| N3 | [Room-context verification/repair](03_context_verification/README.md) | B4/V1/BM decisions, bounded native-context repairs | One N2 import; N4 hooks |
| N4 | [Replacement scope/import](04_replacement_scope/README.md) | Target, destination and workspace native adapters | N0; existing target importer |
| N5 | [Action replay and paired policy](05_replay_and_policy/README.md) | Native task outcomes and matched-horizon physics traces | N0/N1; N2/N3 per arm |
| N6 | [Gaussian observation intervention](06_gaussian_observations/README.md) | Same-state native/GS observations; optional Option C | Frozen import contract; independent of core native runs |
| N7 | [Tables, conclusions and demo](07_tables_claims_demo/README.md) | Generated tables, claim decisions, continuous evidence videos | Scaffolding now; results incrementally |

The [configuration template](cohort.template.yaml) is deliberately non-executable until actual scene/task IDs, checkpoint receipts, horizons and pins are resolved. `EXECUTION_STATUS.md` is a TODO board, not evidence that a task has run. Each owner adds its own `STATUS.md` only with actual commands and artifacts.

## Fixed experimental stages

| Stage | Layouts x tasks x instances x resets | Resets per arm | Arms / scope |
|---|---|---:|---|
| DEV expansion | 2 x 2 x 2 x 5 | 40 | REF + B0 first, L0 target-only |
| TEST core | 8 x 3 x 2 x 10 | 480 | REF/B0/B3/B4/BM, L0: 2,400 planned episodes |
| Verification-only control | Same TEST roster | 480 | V1, separate optional addition |
| Scope extension | 4 fixed TEST layouts x 3 x 2 x 10 | 240 | B3/B4 at L1 and L2: 960 additional episodes |
| GS observation block | Same declared scope subset | 240 | B4_GS adds 240, reuse matching B4_NATIVE |
| Full-room cases | 2 predeclared cases | Separately budgeted | L3, no automatic full-room claim |

All counts are proposed budgets, not observations or power guarantees. Core L0 is a **target-only oracle-context diagnostic even at 48 instances**. Scene/workspace claims require the separately reported scope extension. Do not run a Cartesian product of all backends, sensors, scopes, policies and seeds. A second policy, RGB-only, Harmonizer and BEHAVIOR are independent extensions, not blockers.

## Immediate dispatch and stopping rules

1. N0 fixes only validity-critical contracts. In parallel N1 inventories existing canonical states and N7 prepares empty tables.
2. Release REF/B0 DEV jobs as soon as their targeted identity/capture/import smoke passes. N2 builds candidates; N4 develops destination/support import; N3 works against actual native collision.
3. While DEV runs, finish N0-N5 integration. Freeze TEST roster, gates, material policy, horizons and generation seed before TEST outcomes are read. Launch native TEST by independent instance jobs.
4. N4/N3 release L1/L2 on the fixed subset; N6 adds GS without delaying native comparisons. N7 publishes each completed declared block and later binds one release manifest.
5. An environment import bug pauses only affected units. A finite but unsuccessful scene is scientific evidence: do not globally suppress baseline rollouts because a conservative verifier predicts failure. Numerically unsafe/load-invalid states terminate with typed coverage loss.
6. Do not wait for a statistically positive result. If a method is worse, preserve the result and narrow its claim. Fix independently diagnosed software defects in a new version; never silently alter TEST thresholds, cohort or official success semantics.

## Resource and provenance rules

**Ordinary independent Slurm jobs only, no job arrays.** Inspect current allocation and queue, use permitted idle resources, never cancel unrelated jobs, and stay within account quotas. Keep policy/model services warm. Reuse hash-matching captures and candidate outputs once per canonical instance, not per reset/arm. Use per-episode shards and a single validated merger, not concurrent writes to one JSONL. Request compute explicitly; do not run heavy experiments on login nodes.

Reuse `robo.manifest`, `robo.eval.harness_runner.run_native_episode`, existing native adapters/policy, metrics and `robo.eval.paper_pipeline`. Stage commits may differ: bind compatible identities in an immutable release manifest instead of rebuilding everything for a common cosmetic Git hash. Old outputs never change.

## Definition of done

- [ ] Remaining DEV pairings and 8-instance DEV matrix measured with retained failures.
- [ ] Native scorer/frame, canonical reset, full-horizon replay and concurrency contracts tested.
- [ ] 48-instance TEST core has one terminal outcome per planned unit, or an explicit incomplete budget report.
- [ ] B0/B3/B4/BM costs, coverage and native outcomes are comparable; BM is not mislabeled exact FLOP matching.
- [ ] L1/L2 scope extension is complete or clearly separated as unavailable, never called room-scale based on L0 alone.
- [ ] Tables use standard metrics and source-bound sample counts, with cluster-level paired uncertainty.
- [ ] Genuine continuous videos and conclusions reflect only measured scope.
- [ ] Paper and code commits, claims and artifacts are cross-linked. No unsupported result is manufactured.
