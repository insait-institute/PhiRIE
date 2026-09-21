# N2 measured DEV handoff

## Current boundary — 2026-09-07 04:20 UTC

Owner root; integrated branch main through29281f6. L0 TEST construction is COMPLETE:48planned target builds,32built/16TRAIN-discovery failures; all32eligible shared pools complete, no remaining target generation. N3 consumes the same pool for B3/B4/BM. Final192method decisions:88READY/39ABSTAINED/65BUILD_FAILED, bound at `20260907-73ff4ab-v3/.../test_context/bindings_complete_ready_005`. Geometry152rows uses86receipts in `20260907-da23b9d-v5/.../test_fidelity/collection.json`. Candidate model/config/input hashes remain in immutable receipts described below. Final full policy outcomes remain running; construction completeness does not imply manipulation success.

Destination construction is COMPLETE for the declared24instance subset:20built/4TRAIN-discovery failures;20eligible shared pools complete at `20260907-48df68a-v1/.../shared_destinations` collection051. GPU840806/841046 and CPU recovery841028/029/030/114 all completed. Original resource timeouts retained in cost snapshot `20260907-48df68a-v3/.../shared_destinations_cpu_retry/cost_snapshots/001`. TRAIN-only anonymous workspace residuals use the existing producer in CPU841251 and are a separate N4 engineering stage. Next executable action: consume final candidate handles in already-running native engines and existing final table producer; no target regeneration or TEST threshold changes. Claim: construction feasibility PASS, agentic improvement NOT_RUN pending complete paired TEST. Earlier dates below are historical.

Owner: root. Branch `agent/icra-n2-candidates`; integrated on main. Source stages: B0 `98722dd5278b8179f37c48d39a70141204ecc2c6`; shared pool `a4d45f7bae87bac133cd734c02b88363afa742db`; evaluator `a0129bb`; table binding `41d9446`. State: PILOT_RUNNING, TEST NOT_RUN.

Smoke: native Python `-m pytest -q tests/test_roundtrip_shared_candidates.py tests/test_roundtrip_build_isolated.py` (22 targeted tests passed for shared stage); `tests/test_roundtrip_fidelity_native.py` (2 real MuJoCo/mesh tests passed, including negative scale and no-visual guards); `tests/test_native_scale_geometry.py tests/test_native_scale_tables.py` (12 passed). Smoke is not experimental completion.

Real build command: `.venv/bin/python run/sim_recon_sim/build_isolated.py --code <frozen worktree> --config <dispatch JSON> --capture <public capture> --out <private worker output> --receipt <isolation JSON> --forbidden <native vault> --forbidden <native API root> --bubblewrap <pinned bwrap>`. Exact argv, source/config/model/capture hashes and log paths are in each dispatch receipt below. Constructors cannot see the native evaluator, reference asset closure or held-out views.

Outputs under `/group/worldcept/code/SimAny/outputs/icra2027/`:

- Original missing-pair B0: `20260906-98722dd-v2/sim_recon_sim/scale_up/build_seed{1,2,3,4}`; jobs839497–839500. 3/4 built. Seed3 no usable automatic TRAIN target mask, preserved method failure.
- New eight-instance B0: v3 for layout11 (jobs839561–839564), v4 for layout12 (839583–839586). All eight terminal: 5 built, 3 automatic-mask failures (slot0b0,slot21e,slot767). No task outcome used to alter discovery thresholds or replace instances. Five successful builds are not eight successful builds. Native REF remains executable for all eight.
- Shared TRELLIS/RVG pilot: `20260906-a4d45f7-v1/sim_recon_sim/scale_up/shared_seed2`; job839579 completed4m29s on one Hala A6000. RVG available, 4 confirmed TRAIN views, producer12.478s and model wrapper61.435s. B0 reuses exact collision/mesh; B3 selected a genuine signed-source-up registration action with new transform/proposal ID. Residual6.306→5.833mm is construction evidence only. Native manipulation improvement NOT_RUN.
- New DEV shared pools: `20260906-a4d45f7-v2/sim_recon_sim/scale_up/shared/<slot>`; ordinary independent jobs839619–839623. Three failed-discovery slots explicitly NOT_RUN at this stage; no generation without usable observations.
- Independent geometry: `20260906-a0129bb-v2/sim_recon_sim/scale_up/geometry/<slot>/geometry_metrics.json`, CPU job839635 completed23s. All5 available B0 assets evaluated, 20,000 deterministic area samples per surface/seed2027, whole native visual target surfaces held out from construction. Canonical `robo.eval.fidelity_metrics.geometry_metrics`; no evaluator registration. Runtime2.93–4.98s/instance. CD0.630–1.315cm, F1@20 0.686–0.939, coverage5/8; these ranges are descriptive DEV conditional quality. Exact per-instance metrics and input hashes are machine-readable.
- CPU job839630 failed before log/script launch on gcp-us2-login, signal53; resource launch failure, no metric computed. Same computation resubmitted as ordinary CPU job on Hala839635; original stage preserved.

Source/checkpoint hashes: B0 `build_manifest.json.model_identities`, shared `rvg/pinned_models.json` and `rvg_receipt.json`; selection controller config SHA `5c33f01d8cfc46fc562cd9fa0d43767da07bd1bd484c51846d0ca5be35faf703`; isolation manifests retain every mounted source file hash and only allocated GPU device. Final release binds compatible stages; correct artifacts are not regenerated for a common source hash.

Passed: actual GPU isolation, native API exclusion, two-tool real pool, genuine retry transform, production CoACD export, frozen world geometry evaluation. Failed scientific examples preserved: automatic discovery3/8; original mango native B0 failure despite usable geometry. No B3 manipulation improvement claimed.

Remaining: finish all5 shared pools and a second-shape native B3 import, complete per-stage method cost attribution, observed-surface/TSDF control, public role extraction for destination/support, held-out appearance, native collision/visual camera overlays. Native geometry metrics do not establish physical frame correspondence or physics fidelity. Target-only retained-native context/uniform appearance only; no GS/full-room claim.

Claim gate: operational construction/geometry pilot PASS; agentic improvement NOT_RUN. Next executable action: consume candidate_pool.json selected handles in N3/N5, finish 80-unit REF/B0 roster after adapter state-size fix; extend B3/B4/BM from exactly these pools. No TEST release yet.


## 2026-09-07 measured completion and TEST dispatch

DEV candidate/geometry/appearance pilot complete for all5available of8planned builds. Shared pools5/5, observed TSDF5/5. Authoritative complete paper release `20260907-f1ef144-v1/sim_recon_sim/scale_up/paper_tables`; no-mask failures remain coverage losses. B0 primary3/40success vs B3 7/40, both25/40executed, descriptive only.

Held-out native appearance: source2c197a2 renders fixed2heldout cameras without state advance; native RGB identity controls pass. Output stages2c197a2-v2 and1a71f74-v1; exact-image PSNR handling9296360, CPU839996 at388346b-v1. Existing evaluator computes metrics; fullframe native background limits interpretation. Geometry B3 F1slightly belowB0, retained.

TEST initial B0 source2c197a2, stage20260906-2c197a2-v3; all48ordinary jobs dispatched,12built/8no-mask failures/28unresolved at00:17UTC. GPU→CPU shared pool split sourcebbc61c4, stagesv1/v2, one RVG generation per canonical then isolated CPU registration/selection. Source/checkpoint/config/TRAIN identities remain bound. No TEST policy outcome used for construction.

Commands: `python <stage>/test_builds/dispatch_available.py` and `python <shared-v2>/dispatch_available.py`. Private job JSON/scripts contain full actual argv and hashes. Next: resume only missing shared pools and supply frozen selected artifacts to native-context probes and policy engines. Optional workspace-role construction remains separate incomplete work, not declared complete by target-only outputs.


## 2026-09-07 TEST construction terminal boundary

All48 B0 attempts terminal:32BUILT,16proven automatic-TRAIN discovery failures. Source2c197a2/buildstagev3 reused; sharedstagebbc61c4 GPU generation then CPUselection, remaining7RVG/CPU pairs840378–391 submitted,30/32 pools available at01:05. New producer metadata fixe636bd2 uses referenced OBJ bounds; existing correction133cca7 preserves exact mesh/T/scale/physics/collision and changes onlyworld_dims. No old asset rewritten or model regenerated for metadata alone.

First TEST geometry CPU840151 PASS6s (sourceae09e35), cohort split bound to build/canonical. Geometryfull stage3b095ad-v1; accepted B4/BM actual selected geometry source78409d7/da23b9d, pilot840286 PASS2s with cachednativepoints, noalignment; contextstages1750c68-v1 and7daa2ef-v1. Exact source/capture/checkpoint/metric hashes remain in receipts. Additionalfidelity orchestration delegatedN7 while rootmaintains generation dispatch. No TESTquality numbers fed tocontroller.

Actual sink role construction840369 BUILT2:48,16parts, source d8607c1 (N4 role extension toexistingfactory); target/control sourcesunchanged. B3sinkRVG840416reuses SAM/TRELLIS/capture. Workspace/Cabinet role recovery is implementation work, notcompletedfromtarget-only assets.
