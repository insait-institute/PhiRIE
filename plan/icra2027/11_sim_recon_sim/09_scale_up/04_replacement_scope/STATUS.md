# N4 — replacement scope: current authoritative status

- Owner: `/root/e4_endpoints`. Tested source branches: `agent/icra-n4-test-scope` (e459a10), `agent/icra-n4-l2-route` (168f1b5), `agent/icra-n4-l2-coverage` (1ed493a), `agent/icra-n4-l2-import` (00ec440). Frozen runtime worktrees are unchanged.
- State: TEST L1 FULL_RUNNING on the complete1200-row phase; L2 complete scope NOT_RUN. Partial four-role DEV engineering completed with finite dynamics and a negative initialization gate; it is not a completed L2 benchmark. Bounded L1 DEV integrity PASS; no population scope-benefit claim is enabled from partial results.
- Authoritative TEST phase: `/group/worldcept/code/SimAny/outputs/icra2027/20260907-6a60a51-v6/sim_recon_sim/scale_up/test_l1_execution`. Common scope_run_id `native_test_L1_engine_v2_20260907-6a60a51-v6`; full roster SHA256 `121819ef11e4ff4eb2bfe53b718c2a185aa3ad6e43a7e908fb4b9f3a5d2022ea` is recorded verbatim in contract/budget_v2.json (use that authoritative receipt). Each of24instances has10resets × REF/L0B3/L0B4/L1B3/L1B4. Original720fresh controls allocated L1; later L2 requires up to720 additional fresh controls, total maximum2400scope+control units. Earlier1680 budget and unexecuted superseded stages remain historical.
- Latest execution snapshot at04:25UTC: `execution_snapshots/0047/episode_ledger.jsonl`,1200planned,659terminal,59executed. First canonical840963 is complete (REF10/10success, L0B38/10, L1B30/10); later canonical measurements continue. These are partial population results, not a complete scope-benefit comparison. No target, scene, reset or destination was selected using policy success.
- Ordinary scope submissions:24/24 complete; construction watcher251344 consumed final N3snapshot018 and exited normally, including failed-destination blocks with mandatory fresh controls. Individual `dispatch/native-*.submission.json` receipts are authoritative. Sourcee459a10 remains immutable. All submissions are ordinary jobs, managed by the shared global GPU controller; the independent execution collector remains active after construction dispatch exits.
- Independent canonical final collector PID371902 runs `/group/worldcept/code/SimAny/.venv/bin/python <phase>/dispatch/watch_execution.py`, with PYTHONPATH pointing to frozen n4-test-scope. It publishes immutable `execution_snapshots/NNNN/{episode_ledger.jsonl,status.json,collection.json}` on terminal changes until all1200terminal or24h. Source/script/config hashes in `dispatch/execution_watcher_receipt_002.json`. First launcher368229 failed ENVIRONMENT (missing PYTHONPATH) before outputs; original log retained. Collection continues when all24construction submissions finish.
- Observed supports840843 COMPLETE8:35:2/2 TSDF components built, original strict import1/2 (source part14 invalid, destination10parts valid). Source0a41628; `/group/worldcept/code/SimAny/outputs/icra2027/20260907-c12bcaa-v1/sim_recon_sim/scale_up/observed_supports/components/workspace_components.json`. No silent repair.
- Explicit new source-support repair841071 COMPLETE1:44,4CPU8GB, source168f1b5: one declared per-part collision-regeneration action; repaired16parts pass strict importer and real MuJoCo compile. Original visual/estimated transform/physics bytes unchanged. `/group/worldcept/code/SimAny/outputs/icra2027/20260907-4132cd9-v1/sim_recon_sim/scale_up/workspace_repair/{repair/workspace_repair_receipt.json,dispatch/strict_import.json}`. Command `python -m robo.roundtrip.workspace_repair` with frozen dispatch arguments. This new artifact is a separate observed diagnostic method, not a retroactive B3 repair.
- TRAIN-only sampled workspace coverage841084 COMPLETE14s,4CPU8GB, source1ed493a:36,828samples;14,535unknown;1,146unclassified surface samples in46residual components, none discarded. `/group/worldcept/code/SimAny/outputs/icra2027/20260907-1e36d8c-v1/sim_recon_sim/scale_up/workspace_coverage/coverage/workspace_coverage.json`. Unknown space is diagnostic uncertainty, NOT a B3 accept/reject threshold; original diagnostic admission wording is superseded in source00ec440 by completeness NOT_ESTABLISHED and controller_decision=null. No obstacle-absence or complete approach/transport coverage claim.
- Four-role native engineering841108 COMPLETE53s/exit0, ordinary4CPU16GB15min, source00ec440. `/group/worldcept/code/SimAny/outputs/icra2027/20260907-5cf76bd-v2/sim_recon_sim/scale_up/observed_workspace_native`; command `python -m robo.roundtrip.workspace_native_gate --config <stage>/dispatch/config.json --out <stage>/native`. B3target+plate remain identical; add repaired observed source-support and observed counter top. Actual200step unchanged role control precedes imported-scene settle and all-native substep contacts. Counter base/doors and all unreplaced native entities remain explicit. Partial source bottom does not recover unseen basin walls. Zero policy episodes from this CPU gate.
- Evaluator-only native inventory840952 COMPLETE2:23, source24c7f2e:1447native collision geoms,127 conservative workspace AABB/plane candidates across9root bodies. `/group/worldcept/code/SimAny/outputs/icra2027/20260907-6a60a51-v8/sim_recon_sim/scale_up/native_workspace_audit/inventory/native_inventory.json`. Zero physics steps; no GT entity or geometry feeds constructor prompts/masks/selection/placement.
- Existing DEV scope worker840729 PASS (REF400/L0B3346/L1B3358 success ticks); actual SAM3 workspace840687 yields4role clusters, obstacle0masks unresolved; Cabinet840661 is generated-shelf/privileged-target engineering only. Cabinet target840579 genuine empty-SAM3-mask failure preserved. All24TESTdestination ordinary jobs840601–616 and840731–738 remain in N3 construction accounting.
- Smoke: repair2PASS0.80s; coverage+repair4PASS0.83s; native workspace/import/probe34PASS1.87s, including exact factored native-probe equivalence, transformed-counter unchanged identity, tamper rejection and unknown-role negatives. Reproducible command: native Python `-m pytest -q tests/test_roundtrip_workspace_import.py tests/test_roundtrip_scope_verification.py tests/test_roundtrip_fixture_scope.py tests/test_roundtrip_scope.py --basetemp outputs/tests/workspace-probe` at00ec440.
- Actual841108 results: unchanged source/counter support identity200steps PASS with exact integration bytes; imported four-role scene executes250steps finitely. Initialization FAIL: penetration71.97197mm, target drift130.558mm/rotation153.237deg, plate drift15.444mm, retained distractor drift8.020mm. All-native audit records23,634contact events,2,008outside declared bounds. Maximum contact is retained `sink_1_main_group_g30` versus new source-support `sink_1_main_group_main__roundtrip_geom_3`, step133. Frozen roster/XML show g30 under explicitly retained Shelf002, not duplicate removed direct basin geometry. No geometry repair is selected from this evaluator observation; no B3 rollout veto is inferred from this B4-style conservative probe.
- First actual TEST L1 r0 paired diagnostic: same process/reset REF327success, L0B3420success, L1B3600failure/error=null. L0 target height increase139.18mm and max displacement699.57mm; L1 height increase0mm and max displacement35.91mm; minimum gripper distance659.10mm;0recorded robot-target/finger-target-contact ticks. Reconstructed destination collides robot link1 with64.7007mm maximum penetration at tick4/substep4. This is measured pre-acquisition interference evidence, not an inferred grasp boolean. Source-hashed report and all continuous video/result/trace paths: `<TEST phase>/diagnostics/first_declared_r0/diagnosis.json`. One fixed reset is diagnostic, not a completed population result.
- Same-policy partial workspace pilot841163 COMPLETE7:07/exit0 onHala1A6000/8CPU64GB. Source7eace81,46tests PASS2.62s; existing scope_pilot/paired/canonical harness reused. All3fresh same-engine episodes executed: REFsuccess359ticks, L0B3success356ticks, DEV_PARTIAL_WORKSPACEfailure900ticks/full declared horizon; errorNULL for all3. Engine `scope-engine-60e94122728141b88f3ddf0ecb361fda`, process1594d6fb-8cc0-4c20-b91b-36751c61e3e2. Same canonical/reset/RNG/checkpoint verified. The owned server terminated normally. This finite B3 diagnostic was executed despite the conservative initialization failure; unknown voxels were not a veto. It remains separate from24TESTscope/1200rows, not a completed L2 benchmark.
- Actual pilot root `/group/worldcept/code/SimAny/outputs/icra2027/20260907-c2dc6cf-v2/sim_recon_sim/scale_up/dev_workspace_policy`; immutable `pilot/scope_pilot_summary.json` and `diagnostics/{diagnose.py,diagnosis.json,videos.json}`. Three continuous videos are `pilot/{REF,L0_B3,DEV_PARTIAL_WORKSPACE}/episode/continuous.mp4`, respectively18.00/17.85/45.05s and360/357/901frames; no episode splicing. Partial resultSHA7955f5e5dcaf125a7afb9ffb600e60db811b24806a9978fd89e37f4318ef1dae. Reproduce via frozen `dispatch/pilot.sh` in a new stage; original source/config/output remain immutable.
- Measured partial failure: target rises at most9.08mm and moves129.26mm from its own first recorded post-action pose; L0 target rises408.77mm and moves735.90mm. Partial minimum gripper distance212.45mm; zero reconstructed-target/robot-contact ticks in22,500native substeps; object_in_receptacle never true. This establishes failed target acquisition/transport, not a fabricated grasp label or uniquely isolated per-component cause. Native physical time stays monotonic, maximum2ms increment deviation2.45e-15s; recorded policy qpos/qvel finite. Warning counters and substep acceleration were not recorded.
- Partial contact evidence:1,813,930native contact events,180,000outside the declared workspace (includes retained-context pairs). Maximum penetration71.97098mm at tick5/substep8, retained sinkShelf002 geometryg30 versus reconstructed source_support geom3; maximum force37,159.03N at tick0/substep2, retainedg21 versus source_support geom7. Retained shelf collisions are explicit context, not duplicate removed basin geometry; no GT-based deletion or constructor tuning follows. Scope expansion has an actual negative DEV result; population/full-L2 claims remain disabled.
- First TESTscope canonical840963 COMPLETE31:41/exit0: all50planned cells terminal, REF10/10success, L0B38/10success, L1B30/10success; inherited B4abstentions10L0+10L1 are unexecuted/null. Source-hashed full-block record `<TEST phase>/diagnostics/first_declared_r0_numerical/completed_block.json` binds execution_snapshots/0018. Construction dispatcher consumed final N3snapshot018 and exited normally with24/24canonical jobs submitted; execution collector371902 continues until1200terminal.
- Numerical safety audit of firstL1r0:600policy ticks/15,000native substeps, exactly25per tick; time increases from0.5 to30.5s, maximum expected-time deviation6.03e-12s and tick/substep boundary deviation0. Recorded qpos/qvel all finite; no shape change; checked worker/scheduler/server logs contain no BADQACC/BADQPOS/BADQVEL or instability warning matches. No actual auto-reset evidence; labels remain unchanged. Boundary: native warning counters and substep qacc/qpos/qvel were not recorded, so this does not certify every internal warning. Source-hashed audit/script: `<TEST phase>/diagnostics/first_declared_r0_numerical/{audit.json,audit.py}`. No rerun, modified threshold, speculative validity flag or frozen-worker edit.
- Residual obstacle construction841251 COMPLETE14:50/exit0, source9080a79, ordinary4CPU16GB60min on gcp-eu1-rtx6000-26hb. All46frozen TRAIN residual components retained:43actual maskedTSDF/CoACD assets;3 BUILD_FAILED for insufficient confirmed TRAIN views (IDs16/33/41). Strict independent static asset importer plus real MuJoCo compile admits34/46;9 built assets fail closed-convex collision checks (IDs6/13/20/22/24/29/30/32/35). Original artifacts, failed parts, and CoACD concavity-cap warnings remain unchanged. Total578collision parts produced; fused+collision component runtime600.14s. These are geometry construction/admission diagnostics, not obstacle recovery or native policy outcomes.
- Residual receipts: `/group/worldcept/code/SimAny/outputs/icra2027/20260907-a394b38-v1/sim_recon_sim/scale_up/residual_obstacles/{completion.json,artifact_admission.json,binding_assessment.json}`; authoritative producer `construction/components/workspace_components.json`, SHA256bc28a92676050bb78667a62f2012b5c28f781c34c1199fd90d9e32cacc0fd32b. Reproduce the frozen command with `bash <stage>/dispatch/build.sh` in a newly reserved stage; submitted paths remain immutable. Empty-root CPU preflight PASS with no GPU/native API access. Smoke21PASS0.25s includes all-component retention, shared producer execution, and leakage/threshold negatives.
- Exact remaining L2 boundary: the declared target/destination/source-support/destination-support contract resolves4existing roles;0 anonymous residual components have justified one-to-one semantic native bindings or static/free body semantics. No native geometry/GT association, arbitrary body deletion, or duplicate native collision is introduced. New artifacts remove the missing obstacle-geometry implementation gap; TRAIN-only semantic association of split/merged fragments and an explicit relevant-obstacle inventory/context boundary remain required. All46unclassified IDs and14,535unknown samples are retained. Unknown occupancy is not a B3 rollout veto. Full L2 remains NOT_RUN, and partial DEV841163 is measured only under its explicit four-role engineering scope.
- Claim gates: existing DEV L1 integrity PASS; TEST L1 coverage measured partially, benefit NOT_RUN; complete L2/L3 NOT_RUN; obstacle absence and universal geometry/scorer preservation NOT_RUN. All config and checkpoint hashes live in immutable submission/config/engine receipts. No hand-edited paper numbers.

## 2026-09-07 observed support construction launch

- Source0a41628, ordinary CPU job840843:4CPU/16GB/30min, gcp-eu1-rtx6000-26hb, no GPU devices. Both declared source/destination support clusters from840687 are frozen before execution. Existing masked TSDF2mm/6mm and CoACD producer is shared with the original observed-object control, with explicit workspace cropping only for this new method.
- Stage `/group/worldcept/code/SimAny/outputs/icra2027/20260907-c12bcaa-v1/sim_recon_sim/scale_up`; command `bash <stage>/dispatch/observed_supports.sh`; submission and isolation receipts in dispatch. Expected `observed_supports/components/workspace_components.json`; build result is not claimed until that receipt is complete. Empty-root receipt records devices=[], network unshared, no unsandboxed fallback,36inventory mounts and41source files.
- Method `OBSERVED_WORKSPACE_COMPONENT_TSDF` is a partial observed-surface diagnostic. It is not B0/B3 generation, complete support shape recovery, a complete obstacle inventory, or an L2 outcome. Native role mapping and policy remain NOT_RUN. Failed component construction stays in the two-component denominator.
- Source tests:39passed16.39s (workspace_components, observed_surface, build_isolated, scope_bundle). Every-physics-step contact labels additionally verified with native interpreter:4passed1.50s, including transient contact and byte-exact physics identity. General constructor .venv lacks robosuite; native tests require `/group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python`.

## 2026-09-07 completed observed components and prospective TEST scope budget

- CPU840843 **COMPLETED8:35, exit0**, source0a41628:2/2observed supports built. Source support16CoACD parts (37.792s fusion+collision), destination support10parts (12.683s). Cold shared-library I/O explains most job wall time. Both visual meshes remain partial/nonwatertight, explicitly disclosed.
- Strict static importer audit after component freeze: source support FAIL `part_14.obj` not a closed convex volume; destination support PASS10parts. All stored artifact hashes match. Thus2/2producer builds,1/2strict import admissions,0native episodes. Do not silently repair or drop the failed source part. Authoritative `20260907-c12bcaa-v1/.../dispatch/component_integrity.json` and `observed_supports/components/workspace_components.json`.
- Before any TEST L1 episode, `configs/experiments/sim_recon_sim/scale_up/scope_engine_budget_v2.yaml` declares the process-lifetime amendment: original720fresh controls are allocated to L1; L2 requires up to720additional fresh controls. Scope outcomes remain960; maximum total2400units=960scope+1440controls. Earlier1680budget and original scope subset are preserved. No L1/L0 reference may be reused across the later L2 engine.
- No idle policy engine is reserved for unfinished L2. Current inherited target statuses permit 460executable controls per level before runtime failures (240REF+150B3+70B4); the remaining260controls inherit original target build failures/abstentions, not native policy failures. L2 remains unmeasured until implemented and admitted; no fabricated construction failure from missing code.

## Historical evidence log (superseded state fields below are retained provenance)

## Implemented boundary

`import_reconstructed_object` accepts rigid target/receptacle/support/obstacle roles, either original free-joint semantics or static bodies. Metric aligned.json transform is applied once. Static parent/world conversion uses the parent frame only on the evaluator side. Articulation, moving parent chains, mocap bodies and ambiguous pose encodings fail explicitly. Existing native geometry/sites are removed, frozen contact/material priors retained, and reconstructed bounds/contacts refresh native movable-object handles.

`robo.roundtrip.scope.import_scope(xml, entities, scope)` implements bounded L0 and L1 (one target plus one container or support). It produces source-bound receipts, complete original-body/direct-world-geom inventory, retained oracle-context labels, and `replaced_joints` keyed by joint name. It is a pure XML transaction; existing assets and environment are untouched on failure. `bind_scope_objects` handles existing native movable target/container roles after adapter import. Generic fixture-specific scorer sites remain unsupported, explicitly rejected by this binding path.

A container's supplied convex parts are imported individually, preserving openings when those parts represent them. `robo.roundtrip.receptacle_probe.probe_entry` runs an unassisted small-sphere drop with predeclared observed start/interior bounds. It separately reports entry, final contact and terminal speed; a failed diagnostic never silently substitutes native geometry. This is a single-path cavity diagnostic, not a whole-mesh containment or task-success metric.

## Validation

- Smoke: **45 passed in 1.04 s**. `/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q tests/test_roundtrip_importer.py tests/test_roundtrip_scope.py tests/test_roundtrip_paired.py tests/test_roundtrip_build.py`.
- Cases: transformed-parent support world placement; unchanged robot subtree; multiple free bodies imported in reverse order retaining named joint order; duplicate/missing L1 role; articulations/moving parents/mocap rejection; missing/invalid collisions; asymmetric scale/quaternion; removed original collision and external references; open-container entry vs closed-hull negative; exact unchanged destination physics; integration-state dimension guard.
- Actual native geometry control: `/group/worldcept/code/SimAny-wt/n4-scope/outputs/n4-scope-smoke/native-destination-identity-v1/import_identity.json` and `actual_imported.xml`.
- Actual source: preserved `20260906-4ee6462-v2/sim_recon_sim/reference/pilot/canonical_seed0` native CounterToSink scene; destination subtree `sink_left_group_main`, including unchanged faucet articulation.
- Result: MuJoCo **3.3.1**, **200 × 0.002 s** saved-control integration steps, **20 compiled-field groups equal**, integration states byte-exact, qpos and qvel max absolute differences **0**, runtime **3.5885 s**. This is an engineering U1 control, not an additional policy episode. Rendering and native predicate identity for this role remain NOT_RUN.
- Source native XML SHA-256: `6d5c5c16f7a66ee8859789d7cae9cc13ab365747154b882b234a0c0799a565bf`.
- Reproduce with the native interpreter and a JSON list `[{"body_name":"sink_left_group_main","role":"receptacle"}]`: `PYTHONPATH=. /group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python -m robo.roundtrip.scope --xml <canonical>/scene.xml --canonical-state <canonical>/canonical_state.json --entities <roles.json> --out <new-directory>`.
- N0 scorer control independently rerun: `PYTHONPATH=. /group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python -m robo.roundtrip.scorer --out outputs/n4-scope-smoke/scorer-frame-control.json` — PASS. Physically identical frame re-expression changes raw body-origin success, while fixed justified scorer transform restores it. Generated-asset absolute correspondence remains NOT_ESTABLISHED.
- Hardware: local CPU; no Slurm jobs or checkpoints loaded.

## Remaining pilot/full work and claim gate

- Pilot command/result: NOT_RUN pending a sealed observation-derived destination container. N0/N5 named multi-joint adapter hook is delivered in `b30be80` (depends on `dc08424`) and must be merged before native use. First supported prospective L1 task is `PickPlaceSinkToCounter`; native counter support remains explicitly retained reference context.
- Current N1 target-centered 6-TRAIN/2-held-out captures are not automatically a destination/workspace scan. Container visibility and automatic segmentation must be demonstrated from public data; no native crop or mesh enters construction.
- Whole sink replacement remains unsupported: the actual sink subtree contains articulated native parts. The identity clone does not imply reconstructed basin support.
- Native SinkToCounter predicate is object/container contact plus XY body-origin distance below `0.7 * reconstructed container horizontal_radius`, container/counter contact and gripper distance. It is not whole-mesh containment. Origin/radius correspondence can affect raw success, and strong retention claims remain NOT_RUN.
- Full L1/L2 4-layout scope matrix: NOT_RUN. No scope downgrade or L0 fallback is counted as L1.
- Continuous reconstructed L1 native episode, native renderer/predicate identity, retained-context contact logging and independent containment diagnostics remain outstanding.
- Scientific claim gate: NOT_RUN. Smoke controls do not establish manipulation benefit or room reconstruction.


## 2026-09-07 bounded role-wise B3 continuation

- Owner: `/root/e4_endpoints`; code branch `agent/icra-n5-role-agent`; tested commits63ea7e4 and17683f4. Existing scope/importer and canonical rollout ledger are reused. No L2 implementation was added under another name.
- Completed hybrid L1: first predeclared layout12 SinkToCounter lemon/reset0, native success397steps, errorNULL. Same L0 B3 target artifact and reset had success348steps. Result `/group/worldcept/code/SimAny/outputs/icra2027/20260906-0c404ab-v2/sim_recon_sim/scale_up/l1_scope/slot-15e3e2ffaabc11f8f956/runner/episode/result.json`; explicit variant `hybrid_scope_diagnostic_target_B3_destination_B0`. This is one DEV integrity pilot, not population benefit evidence.
- Full role-wise B3 bundle support: `python -m robo.roundtrip.scope_bundle` now accepts `--destination-candidate-pool <selection/candidate_pool.json>` alongside its matching B0 role build manifest. The unique A3/B3 accepted row is fixed by the constructor; native results never select a destination. Proposal role-prefix/canonical/cohort/capture identity, source B0 and RVG receipts, selection ledger and imported artifact hashes are validated. Rejection/abstention/missing selection fails without B0 fallback. The unchanged target is checked against the completed same-process L0 B3 import.
- Resumed CPU pools may specify `--destination-rvg-receipt <original/rvg_receipt.json>`; its exact hash and original B0 binding must match the pool. Existing hybrid bundles remain unchanged; full role-wise variant is `B3_target_B3_destination_L1`, with both entity constructors recorded B3. Room/support oracle context and binding-specific native scorers remain explicit.
- Smoke:20 scope/paired tests passed2.97s;21 scope/isolation tests passed1.27s;25 static/importer tests passed1.82s. Negative cases include wrong role/canonical/proposal dependency, abstention, changed selection ledger, changed pose artifacts, static moving parent and articulation. Actual completed slot21 receptacle pool passed read-only closure validation; it is not a native task result.
- Matching lemon original pool839880 timed out after10:11; complete RVG generation preserved. CPU-only selection resume840024 uses source17683f4,4CPU/8GB/30min, noGPU. Stage `/group/worldcept/code/SimAny/outputs/icra2027/20260907-3e2a818-v1/sim_recon_sim/scale_up`, command `bash <stage>/dispatch/select.sbatch`. Central allowlist and cached-RVG hashes passed; original registration logs/timing remain intact. Completed initial CPU registration was repeated11s rather than trusting an unsealed cached-call contract; no RVG regeneration. Missing retry/materialization is still running at this handoff.
- Full-B3 L1 real pilot remains NOT_RUN until matching pool completion and a same-process policy-engine L0/L1 comparison. The old warm service has expired; old REF/L0 cannot be silently paired with a new engine.
- Exact L2/static assessment: `<stage>/scope_capability_assessment.json`. Static rigid XML import under static parent chains is tested, but native fixture-specific handles/sites/bounds/scorers are not bound by the movable-object route. Constructor role assets currently support only receptacles, not support/obstacle recovery. L2 lacks a complete declared approach/transport workspace inventory and retained/outside-workspace contact accounting. Whole sink/cabinet articulation is unsupported; native fixture identity cloning does not establish reconstruction.
- Scientific scope/room-retention claims: NOT_RUN. Actual generated-container opening probe and whole-mesh containment remain NOT_RUN; absolute scorer correspondence NULL. No native geometry or pose may substitute for missing public support/obstacle evidence.


## 2026-09-07 completed B3/B3 L1 and workspace evidence

- Full role-wise B3 pilot PASS: ordinary GPU job840179 completed5:07 exit0, source82af20c, fresh single policy engine. REF succeeded359steps; L0B3 succeeded356; L1B3 succeeded388. All three native episodes have errorNULL, same engine process UUID and canonical reset. The canonical importer verified unchanged target fields across L0/L1. Policy same-process interleaving and18 full-chunk repeat gates passed; prior real DEV engine admission was hash-validated. The owned server terminated after all three episodes and released itsGPU.
- Authoritative `/group/worldcept/code/SimAny/outputs/icra2027/20260907-4a21559-v2/sim_recon_sim/scale_up/scope_pilot/scope_pilot_summary.json`. Reproduce with `bash <stage>/dispatch/scope.sbatch`; source helper `robo.roundtrip.scope_pilot` only sequences the established server/gates and canonical paired runner, with no new episode ledger. Failure before server readiness retains all3planned episodes and no synthetic rollout.
- Matching B3 destination pool840024 completed11:37 onCPU, no RVG regeneration. It invoked a genuine bounded registration retry and ultimately selected original TRELLIS, reason `best_available_below_gate`. Pool hash836ff2f53ab47a513e0cdbd740dbfccca37beb2b708f3cae7771c44d23105f05. This is a B3 controller execution, not evidence that retry improved this plate. Existing hybrid L1 result remains unchanged. One canonical pilot provides no population benefit claim.
- Public workspace preparation implemented in `robo.roundtrip.workspace`: metric backprojection reuses existing capture helpers, consumes every sealed TRAIN frame with deterministic stride4 and a declared world extent. It produces source-hashed surface samples plus explicit planned semantic-role slots; it does not label geometric crops as object masks or recovered obstacles.
- Actual CPU preparation `/group/worldcept/code/SimAny/outputs/icra2027/20260907-4a21559-v3/sim_recon_sim/scale_up/workspace/workspace_manifest.json`:6/6TRAINframes contain points in the declared region;152200 sampled surface points. Extent comes only from existing automatic TRAIN target/plate point clouds plus frozen padding. `L2_READY=false`, semantic roles remainNOT_RUN, obstacle inventory completeness and approach/transport coverage are not certified. Thus public depth is available; semantic support/obstacle construction and native fixture/contact integration are missing implementations.
- Source82af20c tests:30 combined workspace/scope/paired/local-engine tests passed2.25s;2 scope allocation/dead-engine negatives passed0.48s. No hidden reference assets or held-out views are read by workspace preparation. Static-destination fixture binding is the next separate gate; current free plate result is not relabeled as static or L2.

## Static sink component extension — 2026-09-07

Owner `/root/e4_endpoints`, branch `agent/icra-n4-static-paired`; factory source
`d8607c1` remains frozen in `/group/worldcept/code/SimAny-wt/n4-workspace`.
The existing rigid-asset importer now authors an observation-built basin shell
inside the unchanged static native fixture frame. Generated estimated world
placement and mesh scale are preserved. Fifteen direct physical geoms are removed;
four faucet/accessory child bodies and two invisible goal/metadata geoms remain
explicit privileged context. Native goal regions are the unchanged evaluator
rubric, not generated-interior correspondence or constructor inputs. No Cabinet,
L2, whole-fixture, opening-preservation or strong task-retention gate is promoted.

Smoke: native Python `-m pytest -q tests/test_roundtrip_fixture_scope.py
tests/test_roundtrip_scope.py tests/test_roundtrip_scope_bundle.py
tests/test_roundtrip_paired.py tests/test_roundtrip_scope_pilot.py --basetemp
outputs/static-paired-tests-v3`: **39 passed**, including transformed Euler parent,
world-vertex preservation, static ancestor rejection, stale-geom references,
retained child hierarchy, target-plus-basin inventory, unchanged native regions,
positive/negative pinned-native scorer probes and sealed public-role matching.

Actual CPU control: `python -m robo.roundtrip.fixture_scope --xml <first-DEV-whisk
canonical>/scene.xml --state <same>/canonical_state.json --body-name
sink_left_group_main --out <stage>/native_sink_identity`. Stage:
`/group/worldcept/code/SimAny/outputs/icra2027/20260907-f638a0a-v3/sim_recon_sim/scale_up`.
`native_sink_identity/import_identity.json`: **PASS**, 6.856 s, 200 integration
steps byte-exact, zero qpos/qvel difference, 20 compiled field groups equal, 8
native scorer probes identical with positive and negative outcomes. This is an
actual native scene physics control plus explicit scorer queries, **zero policy
episodes**; renderer observation identity remains NOT_RUN. Prior `f638a0a-v1`
control failed from a missing quaternion in the scorer-probe mock (code bug), log
preserved; the source fix and rerun received new commits/freezes.

Actual observation-only sink basin factory: Slurm **840369**, ordinary one A6000,
Hala, 4 CPU / 64 GiB / 15 min, source `d8607c1`, RUNNING at 00:52 UTC. Exact
submission, source/model/config identities and command are recorded in stage
`dispatch/sink_basin_submission_840369.json`, `dispatch/sink_basin_build.json`,
`dispatch/build_sink.sh`; constructor isolation excludes native private data.
Expected output `sink_basin_build/build_manifest.json`. Public instruction maps
`sink` to the explicitly declared `sink basin` phrase. Failure is a failed planned
destination component, never automatic L0. Pilot/full native static-scope episodes
remain NOT_RUN until a sealed generated component is available and executed.

The canonical paired route now supports the exact native `env.sink` task fixture
binding, avoiding random substring lookup. Both target and destination use the
existing import/state-restoration and episode ledger. First B3 target / B0 basin
is explicitly `hybrid_scope_diagnostic_target_B3_destination_B0`; B3/B3 requires
the independent shared destination pool and its sealed selection. No hand-edited
physics, reference-pose snapping, goal-region expansion or rollout fallback.

## 2026-09-07 actual static scope outcomes and Cabinet acquisition

Owner `/root/e4_endpoints`; diagnostic branch `agent/icra-n4-scope-diagnosis`,
producer commit `11f531f`. Previous dated RUNNING/PENDING entries above are superseded
by the following immutable receipts. No source frozen for a queued job was edited.

- Static B0 basin build **840369 COMPLETE** (2:48), source `d8607c1`,16 CoACD parts.
- Initial hybrid **840414 FAILED** (4:00), source136533f: native handle mapping
  referenced removed `sink_left_group_g0` during XML reset. Its REF273success and
  L0B3 600failure are preserved, but L1 did not execute. The fix stages declared
  fixture handles before native XML reset; no native geometry substitutes for
  removed basin geometry. Original stage `20260907-3b70b68-v5` is immutable.
- Corrected hybrid **840538 COMPLETE** (5:26), source `533d0c2`, fresh engine:
  REF292success; L0B3 600failure; L1hybrid600failure. All3 executed with errorNULL.
  Same-engine summary `/group/worldcept/code/SimAny/outputs/icra2027/20260907-2c0570a-v1/sim_recon_sim/scale_up/sink_scope_pilot/scope_pilot_summary.json`.
  Native preflight in that stage passed29.502s with all unreplaced qpos byte-exact,
  compiled fixture handles valid and zero policy episodes.
- Descriptive trace diagnosis (not a new headline metric): generated target settles
  down8.90mm in the first recorded tick in both reconstructed arms. Relative to its
  own saved initial state, max displacement is9.75mm L0 and31.57mm L1; neither rises
  above initial height or ever enters the native sink goal. REF transports35.25cm
  in XY and succeeds. Continuous videos show failed acquisition/no destination
  transport. Contact pairs/forces were not recorded, so a particular grasp/contact
  failure mechanism remains unresolved. No grasp/lift success rubric was invented.
- Retained sink distractor has identical pre-step coordinates in all arms, but L1
  moves it55.41mm in the first tick and133.90mm maximum; L0 maximum is0.035mm.
  This is an actual post-reset scope interaction, not a reset-coordinate mismatch.
  Diagnosis is source-hashed at `/group/worldcept/code/SimAny/outputs/icra2027/20260907-9ff689e-v2/sim_recon_sim/scale_up/sink_diagnostic/scope_trace_diagnostics.json`.
  Reproduce `python -m robo.roundtrip.scope_trace_diagnostics --pilot <840538-stage>/sink_scope_pilot --out <new-directory>`.
  Smoke `python -m pytest -q tests/test_roundtrip_scope_trace_diagnostics.py`:
  **3 passed**, including missing-tick/empty-trace negatives and saved-initial-state
  baseline rather than first-tick displacement. CPU only, no rerollout.
- Shared RVG basin **840416 COMPLETE** (1:21 GPU), selection **840468 COMPLETE**
  (12:29 CPU). B3 selected initialRVG after a genuine bounded registration retry,
  reason `best_available_below_gate`, support `unsupported`; no retry-benefit claim.
  TrueB3 scope **840568 FAILED** (1:00), source533d0c2, before policy startup:
  selected01/collision/part_14.obj is watertight/positive-volume but fails strict
  convexity. No episode executed and no candidate substituted. Source pool
  `20260907-5a44f1b-v1/.../shared_sink/selection/candidate_pool.json`; failure
  `20260907-8148028-v1/.../native_import_preflight/failure.json`.
  B3 generated-artifact validity failure remains in coverage. Any new decomposition
  is an explicit additional repair action/new artifact, not a silent B3 correction.
- Cabinet native bottom-support control source175cda9: 200 integration ticks
  byte-exact and24 positive/negative unchanged native scorer queries PASS. Cabinet
  capture **840527 COMPLETE** (28s),12TRAIN/2held-out, fixed DEV-only camera trajectory.
  New canonical `native-ebd03717bc44d977095645d19ba99c9cf1386bea428a8f171c4988ae09b63c0b`,
  captureSHA `a61b1e3dbed827a10d20dcbe2b1a43ed742d5c5432b21fcd4a09cbde3830bacd`.
  Binding `20260907-5a44f1b-v2/.../cabinet_instance/binding.json` records all camera
  identities. No native target pose or geometry entered acquisition trajectory.
  Fresh target **840579** and bottom-shelf **840580** queued from sourceb7ca613,
  ordinary Hala1A6000/4CPU64GiB30min, explicit newly captured identity; no target
  reconstruction reused by seed. Config/commands/submission receipts at
  `20260907-b7ca613-v1/.../dispatch/`. Bottom surface replacement retains native
  walls/doors/upper shelves; whole-container/cabinet coverage is not claimed.
- TEST scope subset is genuinely predeclared: original `test.yaml` commit2cb7700
  (2026-09-06T23:32:22Z) fixes layouts1–4, all3families and2instances/layout:
  24canonical ×10resets ×2methods(B3/B4) ×2scopes(L1/L2)=960planned measurements.
  It is distinct from N5's missing96-reset diagnostic predeclaration. Execution
  requires sealed TRAIN-only destination construction and fresh same-engine
  controls; original primary REF cannot be paired across policy processes.

Claim gates: bounded native L1 execution PASS (free plate and hybrid static sink);
static B3 basin artifact validity FAIL; scope manipulation benefit NOT_RUN;
Cabinet generated-scope execution NOT_RUN; complete L1/L2 matrix NOT_RUN; room
retention/opening preservation/absolute scorer correspondence NOT_RUN. Negative
native outcomes are valid scientific measurements and remain in denominators.


## Before-first-job control scheduling correction

The complete phase includes all24 canonical control blocks regardless of destination acceptance. Every REF reset executes even when both constructed target methods are unavailable; accepted L0 methods execute even if their corresponding L1 destination failed. No control is permanently omitted based on a scope outcome. The unexecuted preparation phase6a60a51-v2 (1200planned/400typed failure shards/zero native episodes) is preserved and superseded before any GPU job because its preliminary admission receipt incorrectly required one READY scope arm. The next immutable phase has the same scientific roster/budget and corrected scheduling admission. L2 remains separately unmeasured, not fabricated as failed.
