## 2026-09-06 11:40 UTC — scoped full evidence transferred to paper

Canonical publication20260906-8cae8a0-v1/source93c9566 and pushed paper83c97b1 now include this task's available full output, preserving all denominators, nulls and failed gates. 389focused/130exactE0PASS;8+8page QA PASS. See E9 STATUS for commands and hashes. Task-specific remaining measurements and limitations below remain open; publication does not promote the scientific gate.

---

## 2026-09-06 11:06 UTC — E2 full paired-view metrics and E4 full terminal audit complete

Live audit `outputs/icra2027/orchestration-status-20260906T110628Z/status.json`; scheduler terminal receipts adjacent. No SimAny E2/E4 jobs remain running or pending. Other user jobs in squeue are not this project. Code before this status update99ca6ce; paper remains18def27, clean and not yet updated with these outputs. This is completion of the current execution waves, not all project TODOs.

E2 source7a1f85b/freeze20260906-b689700-v1: all48remainingrender/copy jobs COMPLETED0, alongside two passed pilot units; canonical metric job835594 COMPLETED43m00 onSOF3. `fidelity/public_factorized/metrics/receipt.json` PASS, all metric seal members independently rehashed here with zero mismatches; LPIPS backend error null. All50scenes/400plannedviews remain: raw available400/400, composite320/400, analysis uses identical320views across40scenes for both methods. Those40 include36positive cleaned backgrounds and4zero-object NO_REMOVAL backgrounds, not40successful interactive simulators. Descriptive paired-support means: raw PSNR21.581642249757337/SSIM0.8444029330571121/LPIPS0.2993210877291858; composite20.437798054410234/0.8318254927240997/0.3167900979751721. Composite-minus-raw deltas -1.1438441953471035/-0.012577440333012446/+0.017469010245986294, all worse direction. No paired-significance assertion or threshold tuning. TableSHA d99f0f11bde74fff8959548c874bbdff8fadb2ef8b5eb4eb1e900d3ec024ed06; coverageSHA393cebf34b0285540f94f84cde5815257446fe2200a1a2d0ecd28b868ab18cb0. Geometry/GT-discovery/Harmonizer rows remain unmeasured, paper_ready=false. Exact canonical command is in metrics/receipt.json: `python -m robo.eval.fidelity_metrics --manifest ... --out ... --lpips-device cuda --bootstrap-samples2000 --bootstrap-seed42` (CLI flags and values are separate argv entries in receipt).

E4 source05184e1/freeze20260906-b9bdcf3-v1: all47remainingindependent scene cache jobs COMPLETED0 plus3passedpilot cases. Publication/independent metadata audit835518 COMPLETED10m36 onSOF6. `completion_audit.json` PASS; SHA8142867d109474b579ce3073fb88be4c1a1b094f25b2dbf42826081136cae3d7; three sealed members rehashed here. Full50/1871 cohort:6155inputqueries,5886predeclaredbudgetexclusions,269selectedqueries,2690logicalqualificationcells.1670prerequisitechecks FAIL,1020source-unavailable cells NOT_RUN; zeroqualified, zeroactual900-stepchecks, zeroexecutedpolicyepisodes; policy success null, not0%. Scene states23QUALIFICATION_COMPLETE/16PLANNING_UNAVAILABLE/2EXPORT_REJECTED/9SOURCE_NO_QUERIES. These are prerequisite/applicability failures, not failed policy rollouts. PriorOOM/collision/no-tabletop failures and834890timeout remain preserved. Exact audit command in completion_audit.json invokes e4_terminal_cache_publish.py validate-full with05184e1 and the frozen config/output.

Claim gates: E2 overall appearance-improvement claim FAIL on observed conditional means; E4 construction/task applicability FAIL and manipulation claim NOT_RUN. Narrow E3 evidence-selection/registration-retry claims remainPASS. E6 predictive applicabilityFAIL; Harmonizer/physicalsuccess/finalheroNOT_RUN; finalsubmissionFAIL. Next critical path is canonical E9 transfer of authenticated E2/E4 outcomes and claim-safe prose, then build/page/font/render QA and paper push. E1other input families/geometry/runtime, external E5base access, additionalphonecapture and genuinephysicaltrial gaps remain; do not declare projectcomplete.

---

## 2026-09-06 08:58 UTC — real recovery pilots pass; full ordinary jobs released

E4 cache pilot835440/835441/835442 all COMPLETED (8m36/2m49/1m18); independent cached-result equality and original-source metadata gate PASS in20260906-b9bdcf3-v1/dispatch/pilot_completion_gate.json. Remaining47 ordinaryCPU jobs835471–835517 released from unchanged05184e1 source/E0; final canonical publication/independent metadata audit835518 depends on all47. Latest live counts21COMPLETE24RUNNING2PENDING, no new failures. Expected `full_qualification/` and `completion_audit.json`. Every logical cell and previous construction failure remains; policy success is null.

E2 repaired pilot835460CPU/835461GPU COMPLETED48/90s, independent scene_result validation PASS: fixed09c has8raw/0composite views and predeclared0d has8/8. Gate20260906-b689700-v1/dispatch/pilot_completion_gate.json; no quality-based admission. Remaining48ordinary jobs835532–835537 and835552–835593 released (39GPU9CPU);The exact authoritative roster is dispatch/full_submission.json. Paired metric job835594 waits all48 and runs existing `python -m run.icra2027.e2_public_factorized --phase metrics`, which invokes pinned `robo.eval.fidelity_metrics`; all50scenes/400plannedviews remain. Latest11COMPLETE3RUNNING34PENDING, no measurement failures. GPU allocation used discovered idle SOF2 and GCPA100 cards first, then explicit one-node/one-task H200 requests. Held835538 was canceled before release after resource validation caught a multi-node nodelist request; exact rejection snapshot retained, corrected replacement835552 preserves same scene/source/config. No seven-GPU measurement occurred and no array was submitted.

Main source/status pushed after review; paper remains18def27 pending completed fullE2/E4 publication. Portable current-paper archive and original E1 aggregate are complete as recorded below. Final scientific submission remains FAIL; narrowE3 gatesPASS, unsupported manipulation/Harmonizer/physical claimsNOT_RUN.

---

## 2026-09-06 08:48 UTC — full E1 aggregation and E2 backgrounds complete; recovery audits active

Live receipt `outputs/icra2027/orchestration-status-20260906T084810Z/status.json` plus sacct/squeue snapshots. Artifact checkout402cda0 and paper18def27 remain unchanged. No common final scientific freeze and no claim that all TODOs are complete.

E1 sourcecd9e8ff/freeze20260906-3e8ab6d-v1: all50ordinary835082–835131 COMPLETED0; canonical aggregation835404 COMPLETED55s, noGPU. Generated `construction/table/construction_table.json`:1871inputs,399controller accepts,394verified exports;44successful scenes,4empty,2preserved room rejections.361/394 independent historical-protocol isolated drops stable; yield0.21058257616247997 and stability0.916243654822335. F1/fullruntime null;valid_for_paper=false. Four other input families remain NOT_RUN in250scene/regime records. Original E3 acceptance probe and E4 room stability are distinct and preserved. Command `python -m run.icra2027.e1_full_drop --config STAGE/execution.json --stage STAGE --expected-commit cd9e8ffdfcae2a32d0d880b8f27e9cad04b2a5c8 --aggregate`; exact launch in STAGE/launchers/aggregate.sbatch.

E2 source8a09ebe/freeze20260906-0be94c7-v1:39new fills/no-ops835033–835071 COMPLETED; independent835077 PASS1m55.40bundles include originalpilot,36cleaned positive scenes and4unchanged zero-object backgrounds;10TRAIN-blocked/80missingTESTviews retained. No new held-out quality metrics yet. Full-render43dcd9e/v2 pilots835438/835439 failed before target render publication: PythonABI overlay mismatch (environment) and missing explicit original VERIFIED_REUSE receipt branch (codebug). New source7a1f85b repairs canonical original-source reuse validation;83focused/130exactE0 PASS; actualE0 4f17532d81db9a7bb6404c49d16778fc8ff68b825636856412542d255caf5cde. Newfreeze20260906-b689700-v1 pilots835460/835461 use unchanged original50/400cameras,36positive/4noop/10blocked scope and correct pinned Python3.10 runtime. No threshold or metric changes; full48followups wait realpilot validation.

E4 original terminal834890 TIMEOUT52m06, extension rejected by Slurm; source18cbd0d/freeze20260906-c3ba498-v1 preserved with no sealed aggregate. New05184e1/freeze20260906-b9bdcf3-v1 uses original-source independent per-scene replays and strict sealed cache composition,73focused/130exactE0 PASS, actualE0 dfb9d98da83d28fdbcb4875eb748502b8bd144a271c389eae212c324207d75f5. Ordinary CPU-only pilot835440/835441/835442 covers original prerequisite/no-tabletop/protected-carve cases; two are already COMPLETE. All50 original outcomes remain fixed, no construction or policy measurement. Full release awaits all3pilot seals.

E9 portable producerb71413d/freeze20260906-313ec4a-v1 completed current paper18def27 archive:311numeric fields,20claim decisions,266members; standalone relocated standard-library verification PASS.64focused/130exactE0 PASS; actualE0 c6990e2bde05809a5531c3a567fd86359f59a89c66750e80a4a7c03e68831857. Canonical `python -m robo.eval.paper_pipeline --bundle-existing ... --paper-root ... --publication-receipt ... --out ...` command in PORTABLE_BUNDLE.md. Archive `SimAnyRoom-audit-18def27.tar.gz`,53253760bytes,SHAee26cd841def89ef577edfb744893d3e8509291269c967e49907e1b7753c212b. Originalf0d3ee7/v2 packaging failure (numeric-leading scene key parser) is preserved; newsource/freeze fixes it. Publication remains20260906-6dcb0e7-v2,8+8pageQA. Paper has not yet been updated with new E1/E2/E4 outputs.

Claim gates: narrow E3 evidence selection/registration retry PASS; broadA4 superiority and E6 predictive applicability FAIL; manipulation/Harmonizer/physicalsuccess/finalhero NOT_RUN; finalsubmission FAIL. Root continues locally after subagent provider quota errors; these are orchestration limits, not scientific failures. No arrays and no silently replaced measurements.

---

## 2026-09-06 04:35 UTC handoff

trellis2_backend / root; agent/icra-e2-common-view-analysis; PILOT_PASSED; Exact common8 metric and paper source audits PASS; SMOKE_PASSED; PILOT_PASSED; NOT_RUN; 1186635 metrics;6b40e24 publisher; none;833824 COMPLETE; 20260906-0281856-v1/dispatch/HANDOFF.json; One-scene diagnostic only;full method matrix incomplete; PASS paired source integrity;NOT_RUN population improvement

Exact commands, Slurm receipts, hardware/runtime and source/config/checkpoint identities are in the central04:35 milestone and immutable linked contracts. Processing integrity does not imply scientific claim success.

## 2026-09-06 04:08 UTC handoff

trellis2_backend; agent/icra-e2-public-factorized-metrics; PILOT_PASSED; Fixed two-scene renders and metrics independently sealed; SMOKE_PASSED; PILOT_PASSED; NOT_RUN; 16642fc; none;833453 COMPLETE; 20260906-36be09d-v1/fidelity/public_factorized; Composite8/16 versus raw16/16; predeclared common8 descriptive analysis pending; PASS metric provenance;NOT_RUN improvement

Commands, hardware, runtimes and exact config/checkpoint hashes remain in immutable contracts and the central04:08 milestone. Scientific claim gates are distinct from successful artifact processing.

## 2026-09-06 03:36 UTC orchestration handoff

trellis2_backend; agent/icra-e2-public-factorized-pilot; IMPLEMENTING; Two render/typed-result seals complete; SMOKE_PASSED:89focused/130E0; CLAIM_FAILED runtime probe routing; NOT_RUN; 4462a9b; none;833264 FAILED; 20260906-59df24a-v1/fidelity/public_factorized; Consumer probes absent runtime module in archived metric checkout; metrics-only new-freeze repair; PASS render integrity;NOT_RUN quality comparison

Exact commands, E0, runtime/checkpoint hashes and prior outcomes remain below and in the referenced immutable contracts. See central EXECUTION_STATUS for current namespace/admission repairs. No failed scope is promoted to a full scientific result.

## 2026-09-06 03:07 UTC — DROID stages closed, compact fill closed, full-input E4 protocol frozen

Integration **50a9fff** is pushed; paper **b2c6cf431772240c71237682e0a7bb1b9d7d137d** remains the generated8+8-page working draft. Artifact checkout402cda0 is clean and unchanged. Live source/job receipt `outputs/icra2027/orchestration-status-20260906T030752Z/snapshot.json` and adjacent`squeue.txt`. No common final scientific freeze or final submission exists.

**E0 regression:** immutablee3c223b job833100/Hala4CPU64G/noGPU completed fulltests in330.94s:2444PASS,2FAIL,4SKIP. Both failures were environment-only (unpinnedOpenPI import andAF_UNIX temporary pathname length); only those2 reran with correct pinnedPYTHONPATH and short repo-localbasetemp, bothPASS2.15s. Exact JUnit cross-check confirms2446distincttestsPASS/4SKIP, **not one all-green invocation**. Imports retainonly4knownoptional failures; configuredcanonicalharness15rowsPASS. Gate `SimAny-wt/integration-global-hala-20260906/outputs/global-integration/combined_validation_gate.json`. OldGCP833009 canceled16m23 after repeatedremote metadata stalls, logs preserved; earlierloginSIGKILL cause remainsunconfirmed. No scientific producer was rerun for these environment corrections.

**E7 GS full stage:** sourcef36f12c/freeze20260906-dba8e6b-v1, all10slots audited:5GScomplete/0GSfailed/5missingTRAINfitNOT_RUN. Pilot832922 and832983–832986 allCOMPLETED0:0. FullauditSHA1f613e32f39a6ee1b0ac7ceaf149f0b5e25083237d6c1361fc000a2d2efcee1d. **Public tail full stage:** source6fbdb1f/freeze20260906-f68957d-v1/E09ca0dc7b8622034572e205fd2e24b1a61cde942edd1a99b8249a4b986076408d. All35ordinaryphasejobs terminal:32COMPLETED,2FAILED,1CANCELLED. Tenworkspace audit4processingPASS/1FAIL/5NOT_RUN. FirstIPRL/RAIL/PennPAL produced0instances; secondIPRL6discovered/6prepared automatic bottle-labeledinstances, not independently verified distinct realobjects. Failed/unavailableinstancecountsNULL. FullauditSHA6f1aa64ae7b02aeb3e382f917759a03228685d41a2428d70fa471d287261fde6. RPLemptyTSDF caused missingmesh/discoveryfailure833082 andfinalizerfailure833084;833083cancelled. Classifications:scientificemptyreconstruction pluswrapperpostconditiongap, notOOM. Narrow fix1e470e8 merged50a9fff:79focused/130E0PASS (digest528ee1c3281fbeaef1e8684214cde21cd464a2c3fc52fede68a1116cbba58686); actualreadonly4nonemptymeshesPASS/RPLmissingFAIL. Frozen6fb runs andnegativeoutcomesunchanged. These stages are not fullsimulatorbuilds, policies, or independent imagefidelity.

**E2 compact background:** freeze20260906-4cf66bf-v1/source9602fc7/E005e8eead5c4c0fa862d02ed47f50e02920091479c1f005b80cf0078d0238c86a. HalaGPU833066 COMPLETE34s:27dd retained7planned/1acceptedfilledobject/3TRAINviews, unchanged1500iterations; cleanbackgroundSHAd91b3c309705a0e18e48be45849f92003b2bfdda27561112e7c0d09bc274f16c. IndependentintegrityPASS. CPU833067 exit1:0/5m51 correctlysealed40aec BLOCKED_UNFILLABLE_ACCEPTED forNO_PLANEcabinet, no cleanbackgroundorrawfallback. Its support-evidence failure is a scientificbuild limitation, notcodefailure. Oldreadinessfreeze20260906-f203ef7-v1 missingPydantic retained; repairedvia existing immutableoverlay, noenvironmentinstallation. Newcompositeconsumer usesexistingrenderer/metrics and willretain16plannedTESTviews including8unavailable40aecviews; originalrawpair830706/830709 fromcbba8ff/20260905-3ff95f4-v2 availableforauthenticatedreuse. Onlyalreadysealedcompactpair TEST scope permitted; fullE3GTgateunchanged.

**E6:** fixed0023clean grounding smoke complete:render832996,fuse833011,discover833014,finalizer833068. FinalindependentQASHAa043424dbbfc1d9a8badafe6ba6afcef297c660c44e8f438471a1e39255ba551. Fourqueriesretained:allmanipulatedrolesunresolved,4sourcesupportrolesresolved,4observedreceptaclepatches, noverifiedrobotframe/physics. GCPfinalizer833016 canceled9m50 whileonlyreadonlymetadatavalidationran; confirmednooutputandresumedonlymissingCPUfinalizerHala833068/27s. Originalmodelsealsuntouched. Fixed0020clean/mild/severepilot20260906-e3c223b-v1/E095998b026ae4088d6c85fed83270fb27b2603af3ee7710658ccc07a26ac7017a/sourcee7c5265:render833103–105PASS,CPUfuse833109–111PASS,discovery833115/833145/833146 andfinalizer833147+remainingtrackedbyagent. Fullcohortreuse/admissionwrapperunderimplementation; fulljobswait12-querypilotintegrity. RGBcohortremains14PASS4FAIL/18conditions72queries; noLOSOclaim.

**E4 next protocol:** b5997db mergedine3c223b,11focused/130E0PASS (8f36231b29319ddb2281276cea3aa1965bf40cc24df0be3e2538600191549a70). Generated`configs/experiments/icra2027/e4_full_protocol.yaml` authenticates50scenes/1871jobs/6155inputsemanticqueries. Input-onlyfixedfirst4taskIDs/family/scene selects269plannedqualificationqueries/2690reset-armcells,5886budgetexclusionspreserved. Aftercanonicalgeometry/script/cameraqualification, first4qualifiedscenes×4tasks×5resets gives80episodes/arm; no learned-policy selection orreplacement. Fewerthan4qualifiedrooms→NOT_RUN withoutrelaxedgates. Oldcompact40cells/0rolloutsfailureunchanged. Existingcanonicalqualifier/materializers are beingextendedwithauthenticatedselection; no newrolloutformat. RealexecutionwaitsE3all50integrityandnewstageE0/smoke/pilot. CorrectedharnessREADME/launcher48c6e1b nowexecutes`bash run/harness/run_paper.sh --smoke CONFIG NEW_OUT`, actual15-rowpositive andmissing-outnegativePASS/130E0PASS.

**E3/E9:**49/50controls sealed;832352 RUNNING,832535→832850 dependencyqueued. No fullGTmetrics/improvementclaimyet. E9nextpublishesavailableclosedDROIDstagesandfullE3results throughcanonicalpaper_pipeline; allpaperexperimentalnumbersremainmachinegenerated. Currentpaper7tables/23artifacts/199claimrows;8+8pages/fonts/renderQA unchanged. Finalhero90/30/8sNOT_RUN, genuine6sselection/retryengineeringclips preserved.

Claimgates:PASSstageintegrity/rawroom50×8coverage/DROIDCPUpublication/workingdrafttypography;FAILlegacyE1provenance/oldE4applicability/finalscientificsubmission;NOT_RUNfullE3improvement/manipulation/Harmonizer/LOSO/strictfullcapture/physicalsuccess/finaldemo. ExternalblockersremainCosmos403,missinggenuinephysicalpairsandadditionalphonecaptures. Nextcriticalpath:E3fullcontrolaudit→fullindependentmetrics→tables/claims; concurrentlyE4qualificationadapter,E2heldoutcompositepilot,E6fullgroundingandE9publication.

---

## 2026-09-06 02:32 UTC — RGB cohort closed, DROID Gaussian continuation and resource-separated grounding active

Integration source **f203ef7abe2bcaa7506e532db503836338abe4fe** and paper **b2c6cf431772240c71237682e0a7bb1b9d7d137d** are pushed; artifact source402cda0 unchanged. Exact live snapshot: `outputs/icra2027/orchestration-status-20260906T023100Z/snapshot.json`; scheduler states in adjacent `squeue.txt`/`sacct.txt`. No common final scientific freeze exists.

**E6:** freeze20260906-2fa8928-v1/sourceb2c827a, all18RGBconditions/72queries independently closed:14PASS/4FAIL/0INCOMPLETE/0NOT_RUN. Four failures remain unchanged floor-normal verification failures (832829/830/831/834);0/72features and no hidden-GT/LOSO evaluation. Final audit `audit/cohort_final`, independentQA SHA890549bffff812d5787ac6a7296bf343977dfb372f3f240e5d14c34d9a747d49. Resource-separated grounding sourcee7c5265 passed71focused/130E0, merged into main. Fixed6-frame0023clean smoke20260906-44f707c-v2, E0 1847932db1f84dab58fcb24880ea594099dce75b1894126a323c3d9e55a6aa70; ordinary HalaA6000 render832996 RUNNING, expected `audit/public_grounding/behavior_task0023_clean/phase_render`. CPUfuse→GPUdiscovery→CPUfinalize follow authenticated receipts. Unsubmittedv1 launcherAPI bug preserved, corrected before model execution.

**E7:** fixed30000step GS pilot832922 COMPLETE14m22, sourcef36f12c/freeze20260906-dba8e6b-v1; independentQA PASS (SHA465a14655a46e4fe0a6a1b68b810de9e78e559a0d456b8e9b73aa0da0e870232). Engineering measurements:1281707Gaussians,852.156683s trainer,2361976832bytes peakCUDA. Remaining original TRAIN-eligible jobs832983RAIL/832984RPL/832985PennPAL/832986IPRL each Hala1A6000,8CPU32G45min, ordinary noarray.832984 COMPLETE5m09 pending independent full audit; other3running. All10slots remain, including5missingTRAINfit NOT_RUN; held-out RAIL/RPL failures did not alter eligibility. GS completion is not strict full-build or independent image fidelity. Local SAM3/TSDF/factory tail is being implemented in separate resource phases.

**E2:** masks832970/832971 COMPLETE53s/57s, sourcefddb149/freeze20260906-a0a1aa3-v1;15views/17objects retained. Explicit LaMa CPU jobs832979/832980 COMPLETE5m20/7m29, sourcee5fcb82/freeze20260906-472a6d7-v1.12/15views erased;3NO_PLANE views explicitly not applicable, no enhancer failure. Both independent erasure/exterior-pixel checks PASS. No quality claim; accepted cabinet NO_PLANE remains complete-background blocker. New fill context20260906-f203ef7-v1/source2c18d10 pending exactE0/review; GPU1500-step fill only for27dd, CPU typed closure for40aec.

**E3:**49/50controls sealed;832352 still CPU-running,832535→832850 dependencies. Timing-only receipt `20260905-859f51d-v1/dispatch/full50_timing_refresh_20260906T022644Z.json`:399proposal records,217.7min elapsed,217.4min wall headroom;19.5–106min remaining scenarios are operational estimates, not scientific results. GT matching/independent full metrics wait for exact all50seal gate.

**E0 integration:** source472a6d7 global imports completed, full pytest received SIGKILL(-9) after80.78s; cause unconfirmed (not provenOOM). Two preceding failures were stale synthetic producer hashes; test-only source9be4ea8 now51focusedPASS and exactE0PASS digest4adef625418caea92965530e00c57fc71436e7c5bb29addf8f70acf5ba2025d6, frozen configs untouched. Bare harness_smoke requiresconfig/out (exit2); configured canonical smokePASS15rows. All logs preserved in `SimAny-wt/icra-main/outputs/integration-20260906T0211`. Complete clean integration preflight **833009 RUNNING**, sourcef203ef7, ordinary gcp26hb4CPU64G30min/noGPU; command `python outputs/global-integration/run.py` runs imports, fullpytest/JUnit, and configured canonical harness. Immutable worktree/output `SimAny-wt/integration-global-20260906/outputs/global-integration`; source/runtime/submission receipts included.

**E9:** paired confidence-interval vector figure formatter94a53d3 merged:97focused/130E0PASS; actual figure NOT_RUN until fullE3 paired-source authentication. Synthetic preview is engineering-only and never entered paper. Current paper remains8+8pages/23generated artifacts/199claimrows, with all10DROID CPU outcomes and catastrophic negatives preserved.

Claim gates: PASS source/stage integrity, raw-room50/400 coverage, DROID CPU publication and draft typography; FAIL legacyE1 provenance, fixedE4 applicability, final scientific submission; NOT_RUN fullE3 improvement, manipulation, Harmonizer, LOSO, strictfullcapture/physicalsuccess, final90/30/8demo. Next critical path: E3full integrity→independent metrics→generated table/figure; concurrent E2fill,E6grounding,E7GS/tail and full integration regression. All task hardware/runtime/config/checkpoint identities remain in the listed immutable stage contracts.

---

## 2026-09-06 02:01 UTC — complete DROID CPU evidence published; real background preparation passed

Main **a0a1aa3a1ac97040e5b2903e3130594a046fe8e8**, paper **b2c6cf431772240c71237682e0a7bb1b9d7d137d**, both pushed and remote-confirmed. Artifact checkout402cda0 unchanged. Live receipt `outputs/icra2027/orchestration-status-20260906T020154Z/{snapshot.json,squeue.txt}`. Final common scientific freeze remains unavailable.

**E7 CPU complete:** original sourcea150028/freeze20260906-f881b55-v1. Final unit832819 COMPLETED24m01, canonical same-host audit832884 COMPLETED17s/PASS, audit SHA `545bf2bcdce90ff0a19967a089b7c3c5f2b52234e8ad5acd18db2b5e4a1cd536`. All10records retained:5CPU COMPLETE/5construction FAILED/0NOT_RUN;5alignment-evaluated/3alignment gates PASS. IRIS832816 failed SfM125/237registered (52.7426%, frozen minimum80%). RPL held-out center RMS60910.346321m is catastrophic original SfM camera extrapolation; RAIL median rotation132.6576degrees remains a persistent orientation discrepancy. Original values retained, no clipping/gate adjustment. `alignment_outlier_diagnostic.json` confirms no meter/cm or similarity-transform mismatch; calibration versus SfM orientation cause is not isolated. CPU completion is not GS/full-simulator success.

**E7 GS active:** sourcef36f12ca0a2b9c71a04e766f706ea0fb2b4276b5/freeze20260906-dba8e6b-v1, E0 `ec7c0e1c039af602c6a93c64205284d2c670293446ba797f2e3e354ba2d1dc5b`.89focused/130exactE0 PASS. Ordinary CPU832907–832916 retain10slots:5TRAIN-fit exports/5explicit missing-fit NOT_RUN. Native16step GPU832917 COMPLETED12s/Hala, integrity PASS; actual trainer3.704s/peakCUDA247094784bytes (engineering measurements). Fixed30000step first-IPRL pilot **832922 RUNNING** HalaA6000/8CPU32G45min; four remaining valid TRAIN-fit slots including RAIL/RPL wait for pilot integrity, never held-out quality. Output `gaussian/droid_iprl_thu_aug_24_21_29_53_2023/train/` and `train_receipt.json`.

**E2 preparation complete:** sourcef205211e9dcd3462a05c4654a6a31a734ac4fa77/freeze20260906-dba8e6b-v2, E0 `9f0b2b024ae6b9886668f796fcbfeb6ae81df6f711414c7b48a69a3598fb4751`. Two original-context audits PASS; ordinaryCPU832895/832896 COMPLETED6m57/7m00. All17planned/5accepted retained. Scene27dd4da69e has1accepted with plane/3views;40aec5fffa has3accepted with planes/9views and1accepted cabinet explicitly NO_PLANE/3views. The latter remains a complete-background-fill blocker; no dropped-object or raw-background fallback. SAM3 runtime preflight PASS; masks await authenticated predecessor seals and new contexts/E0.

**E3:**49/50controls sealed;832352 still runs,832535→832850 remain dependency queued. Timing receipt `20260905-859f51d-v1/dispatch/full50_timing_refresh_20260906T015859Z.json`:344proposal records,190min elapsed/245.1min wall headroom. Unchanged timing scenarios31.7–145.8min remaining are operational estimates, not result forecasts. Full independent GT evaluation remains gated. **E6:** latest report12candidate successful RGB reconstructions (8fresh+4reused),4typed floor verification failures,2stillrunning; independent final18condition snapshot pending. Grounding code/smoke under development,0/72 final features,GTvault unopened.

**E9 publication complete for this working draft:** formattera8267e1 passed96focused/130exactE0 and actual all-ten publication validation. Exact publication source570d664ff39850a433729005d0ce720fb352f655/freeze20260906-f38923d-v1, E0 `9d33b952d31efa786b58ea1d49a0b2b7f3bbe16806e98622238cf9147d557973`. Canonical command `python -m robo.eval.paper_pipeline --config configs/experiments/icra2027/paper_droid_cpu_draft.yaml --out /group/worldcept/code/SimAny/outputs/icra2027/20260906-f38923d-v1/paper_tables --paper-root /group/worldcept/code/SimAnyRoom` exited0.7tables/23authenticated artifacts/199claim rows; previous5sourceJSON unchanged. NewDROID table shows every original workspace and evaluated/planned reference frames before error. Paper QA `audit/paper_qa_droid_cpu_revision.json`:8+8pages,16renders/allfonts embedded/noLaTeX issues,15exact claim sentences PASS. A local QA metadata script initially used a Python lackingPyYAML; corrected to the existing experiment venv, no experimental value changed. Paper andQA pushedb2c6cf4. Final submission staysFAIL; these CPU results do not completeE7 or enable physical success.

Claim gates: PASS contracts/raw-room coverage/CPU publication integrity/draft typography; FAIL legacyE1 provenance/fixedE4 applicability/final scientific submission; NOT_RUN fullE3improvement,manipulation,Harmonizer,LOSO,strictfullcapture/physicalsuccess,final90/30/8sdemo. Next: E3fullaudit/independentmetrics, E2masks/erase/fill with preservedNO_PLANE, E6publicgrounding, E7GS pilot/four remaining eligible stages.

---

## 2026-09-06 01:43 UTC — 49/50 controller scenes; public RGB full and DROID cohort active

Integration **357caca3569bbde7e11bd5395a786fee77f57402** is clean and remote-confirmed. Paper **d970348ab2726f3be8e704471beb2add98566ca3**, artifact checkout **402cda047b28ac17f50cb7406fe82fcdebacaaba** unchanged. Live scheduler/source receipt: `outputs/icra2027/orchestration-status-20260906T014346Z/{snapshot.json,squeue.txt,sacct.txt}`. No common final scientific freeze exists.

**E3:** source0a8b4ca, freeze20260905-859f51d-v1, **49/50 scenes sealed**. Only CPU control **832352** remains RUNNING; **832535** is the dependent integrity audit. Ordinary CPU **832850** waits afterok:832535 to prepare full-evaluation configuration only, using clean d9d7407 and reserved20260906-357caca-v1. Its negative smoke confirms zero GT reads/config creation without the exact50/1871/9355 PASS gate. Full matching/metrics still require reviewed committed configs and fresh E0. Driver/receipts: `SimAny-wt/e3-full-evaluation/outputs/full-evaluation-preparation/20260906-357caca-v1/`. No full independent result yet.

**E2:** strict TRAIN-only preparation, SAM3 masks, explicit LaMa erasure, and background completion are merged. Latest source7dfcdbf:130 integrated focused tests and130 exactE0 PASS; digest `abb0623c50614033f3837bfad968118eeedfade59af2c5290531caf8371cc365`. Erasure source000ccb:92 integrated tests/130E0 PASS, actual LaMa synthetic64x64 CPU invocation1.464s PASS with byte-exact exterior preservation. Existing producer mathematics and fixed recipes retained; missing plane/view/enhancer failures cannot silently produce a valid clean background. Fixed compact pair27dd4da69e/40aec5fffa (7+10 planned objects) real pipeline preparation is next; no background-quality measurements yet. Tests are engineering evidence, not paper metrics.

**E6:** recovery freeze20260906-2fa8928-v1, wrapperb2c827a and immutable measurement7c76b2c; E0 `6bbf8dbec2dde0b978469339eb53efec7e2df10321fe2568138be58d794dfded`. Four original outputs independently authenticated for reuse;14 ordinary GPU jobs832825–832838 preserve18conditions/72queries.832825–832828 COMPLETED0:0 (14m04,13m42,13m18,10m48), pending independent publication authentication.832829/830/831/834 failed the unchanged floor-normal guard (30.8/30.2/31.0/30.2 degrees from camera-up): **scientific construction verification failure**, not OOM. Remaining job states are in the exact scheduler receipt. Earlier832791–793 failed pre-model nominal-versus-usable-memory admission;832794–804 cancelled unstarted. That old freeze is preserved. Repaired admission is E0-bound to measured47,697,428,480-byte A6000 capacity and44GiB minimum for that exact model; native identity smoke832821 PASS. Grounding remains0/72; no GT labels/LOSO.

**E7:** CPU recovery freeze20260906-f881b55-v1, sourcea150028; E0 `8e33c960e923a6bf5699e7eab0c5a5d36a6e4ad5f5f8716dfb9e7cfc37f1cfe3`. Pilot832786 COMPLETED3m44, same-host independent QA832806 PASS. Preliminary pilot:16/47 held-out FK frames evaluated,31 missing retained; center RMS0.0181746807m, median rotation2.80312478degrees. CPU construction is not GS/full-build success. Remaining nine832811–832819: four cohort completions including pilot, five construction failures, secondIPRL832819 still running. AUTOLab/TRI/CLVR fail SfM; REAL fails frozen0.10m TRAIN RMS; IRIS failure retained pending exact final summary. Canonical same-host all-ten summary832884 waits afterany:832819, expected `full_cpu_summary/` and `full_cpu_completion_audit.json`. No task substitution or threshold changes. GS continuation c853f5a passed86focused/130E0 tests and genuine original-source TRAIN bridge832865; final runtime revalidation review precedes clean config/native smoke/pilot. Missing phone captures and strict full builds remain unmet.

**E9/D0:** published paper typography and20 generated source hashes remain PASS (conference8/root8pages,16 rendered/reviewed pages, embedded fonts, no LaTeX errors/undefined references/overfull boxes). No new quantitative table has been manually edited. Genuine selection/retry clips exist; final90/30/8second films await source-valid synchronized manipulation/appearance and final E9 evidence. TRELLIS.2 remains a separate mesh-only pilot,15planned/3geometry-matched/12null, preserving2/3collapse and1/15isolated stability.

Claim gates: PASS contracts, raw-room50/400 coverage, pilot/source integrity and draft typography; FAIL legacyE1 provenance, fixedE4 applicability, final scientific submission; NOT_RUN full agentic improvement, learned manipulation, Harmonizer preservation, task-local LOSO, complete prospective capture construction, physical success, final demo. Next: E3 control→integrity→full independent evaluation; E2 fixed-pair real background pilot; E6 full-output authentication/public grounding; E7 all-ten summary and TRAIN-only GS. All job submissions are individual, no arrays.

---

## 2026-09-06 00:58 UTC — RGB pilot complete; DROID resource failure isolated; bibliography and adapters pushed

Main **f95f60a254d3f3609a3b4a474a9fd2e75cfc7819**, paper **d970348ab2726f3be8e704471beb2add98566ca3**, artifact checkout **402cda047b28ac17f50cb7406fe82fcdebacaaba**. Main and paper pushes confirmed; immutable artifact checkout unchanged. Snapshot `outputs/icra2027/orchestration-status-20260906T005834Z/{snapshot.json,squeue.txt}`. No common final scientific freeze exists.

**E3:** freeze20260905-859f51d-v1/source0a8b4ca remains45/50 sealed, five ordinary CPU controls832300,832324,832352,832364,832382 RUNNING; audit832535 pending their completion. Full population50scenes/1871jobs/9355policy rows remains fixed. Timing-only receipt `dispatch/full50_timing_refresh_20260906T004831Z.json` shows sufficient wall headroom; no measuring job was restarted. Full independent GT evaluation remains NOT_RUN until all50 seals and832535 PASS. Future d9d7407 evaluator and4939a93 paired-CI paper formatter (99focused/130E0 tests PASS) are merged; no fabricated CI values.

**E6 public RGB:** smoke832627 COMPLETED0:0/14m38, independent QA PASS under20260906-4e440ad-v1. Fixed three-condition0020 pilot832703/832704/832705 COMPLETED0:0 on distinctHalaA6000 GPUs in10m15/9m43/9m13. Source7c76b2c unchanged; freeze20260906-0af302d-v1, E0 `01564760669d6c85d6d72a046ae5951320925b91071724fb14c3e91801e44301`. Independent QA PASS3/3 with48publicRGB each; exact outputs/seals in `independent_qa.json`. Construction walls607.61/579.54/549.42s are preliminary engineering timings, not independent image metrics. Still0/72feature rows and no GT labels/LOSO. Generic GPU request rejection ('Failed to parse GPU type:1') occurred before job acceptance; explicita6000 type resolved it without source/config changes. Missing-state task-graph extension remains under review; full18conditions not launched.

**E7:** TRAIN-only DROID implementation259304a (53focused/130E0 PASS; cross-environment synthetic22/22 reference frames) and exact ten-attempt config2958e02 (53focused/130E0 PASS) are merged and pushed. Config `configs/experiments/icra2027/droid_train_only_cpu/execution.json`, SHA `e282774167dc56afb4466fb9e92da00cece9f4e1ec24c577757bc4cc2d3c2b78`; actual E0 `b05acaf79d3771dee634bc48956f9dda586ea36e6fddb4e92058888ddb212344`; freeze20260906-490639d-v1. Ordinary8CPU32G/noGPU pilot **832746 OUT_OF_MEMORY0:125/41s**, scheduler reports1oom_kill.238RGB frames extracted; SfM stopped during SIFT96/238. Classification **resource_failure_memory**. `dispatch/submission.json` and `real_world/workspaces/droid_iprl_thu_aug_24_21_29_53_2023/` retain exact command/partial database/logs/result. Remaining9 blocked by pilot integrity gate. Explicit COLMAP thread allocation is being checked; no dataset/offset/registration/metric threshold changes. Full reconstruction/GS/physical success NOT_RUN.

**E2:** public TRAIN-only automatic removal preparation853a822 merged after62focused/materializer-schema and130exactE0 tests PASS. Exact E0 `c0604a5d8dc1c5b6ba31f58cbd6bf3d12350082be756dd8374d8febeb7373fba`. It reads immutable source factories and writes a distinct new freeze, never mutates old materialization or uses officialTEST frames. No real background-prep jobs yet. SAM3 mask and explicit LaMa erasure extensions underway; no silent Qwen fallback. Official LaMa releasev0.1.0 asset88575092 downloaded205803670bytes, SHA `7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c`, receipt `outputs/e2-runtime-lama-v1/download_receipt.json`. Initial Python CA environment failure resolved with system CA bundle; TLS verification retained. This is checkpoint availability, not an experiment result.

**E9 paper:** d970348 corrects two primary-source bibliography entries and authenticates all20 generated artifacts unchanged from20260906-a61a9ba-v1. `audit/paper_qa_bibliography_revision.json`: conference8/root8pages,16pages rendered/reviewed, allfonts embedded, no LaTeX errors/undefined references/overfull boxes. Main paired-uncertainty formatter4939a93 validates the future full audit and copies16 declared contrasts without recalculating statistics. No final full table or unsupported claim has been published.

Claim gates: PASS shared contracts, raw-room50/400coverage, publicRGBpilot/source integrity and draft typography; FAIL legacyE1 provenance, fixedE4 applicability, final scientific submission; NOT_RUN full agentic improvement, paired learned manipulation, Harmonizer preservation, task-local LOSO, strict capture alignment, physical success, final90/30/8s demo. E5 gated Cosmos checkpoint and E8 missing physical trials unchanged. Next: remainingE3controls→832535→full independent matching/metrics/uncertainty; E7 bounded resource correction/newfreeze/pilot; E6 full RGB and public grounding; E2 complete strict background construction.

---

## 2026-09-06 00:20 UTC — TRELLIS.2 geometry and generated draft published; public RGB smoke running

Integration **61c0289b6be34f800d5308289f2b99d33958f9e3** and paper **eb2f3824da079fea5162559b69c0ba523153cd73** are pushed; paper remote HEAD independently verified. Artifact checkout **402cda047b28ac17f50cb7406fe82fcdebacaaba** remains unchanged. Snapshot: `outputs/icra2027/orchestration-status-20260906T002013Z/{snapshot.json,squeue.txt}`. There is no common final scientific freeze.

**E3 full construction:** unchanged freeze **20260905-859f51d-v1**, source **0a8b4ca**, **43/50 scenes sealed**, all50 TRAIN observation scenes authenticated,1871plannedjobs/9355eventualA0–A4rows. Seven ordinary CPU controls **832300,832324,832340,832352,832364,832382,832396** RUNNING with no scheduler/phase failures. Audit **832535** waits for all50 seals; expected `dispatch/full50_control_integrity_gate.json`. Both832378/832392 completed within original90min despite earlier denied wall-extension requests. Future evaluation source **d9d740737c4c6c2a5737247a31173bc7d0282895** adds predeclared paired scene-cluster uncertainty to the existing aggregate:119focused tests/130exactE0 PASS; E0 **8c60c7e08ab0bf5a7d48f05352681f3e23ee840a9910b59dac7b7ff35311a01f**. Fixed four contrasts, four established metrics,2000resamples/seed0/95%CI; planned coverage and common accepted/matched geometry denominators separate. Fewer than2 supported scenes yields NOT_ESTIMABLE. No full GT config/read or results before complete construction and integrity audit.

**TRELLIS.2 engineering pilot COMPLETE:** new publication source **27c78ed26dc930fb36792245cecf9e7571ccf1c1**, freeze **20260905-a8ab431-v1**, actualE0 **a563beeaf7f4d371c0b09c34225ee7b03338c19ab54c15f71b107c5c04c4753a**.40focused/130E0 tests PASS. Ordinary CPU **832584 COMPLETED0:0/7s** after minimal existing-atomic-publisher repair; first832575 code_bug_cross_worktree_output_publication failure preserved. Frozen15planned/15generated/3geometry-evaluated/12unmatched:null; symmetricCD **19.166706553862742cm**, F1@20 **0.0854090232854222**, collapse **2/3**. Original isolated stability **1/15** retained separately. Geometry source `trellis2_geometry/geometry.json` SHA **91bc5a018b8440c1ce5f5dac0aee33b35d60b1bb28aebb96445ec2bba32717f8**; independent completion audit SHA **55332d82800a8ecfec2f193a33ab63e635858eea58e1f28333856ab49c64f645** PASS. Same sealed15meshes/registration/3reference matches; no generation or registration rerun. NativeGaussian/fulltwin/headline flags remain false. This is not an A0–A4 treatment or improvement claim.

**E9 current draft publication COMPLETE:** source **259f1addebb20a28017039eea9082cf20978c60d**, freeze **20260906-a61a9ba-v1**, E0 **2718e2f9c2b4c4189f344c6fd56da878fcf7be0a04516d8555ad3da5ce1fb5e3**.88focused/schema and130exactE0 tests PASS. Actual `python -m robo.eval.paper_pipeline --config configs/experiments/icra2027/paper_current_engineering_draft.yaml --out /group/worldcept/code/SimAny/outputs/icra2027/20260906-a61a9ba-v1/paper_tables --paper-root /group/worldcept/code/SimAnyRoom` exited0 from clean259f1ad with explicit primary evidence root. Six generated tables/20 transferred artifacts/158claim rows independently authenticated; original three scientific sources unchanged. Independent QA SHA **44808c6e91f0e0d5c9b048e539d839a6e54e452be53e0401ec6eea1275e9036c**. E4 typed40planned/0policyexecuted closure is displayed with null success, not zero success. Paper source **eb2f382** clarifies controller acceptance versus complete twin, isolated plane-probe scope, room-level dependency contract, and TRELLIS.2 mesh-only scope; all14 claim-decision sentences verified. `bash build.sh` PASS: conference8pages/root8pages, all16pages rendered200dpi,15/15fonts embedded perPDF,no LaTeX errors/undefined references/overfull/underfull boxes. QA `SimAnyRoom/audit/paper_qa_engineering_revision.json`; old QA archived. Formatting PASS does not change final submission FAIL.

**E6 public RGB smoke:** source **7c76b2c39ceb515237d8eac490fbf76813092569** merged/pushed61c0289;50focused/130exactE0 tests PASS. Freeze **20260906-4e440ad-v1**, actualE0 **94b9ae0f9017f8861da778492b1994134aa7d5dae87eb200dfe44ed2f1766556**; config/roster/runtime and actual checkpoint content hashes bound. Ordinary **832627 RUNNING** onHala,1idleA6000/4CPU64G/30min,normalQoS,--export=ALL,--no-requeue, noarray. Exact command `launchers/submission_receipt.json`; output `audit/public_reconstruction/behavior_task0023_clean/`. This fixed smallest-clean smoke reads6RGB only; Omega/DA3/scene conversion complete,15000-iteration Gaussian stage active. No terminalPASS yet. Full roster602publicRGB/267preselectedframes/18conditions/72unchangedquery rows;9missing source anchors remain missing. Legacy GT-depth/pose/splat artifacts are excluded. Independent GPU smoke must PASS before fixed0020clean/mild/severe pilot; query grounding, labels and LOSO remain NOT_RUN.

**E7:** all original10DROIDepisodes/9labs incl.CLVR have local public video/H5/metadata;30files **60,862,546bytes** content-hashed in `outputs/icra2027/audit-e7-train-only-20260906T001458Z/input_inventory.json`. Agent `e2_camera_repair` implements narrow TRAIN-only fit/offset/checks and sealed-fit held-out evaluation in `agent/icra-e7-train-only`; nojobs yet. Metadata-only80/20 split with4-index guard prevents FK overlap across offset hypotheses. CPU SfM alone will not count as reconstruction_success/fullbuild. Additional genuine phone captures remain missing. E5 Cosmos403 and E8 absent matched physical hardware/trials unchanged.

Claim gates: PASS infrastructure, current raw-room coverage, pilot/source integrity and paper typography; FAIL E1legacy provenance, E4fixed applicability, final scientific submission; NOT_RUN full agentic improvement, paired learned manipulation, Harmonizer preservation, task-local LOSO, strict real capture alignment, physical success, final90/30/8s demo. Next critical path: remaining7controllers →832535 →fresh independent full evaluation/matching/per-scene metric shards/complete aggregate audit →generated paper update. Parallel: E6real smoke→pilot and E7TRAIN-only alignment smoke. Preserve all failed/unsupported units.

---

## 2026-09-05 16:40 UTC — raw Gaussian full row complete; all proposal jobs submitted

E2 raw Gaussian recovery **COMPLETE** at frozen source `cbba8ff63bfec9002a740826b55259106373a7af`, freeze `20260905-3ff95f4-v2`. Fixed smoke830574 PASS16TESTviews/48s, followed by local CPU preparation of50scenes in402.10s and verified byte reuse of26original bundles. The24 missing scenes were rendered by independent jobs830704,830705,830706,830707,830708,830709,830710,830711,830712,830713,830714,830715,830716,830717,830718,830720,830721,830723,830724,830725,830727,830732,830733,830742, allCOMPLETED0:0 (8–16s each). Canonical GPU aggregate830743 COMPLETED0:0/153s. All work ran onHala1A6000 perGPUjob; reuse/plan stages used zeroGPU.

Preliminary paper result on the fixed full population: **50/50scenes,400/400officialTESTviews**, PSNR=21.601336199935982, SSIM=0.8434523467239964, LPIPS=0.2988105320557952. Each metric has400samples; no missing raw-row scenes. These are the input Gaussian reconstruction-ceiling row only; the other seven TableII method rows remainNOT_RUN, paper_ready=false. Independent source/config/contract/table/coverage/scheduler audit PASS: `/group/worldcept/code/SimAny/outputs/icra2027/20260905-3ff95f4-v2/full_raw_completion_audit.json`. It hashes final table/coverage and authenticates producer receipts, without redundantly rehashing all large inputs. Authoritative JSON: `/group/worldcept/code/SimAny-wt/e2-camera-bank/outputs/icra2027/20260905-3ff95f4-v2/raw_room/aggregate/table/fidelity_table.json` (SHA256 `12c2ede86faa7bb4d44c0be81eb275e722a94867829216517a7cd4e1e0a48871`). No values manually transferred to the paper.

Reproducible exact-source commands and jobmanifest: `20260905-3ff95f4-v2/dispatch/{prepare_full.py,run_scene.sh,submit_fresh.py,full_dispatch_manifest.json}`. GPU CLI is `python -m run.icra2027.e2_raw_room --config configs/experiments/icra2027/e2_raw_recovery_full.yaml --freeze-root /group/worldcept/code/SimAny/outputs/icra2027/20260905-3ff95f4-v2 --phase run --scene-id <scene>`, aggregate uses `--phase aggregate`; all use frozen e2-camera-bank checkout. A rerun must receive a newfreeze.

TRELLIS all49 remaining GPU+audit pairs are accepted (`20260905-02b54da-v2/dispatch/full_gpu_submissions.json`), in addition to the completed15-object pilot. RVG all49 remainingGPU+49CPUaudit jobs are accepted (`20260905-d1e8e21-v1/dispatch/remaining_gpu_manifest.json` and `remaining_gpu_audit_manifest.json`), in addition to its14/15pilot. No submission process remains needed for either generator. One orchestration retry initially lackedPYTHONPATH, failed before any jobsubmission, and was stopped; corrected environment receipts/logs preserve that environment failure and successful submissions. All scientific source/configs remainunchanged.

Canonical pilot recovered unchanged source0f6b708 afterGCP cancellation: Halaobserve830593 COMPLETED0:0/9s (2.3675s producer), CPUcontrol830596 RUNNING onsof1-h200-4. Routing of this still-unstarted CPUjob was broadened from pinnedsof1-h200-5 to zone-sof1, with before/after receipt. GCP TRELLIS830153 independently COMPLETED0:0/19m16s, preserving actualA100runtime provenance; no blanket inference thatGCP isunavailable.

Per-job terminal audits now cover13affectedTRELLISscenes in `terminal_audit/20260905T161447Z` and `20260905T162025Z`: 15Gaussian proposals contain positive-infinite opacity logits; noNaNs were observed in those fields. They remain artifact-validation failures, not inferred geometrycollapses. Original three generator-process failures and unexecuted suffixjobs remain in the denominator. The tested terminalpooladapter is being independently replayed beforemerge; runningpilot/fullgenerator sources are untouched.

Next critical path: finish15job/75policyrow canonicalpilot; integrate tested failurepreserving inventories; prepare evaluation-only matching with existingGTmatcher. All150requiredGTfiles across50scenes exist (stat-only audit); missingautomatic-reference adapter, not missingprivate data, currently blocks geometryevaluation. GTsurfaces must never entercontrollerinputs; use frozen discovery-only matching support and retainunmatchedjobs. Fullagentic/manipulation/LOSO/Harmonizer/real-world and finalpaper/demo claims remainunpassed. Code main3d45ac9 and paper121b5b1 were pushed before this milestone record.

---

# Current wave — 2026-09-05 16:20 UTC

## 2026-09-05 16:20 UTC — shared proposal fleet and fixed pilot dispatched

Owner: root with E2 camera repair, E3 canonical and RVG agents. Published main source `4a3b501ff995ab11ec7fc88149e4e29e412e1b28`; primary artifact checkout remains clean `402cda047b28ac17f50cb7406fe82fcdebacaaba`, paper remains `121b5b196c5700cadaae9f2104d41b1c903aa791`. This wave uses separate immutable prerequisite freezes; there is **no common final scientific freeze** yet. All new jobs are ordinary independent jobs, never arrays.

| Stage | Immutable freeze | Frozen execution commit | Gates / commands / outputs |
| --- | --- | --- | --- |
| TRELLIS initial pool | 20260905-02b54da-v2 | 0dbdee3391a799bf8b56b1d3d71f8a11db39f146 | E0 128 PASS; fixed 38d58a7a31 pilot 829843 + audit829977 PASS, 15/15 available; all50 CPU plans complete; full individual submission/resume commands in dispatch/submit_full_resume.py; trellis_initial/<scene> |
| RVG initial pool | 20260905-d1e8e21-v1 | 3ff95f4ec7dee0a46d8259c8fabbd16b1704bfcf | E0 158 PASS on rerun, original mtime fixture failure retained; pilot829995 + audit830112 PASS,14/15 available,1 insufficient-view job; all50 CPU plans complete; dispatch/submit_remaining_gpu_priority.py --gpu-only then submit_remaining_gpu.py; rvg_initial/<scene> |
| Canonical A0-A4 pilot | 20260905-33bd974-v1 | 0f6b7080de23e926bcab2f3072c84381573370df | E0 128 PASS;49 focused PASS; sealed inventory15jobs/75 planned policyrows; observe830352 CANCELLED by0 after123s before phase entry; control830372 dependencycancelled; Hala missing-phase recovery underway; agentic/input_inventory |
| E2 frozen-camera recovery smoke/full | 20260905-3ff95f4-v1 / v2 | cbba8ff63bfec9002a740826b55259106373a7af | 104 focused PASS;128 exact E0 PASS; both contracts sealed; fixed2-scene16-view smoke CPU plans complete locally; GPU submission rejected by scheduler, no accepted ID at snapshot; raw_room |

Output prefix for every relative freeze path above: `/group/worldcept/code/SimAny/outputs/icra2027/`. Execution checkouts are clean and remain unchanged. Main includes separately tested failure-isolation patch551f536; it is **not** silently applied to these running freezes. New implementation passes44 smoke and58 compatibility tests (8 overlap); generator calls, seeds, preprocessing and thresholds are unchanged. Future recovery requires a new source/config/freeze and explicit original-proposal attribution.

Live snapshot16:20:49: TRELLIS830163/830196 and RVG830201/830217 running on Hala; TRELLIS830153 running on GCP A100. Source/config unchanged by routing. TRELLIS has31 terminal scene pools; producer records say794 available,30 unavailable,119 generation_failed across943 planned jobs in those terminal scenes, **not** independent accepted-asset yield. Only18 independent all-artifact completion audits have passed at this snapshot. Full planned denominator remains50scenes/1871jobs. RVG input preparation is1793 multiview-ready/1871planned,1801 prepared,70 preparation failures and8 insufficient-view jobs,19896 selected views; these are input readiness, not generated assets.

Failures preserved: GPU830022 empty sparse coordinates (tool generation failure);830078/830082 rembg.new_session unavailable (code/environment backend path), causing later objects to lack completion records. Do not claim all uncompleted objects were attempted. Ten CPU artifact audits failed:830029,830031,830114,830116,830150,830152,830156,830162,830187,830193; first four traced to nonfinite Gaussian opacity, remaining six are under per-object audit. No geometric-collapse inference or relaxed finite gate. Seven-scene terminal audit at `20260905-02b54da-v2/terminal_audit/20260905T161447Z/` authenticates source/input/pool/record hashes, preserves every job, and identifies7 invalid Gaussian proposals plus original process failures. Its reproducible read-only command is `PYTHONPATH="$PWD" /group/worldcept/code/SimAny/.venv/bin/python /group/worldcept/code/SimAny/outputs/icra2027/20260905-02b54da-v2/dispatch/audit_terminal_pools.py <scene_ids>` from clean e3-generation-cohort; each invocation creates a new audit directory.

Resource failure is distinct: user A6000 association cap4, and global Slurm MaxJobCount10000/MinJobAge3600 causes sbatch EAGAIN before acceptance. Accepted IDs have per-job receipts; rejected stdout/stderr are preserved and only missing submissions are retried. No administrator setting or other user's jobs changed. Scheduler and counts: `20260905-02b54da-v2/dispatch/cycle_status_20260905T162049Z.json`; RVG dispatch/HANDOFF.md and submission receipts; canonical gcp_cancelled_observation_audit.json; E2 dispatch/run_submission_error*.json.

Compact next dependency wave: keep shared initial proposals for A1-A4; complete the fixed15-job canonical pilot before scene-sharded control; reuse26 E2 bundles and render only24 missing scenes after exact-camera smoke passes. No dataset/policy/threshold changes based on outcomes. Per-scene failed/invalid proposal normalization must preserve original pools and all jobs. Independently frozen automatic-job GT matching remains missing, so evaluation/agentic improvement/geometry headline claims remain NOT_RUN. E4 manipulation, E5 Harmonizer, E6 LOSO, E7 strict real builds, E8 physical pairs and final E9/D0 gates remain unpassed; preliminary paper values are unchanged.

Config/checkpoint hashes and exact commands are in each freeze's contract/freeze_manifest.json, input manifests, shared contract/*_model_resources.json and job receipts. TRELLIS contract2d11affbafb4b7308eda54de5a72ccd34339931810376481c69b8e229c729510; RVG cf479faec22013e0eadbbaa2113711930acdd4f273e3897dd48ff6488d90a709; canonical3d57814bbf0052a8cd5a973c4f2640b988797b0d3d62bf1f0784857e9ced4c84. Hardware/runtime: TRELLIS pilot1A6000/175s scheduler; RVG1A6000/391s scheduler; CPU audits use zero GPUs. Infrastructure gate PASS; no new scientific superiority gate PASS.

---

Historical status below is superseded where it conflicts with the dated current wave.

# E2 current status — refreshed 2026-09-04

Owner: root/E9 audit. State: BLOCKED (room computation complete; full Table II incomplete).
`READY_FOR_PAPER=false`. `ROOM_METRICS_COMPLETE=true`. Claim gate: NOT_RUN.

The earlier submission status below is historical and superseded by verified on-disk outputs and scheduler records.

- Rendering source: `2c7ed67df534`; validation/metric source: `2f3ca585f30a` (`agent/e2-validation-fix`). The second source is recorded separately, not backfilled into render provenance.
- Revalidation inventory job `821620`: COMPLETED/0:0, 2:53 on sof1-h200-2. Metrics `821650`: COMPLETED/0:0, 31:38 on gcp-eu1-a100-80g-qrfh. Earlier attempts `821621` and `821642` were CANCELLED by scheduler UID 0; not scientific outcomes.
- Output: `outputs/icra2027/icra2027-contract-v1-e2-2c7ed67df534-full/fidelity/table-reval-2f3ca585f30a-20260904T121220Z/fidelity_table.json`.
- Output SHA256: `b2e281a65ca82b7e9504fb6dd3d96099fafa8d100a843a1c2da0b0ff22653f40`.
- The stored validator reports valid=true; each of three room methods has 50 scenes and 400 paired views. No object or Harmonizer fidelity row has measured values.
- Read-only formatter import, source hashes and generated table are owned by E9. They preserve `paper_ready=false`; hashing archived metrics does not repair missing generator provenance.
- Smoke/pilot commands and pinned checkpoint identities: historical protocol below. Revalidation commands and exact jobs: `outputs/icra2027/e2-reval-2f3ca585f30a-20260904T121220Z-noarray-submission/{jobs.tsv,recovery_jobs.tsv}`.
- Failed/not satisfied: independent object masks/evaluation surfaces, Harmonizer aligned output, unified paper freeze and all Table II acceptance criteria. No new E2 GPU job is running.

## Historical status (retained)

# E2 fidelity status

Updated: 2026-09-03 12:42 UTC

`IMPLEMENTATION_STATE=READY_TO_SUBMIT`

`A100_PROFILE_STATE=HARDWARE_RUNTIME_READY`

`FULL_SUBMITTED=false`

`READY_TO_SUBMIT=true`

`READY_FOR_PAPER=false`

## Current result

The canonical E2 workflow is now specified as a real 50-scene render, not a
smoke or a legacy-summary aggregation:

| phase | planned workload | accelerator | state | Slurm job |
|---|---:|---|---|---|
| fresh E0 contract | one preflight at final E2 commit | CPU | pending final commit | none |
| pre-freeze developer pilot | one scene, 8 views, 3 methods | 1x A6000 | diagnostic PASS; not accepted evidence | `815356` |
| smoke render | 2 individual jobs for the exact pinned scenes, 8 views, 3 methods | 1x A100-80G/job on `gcp-eu1-a100-80g-qrfh` | each depends on fresh E0; not submitted | none |
| smoke inventory | validate 2 manifests and 64 total PNGs: 48 method renders + 16 GT | CPU | depends on both smoke render IDs; not submitted | none |
| smoke metrics | recompute 16-view PSNR/SSIM/LPIPS smoke table | 1x A100-80G on `gcp-eu1-a100-80g-qrfh` | depends on smoke inventory; not submitted | none |
| pre-fix replacement diagnostics | 5 individual bundles / 8 objects | 1x A100-80G/job | hardware/runtime PASS; 2 bundles passed and 3 failed strict scientific alignment; superseded by method revision | `816198`, `816241`-`816244` |
| formal post-fix replacements | 5 individual bundle jobs / 8 objects, official-train generation only | 1x A100-80G/job on `gcp-eu1-a100-80g-qrfh` | each depends on smoke metrics; pending experiment gate | none |
| full `render-room` | 50 individual scene jobs, 8 official-test views, 3 methods | 1x A100-80G/job on `gcp-eu1-a100-80g-qrfh` | each depends on all 5 replacements; not submitted | none |
| full `inventory` | validate 50 manifests and 1,600 total PNGs: 1,200 method renders + 400 GT | CPU | depends on all 50 full render IDs; not submitted | none |
| full `metrics` | recompute 400-view PSNR/SSIM/LPIPS and publish Table II artifacts | 1x A100-80G on `gcp-eu1-a100-80g-qrfh` | depends on full inventory; not submitted | none |

Pilot job `815356` (`icra-e2-leakage-pilot`) completed on `hala` with exit
`0:0` in 26 seconds and wrote
`outputs/icra2027/e2-dev-pilot-leakage-815356/fidelity/room_runs/5748ce6f01`.
It contains one manifest plus 32 PNGs. It ran before the final clean commit,
fresh E0 contract, five individual replacement jobs, and exact two-scene smoke;
therefore it is development evidence only and is not accepted E2 evidence.
Earlier job `815268` was also a one-scene exporter pilot, and historical job
`814563` was a renderer/LPIPS compatibility smoke. Neither is counted.

The required 62-job DAG is now fixed as fresh E0 -> 2 individual smoke renders
-> smoke inventory (after both) -> smoke metrics -> 5 individual replacement
jobs -> 50 individual full renders (each after all 5 replacements) -> full CPU
inventory (after all 50) -> full one-A100 metrics. Smoke and full use distinct
freeze IDs. The end-to-end smoke therefore has its own canonical
inventory/table, while full renders a separate atomic 50-scene population only
after all five replacement jobs validate.

A100 compatibility job `816031` completed on
`gcp-eu1-a100-80g-qrfh` with exit `0:0` in `5:33`. Against the Hala A6000
compatibility reference `814563`, it produced an identical rendered JPEG
SHA-256 and exactly equal PSNR/SSIM; the maximum LPIPS difference was
approximately `7.4e-5`. This validates the explicit
`E2_GPU_PROFILE=gcp-a100` profile for `render-room` and `metrics` on
`batch`/`normal` with `a100-80g`.

Replacement dependency/profile probe `816065` reached A100 PyTorch, ninja,
and spconv before the dirty-tree gate correctly stopped it. The later clean,
pre-fix jobs exercised the complete runtime and scientific validator on all
five bundles. Jobs `816243` (`286b55a2bf-auto`, three tier-A targets) and
`816244` (`286b55a2bf-factory`, one tier-A target) completed with exit `0:0`
in `4:50` and `3:58`, respectively, and passed publish plus strict reuse
validation. Jobs `816198`, `816241`, and `816242` exited `2:0` only after
generation reached the unchanged final alignment gate:

- `816198` (`7:53`), `3f15a9266d-factory`, object `232`: tier C,
  `F1@20=0.004`, `F1@40=0.012`, size ratio `24.39`;
- `816241` (`4:28`), `0d2ee665be-auto`: object `1007` passed tier A, but
  object `1028` was tier C with `F1@20=0.108`, `F1@40=0.184`, size ratio
  `3.21`;
- `816242` (`3:55`), `0d2ee665be-factory`, object `16`: tier C,
  `F1@20=0.030`, `F1@40=0.062`, size ratio `9.48`.

The three rejected jobs published no bundle; only the two strictly validated
diagnostic bundles exist beneath their pre-fix freeze ID.

These are pre-fix diagnostic outcomes, not an A100 incompatibility and not
formal replacement evidence. They exposed a source-up initialization omission:
the old path assumed source `+z`, while image-to-3D assets may use any signed
canonical axis. The commit-bound revision searches fixed, ordered proper bases
for `+z`, `-z`, `+x`, `-x`, `+y`, and `-y`, using the existing `sym_score`,
yaw/scale search, ICP, A/B thresholds, and `[0.4, 2.5]` size gate unchanged.
The replacement factory-alignment path records the selected
`source_up_hypothesis` and deterministic GT-surface and mesh-sample seeds `42`;
unrelated historical hybrid/SHARP paths retain their prior `+z`-only behavior.
The formal post-fix validation is the five-job replacement stage already
dependency-gated inside the canonical DAG; all five must pass before full
rendering can start. E0/inventory remain CPU-only on `sof1-h200-2`.

The superseded ordinary-job submission `815912`-`815972` was cancelled while
pending before migration and is not accepted evidence. The replacement DAG
must use a fresh clean commit and fresh E0 contract.

The implementation is ready to enter the canonical submission after the tested
method revision and these commands are captured in one clean commit. A fresh E0
contract must freeze that exact commit and the schema-v2 fidelity config; the
older job `814951` contract contains the old placeholder config and is rejected
by the new launcher. Full submission remains false, and the five post-fix
replacement bundles remain an explicit pending experiment gate rather than a
pre-submission pilot.

Repository-local source preflight at this revision found all 100 required
`<scene>_{factory,auto}` object inventories. Across them, all 457 accepted
factory objects and all 671 accepted automatic objects have readable
`aligned.json` and non-empty `trellis_gs.ply` inputs (zero missing). All 100
legacy render summaries declare `official-test` and eight unique frames; those
summaries are provenance cross-checks only, not canonical metric inputs.

Focused launcher/config/exporter/inventory/evaluator checks pass, including
three regressions for signed source-up bases, non-z source alignment, and
legacy z-up API preservation. The maintained suite passes with 445 tests and
one skipped; its two warnings are the existing trimesh zero-volume warnings.

## Pinned contract

- Dataset/population: ScanNet++ v2 `nvs_sem_val`, exact 50 IDs.
- Split: official DSLR test only; train/test overlap is fatal.
- Per scene: 8 unique registered held-out frames.
- Fresh output: 8 GT PNGs and 8 PNGs for each of
  `input_scene_gaussian`, `factorized_gt_discovery`, and
  `factorized_auto_discovery` (32 PNGs total per scene).
- Full coverage: 50 scenes and 400 views for each of the three available room
  methods.
- Harmonizer and object rows remain explicit but unavailable; no anchor value
  is copied into the table.
- Output, temp, and cache paths are repository-local and no prior output is
  overwritten.
- Replacement alignment evaluates the fixed signed source-up order
  `+z,-z,+x,-x,+y,-y` with deterministic GT-surface and mesh-sample seeds `42`.
  The replacement factory path records the selected hypothesis. `sym_score`,
  yaw/scale search, ICP, tier-A/tier-B thresholds, and the size-ratio gate are
  unchanged.
- LPIPS trunk: repository-local
  `.cache/icra2027/e2-render-compat/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`,
  exactly 244,408,911 bytes and SHA-256
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
  Render and metrics phases fail before execution on any drift; metrics records
  this identity in `run_manifest.json`. No checkpoint copy/download is used.
- TRELLIS executable source: repository-local
  `.cache/icra2027/e2-replacements/trellis-source-442aa1e1afb9014e80681d3bf604e8d728a86ee7`,
  223 files, upstream commit
  `442aa1e1afb9014e80681d3bf604e8d728a86ee7`, FlexiCubes commit
  `815e075a2a400d06c48d94c347674344ed6ae5c5`, and tree SHA-256
  `df4059255d1a72a6d82a76f2b9c97f034e1c1fd9c06ddec782bd8282be3f8401`.
- TRELLIS snapshot: repository-local
  `.cache/icra2027/e2-replacements/trellis-image-large-25e0d31ffbebe4b5a97464dd851910efc3002d96`,
  13 files, tree SHA-256
  `8daaba378d461e5ac0a7771c92d8ee70abfe7a95b0bb54149b44d3f12ea79d41`.
- DINOv2 source: repository-local
  `.cache/icra2027/e2-replacements/torch/hub/facebookresearch_dinov2_main`,
  203 files, tree SHA-256
  `bad25746e5bccefe9a039b7ebe5537c72765edc9be6654881699298f37f0cc4d`.
- DINOv2 weight: repository-local
  `.cache/icra2027/e2-replacements/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`,
  1,217,607,321 bytes, SHA-256
  `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.
- SAM3 executable source: repository-local
  `.cache/icra2027/e2-replacements/sam3-source-8e451d5eb43c817b64ae7577fb7b9ae223db88a9`,
  523 files, upstream commit
  `8e451d5eb43c817b64ae7577fb7b9ae223db88a9`, package version `0.1.0`, and
  tree SHA-256
  `73c418359155da5da839853613260e84f48e5bd8e3d494c6614ca9e561a187ae`.
- SAM3 weight: repository-local
  `.cache/icra2027/e2-replacements/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`,
  3,450,062,241 bytes, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  Replacement generation validates all six pinned model inputs and runs
  offline without a model copy or download.

## Live Slurm snapshot

Snapshot time: `gcp-eu1-a100-80g-qrfh` refreshed at 2026-09-03 12:42 UTC;
other rows retain the 10:28 UTC snapshot. This is scheduling guidance, not a
reservation; re-query immediately before submission.

| pool | configured GPU | allocated | currently free | scheduler state | E2 use |
|---|---:|---:|---:|---|---|
| `hala` | 8x A6000 | 8 | 0 | `ALLOCATED` | not selected after the A100 migration |
| `sof1-h200-[0-7]` | 64x H200 | 64 | 0 | mixed/planned/reserved | unavailable now |
| `gcp-eu1-a100-80g-qrfh` | 8x A100-80G | 3 | 5 | mixed/planned | selected for replacements, `render-room`, and metrics; `batch`/`normal`, profile `gcp-a100` |
| `gcp-eu1-rtx6000-vz3w` | 8x RTX6000 | 8 | 0 | mixed/planned | unavailable now |
| `gcp-eu1-rtx6000-txtb` | 8x RTX6000 | 0 | 8 nominal | drained, maintenance, reserved | not schedulable |

The live user association reports QOS
`background,debug,invariant-bench,normal` and a group limit of
`gres/gpu:a6000=4`. The canonical E2 GPU jobs now request A100-80G, so that
A6000 limit does not throttle this DAG. Replacement, render, and metrics
concurrency is left to Slurm on the selected GCP node. All 62 jobs use
`--account=runyi_yang` with automatic requeue disabled; a failed, preempted, or
node-failed attempt is audited and explicitly resumed rather than silently
re-entering the dependency chain.

## Submission and experiment gates

Implementation readiness is true. Checked items are pre-submission gates;
unchecked items below are dependency-gated experiment completion checks and do
not prevent enqueuing the canonical 62 ordinary-job DAG:

- [x] maintained test suite passes before the final clean commit;
- [x] exporter test proves eight distinct PNGs per directory and a 32-entry
      hash manifest;
- [x] launcher shell syntax and strict negative gates pass;
- [x] `sinfo`, `squeue`, association, and requested QOS are refreshed;
- [x] A100 renderer/metrics parity and replacement hardware/runtime probes pass;
- [ ] E0 reaches terminal `COMPLETED/0:0`; exactly one `run_root` is parsed and
      its clean full-SHA contract path and canonical digest are validated and
      recorded;
- [ ] exact two-scene smoke render, inventory, and metrics all succeed;
- [ ] all five formal post-fix replacement bundles pass strict alignment and
      artifact validation;
- [ ] all 62 E0/smoke/replacement/full job IDs are recorded.

## Final handoff fields

Fill after the dependency chain reaches terminal states:

```text
code_commit=
smoke_freeze_id=
full_freeze_id=
e0_job_id=
e0_contract_path=
e0_contract_sha256=
smoke_render_job_ids=                 # 2 IDs
smoke_render_jobs_completed=/2
smoke_inventory_job_id=
smoke_metrics_job_id=
replacement_job_ids=                  # 5 IDs
replacement_jobs_completed=/5
full_render_job_ids=                  # 50 IDs
full_render_jobs_completed=/50
inventory_job_id=
metrics_job_id=
smoke_output_root=
full_output_root=
room_scene_manifests=/50
pngs_per_scene=/32
views_input_scene_gaussian=/400
views_factorized_gt_discovery=/400
views_factorized_auto_discovery=/400
inventory_valid=
artifact_hashes_verified=
paper_ready=false
claim_status=
```

Even when the full three-phase chain succeeds, `paper_ready` remains false
until E5 supplies the Harmonizer condition and leak-free held-out masks plus
registration-independent surfaces exist for the object rows. Full room
appearance evidence and full Table II claim readiness are separate gates.

## Fresh automatic pilot input audit — 2026-09-05

CPU readiness source `f1aca5eae5244aeaa3580dfe3c73d50bf37fa5fd`, clean at
execution, binds the v30 fresh TRAIN Gaussian, v34 complete discovery, fresh
TRELLIS `20260904-4d0787c-v2` and RVG `20260905-4d0787c-v1` pools. All five
discovered jobs remain present: two prepared and three preparation failures.
This is an input audit, not a new experiment freeze or a Table II result.

Actual command, run from `/group/worldcept/code/SimAny-wt/e2-fresh-readiness`:

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e2_fresh_readiness \
  --config configs/experiments/icra2027/e2_fresh_readiness.json \
  --out /group/worldcept/code/SimAny/outputs/icra2027/audits/e2-fresh-readiness-f1aca5e/readiness.json
```

Exit 0. Output SHA-256:
`cc5d536a668bd269c6a2d543ac2321e351c25ef13f5fd626d0a274e61e5ff90d`.
All 27 official TEST frames have registered cameras and available RGB files.
The unchanged canonical eight-view selector returned:
`DSC06797.JPG`, `DSC06800.JPG`, `DSC06804.JPG`, `DSC06808.JPG`,
`DSC06811.JPG`, `DSC06815.JPG`, `DSC06819.JPG`, `DSC06823.JPG`.
All eight image dimensions match source calibration. Frame overlap is zero
with all 48 optimization frames and the 23-frame generation-input union;
the selector excludes all 2,364 official TRAIN frames. No selected evaluation
RGB hash equals a selected training RGB hash. Public calibration can reflect
dataset-wide SfM; the frozen image optimization boundary is stated separately.

Validation command:

```bash
mkdir -p .tmp outputs
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e2_fresh_readiness.py tests/test_fidelity_room_export.py \
  --basetemp .tmp/e2-readiness-tests-final-20260905
```

**30 tests passed in 1.34 s**. Initial invocations exposed test-environment
setup only: the existing room-export tests reject default `/tmp` paths, and
the new checkout initially lacked the local `.tmp` parent. The passing run
uses that established repository-local path contract; no evaluator was changed.

Room RGB/pose inputs are available, but a fresh raw-Gaussian render bundle and
source-bound E0/config/smoke remain pending. Existing room export requires
factory and automatic composite builds and is not directly applicable to raw
initial pools. All object rows still lack held-out photo masks, authenticated
registered predictions and independent metric evaluation surfaces. The
TRAIN-derived mesh and its legacy-named `gt_points.ply` samples are explicitly
excluded as independent evaluation surfaces, even if resampled or rehashed.
Object visibility in TEST photos remains unverified.

**PSNR, SSIM, LPIPS, object appearance, CD and F1@20 mm: NOT_RUN.**
No GPU, scans, GT-controlled construction, oracle selection or new metric
records were used. One scene of input readiness does not fulfill the fixed
50-scene population; `paper_ready=false`.

## Fresh raw-Gaussian eight-view smoke — 2026-09-05

The raw-only adapter completed at clean source
`0cde077956e7f8a7f7c15c0058ceb55e3c765561`, freeze
`20260905-edd5482-v4`. It uses the exact eight TEST cameras listed above,
the v30 fresh Gaussian, the unchanged `common.load_gaussians/render_view`,
and canonical `robo.eval.fidelity_metrics`. No object or GT-controlled
composite is loaded. The input Gaussian is the only populated method row.

Validation before submission: 52 focused CPU tests passed in 0.90 s;
independent review passed 17 adapter tests in 0.50 s. Actual CPU imports
validated the mini-viewer prebuilt gsplat renderer and main Python LPIPS
with frozen local weights and user-site disabled. Renderer schema imports
use the existing v30 scoped Python 3.10 pydantic cache. No packages or
models were installed/downloaded. Exact-source E0 passed 128 tests in
22.00 s at:

```text
/group/worldcept/code/SimAny-wt/e2-raw-room/outputs/icra2027/preflight-smoke-20260905T004120Z-3798192/preflight.log
```

The dedicated canonical E0 digest is
`63a0b76812b330a7974d5f20ddf10f4ff4ce0c0fd13e31d9abfd8a9a68b56339`.
CPU `--phase plan` then revalidated the full readiness chain and fixed
camera/image/runtime inputs. Job **825227** ran the source-bound
`run/icra2027/e2_raw_room.sbatch` and finished **COMPLETED / 0:0 in 18 s**,
Hala, one A6000, four CPUs, 64 GB, ten-minute limit. Partition and QoS were
both explicitly `debug` and were verified unchanged. Actual selected GPU:
`GPU-859eda35-2554-7da2-b14d-08f16680bd0d`.

Canonical input-Gaussian results, one scene / eight held-out views:

| Metric | Value | Views |
|---|---:|---:|
| PSNR | 5.793959259955809 dB | 8 |
| SSIM | 0.13261395748345295 | 8 |
| LPIPS | 0.9085933044552803 | 8 |

These low values describe the first-48-TRAIN engineering reconstruction
under the fixed TEST cameras. They are retained without changing the
view roster, recipe or thresholds in response. They do not establish a
full 50-scene comparison, factorization quality, or object fidelity.
The other seven method rows remain null, object records remain zero,
and `paper_ready=false`.

The unchanged evaluator requires repository-local, non-symlink paths.
Therefore the 16 paired PNGs, render/metric receipts, canonical manifest
and table are under:

```text
/group/worldcept/code/SimAny-wt/e2-raw-room/outputs/icra2027/20260905-edd5482-v4/raw_room
```

Shared canonical E0, submission, plan, execution, output index and postrun
receipts are under:

```text
/group/worldcept/code/SimAny/outputs/icra2027/20260905-edd5482-v4/raw_room
```

Independent postrun verification rehashed all 26 local output files,
confirmed exact decoded reference RGB for all eight images, and validated
the original source/camera plan plus render-receipt-to-metric-input digest
chain. Canonical validation reports eight samples per metric, no coverage
loss, no generator leakage and complete LPIPS weight provenance.

- `postrun_audit.json` SHA-256:
  `4ba9de9856f5dfd12574f190c36cf41169ad2a171a251e0e698f01a7c6352749`
- `table/fidelity_table.json` SHA-256:
  `ac1d20ab0fd66702eda558a9071df7e7ec30fe066f1a4ae15aee57a3de3d5d5e`
- Frozen experiment config SHA-256:
  `249e02b07cd8ce77bc162e2c9dfeba4bb81b9a53173d90971c5c62f604e42506`

Measured stage wall times were 6.217 s for rendering and 3.699 s for metrics;
the driver took 14.852 s including subprocess overhead. Runtime evidence
uses targeted executable/module/RECORD/native byte identities, not a claim
of complete dependency closure. No additional GPU rerun is needed for this
smoke. Full cohort and missing method/object evidence remain separate work.

## 2026-09-05 fixed fresh Gaussian cohort

Owner root; branchagent/icra-e2-raw-cohort; finalsource29042d60f9bbf5006cb02498af89b5ed3564dfe4, clean and published inmain.
Smoke command: `python -m pytest -q tests/test_e2_raw_cohort.py tests/test_e2_raw_room.py tests/test_e2_fresh_readiness.py tests/test_e3_discovery_cohort.py --basetemp=.t/c2`, PASS82/7.01s including leakage, missing-scene, source, metric-sample and worker-routing negatives. ExactE0 preflightPASS128/22.58s; integrationfullsuite1384passed4skipped168.73s.
Pilot command: generated `20260905-b3b85c6-v1/raw_cohort_smoke.sh` calls existing e2_raw_room run twice andaggregate once.826386 COMPLETED0:0/48s/Hala1A6000/4CPU32G. Two fixed scenes38d58a7a31+5748ce6f01;16TEST views. Raw-only PSNR20.1183147797569,SSIM0.8231592801936524,LPIPS0.3284507356584072; seven other rows null, paper_ready=false. Machine tableSHAa6f4c07ce8bc37ccfa553d02b855270c1b21d780d716ae80745b7e75f3a6ca11; source path in shared raw_room/aggregate_receipt.json.
Full command: same canonicaladapter from cohort_plan.sh/cohort_run.sh/cohort_aggregate.sh in20260905-b3b85c6-v2, source/config/script/job identities in *_submission.json.50CPU plans826397 complete; GPU826398 has26complete/24strict-preflight failures. No failed unit has an execution_claim or render receipt.826399 dependent aggregate cancelled.
Diagnosis826527 CPUsof1-h200-5: original1.26.4 driver reproduces24/24 failed plans; frozen2.2.6 render runtime reproduces5/24. Failure is floating-point camera recomputation across NumPy/BLAS/host dispatch, not a model result. No camera value or tolerance was changed. Preserve all26successful outputs and24unexecuted source plans; a future recovery must authenticate the frozen camera bank and reused generator/config/checkpoint/source identities.
FullE0digestbfaf7ab51e2ba0f3d139bcc2e5f7045a6f38026163e45c77422901c2a2730c15; smokeconfigSHA26e5b247abade4403caad7315781da1bf3a85bb2792a1cb638397bdd50799460. AlexNet checkpointSHA7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02; otherruntime/source identities frozen in configs andreceipts.
Passed criteria: fixed ordered50roster,8views per scene, no TRAIN/TEST filename or byte overlap, canonical400-view aggregation smoke, truthful26/50coverage. Failed criterion: fullGPU preflight replay across environments. Full400metrics NOT_RUN, completeTableII/E2 claim NOT_RUN.
