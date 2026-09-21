# N7 — Tables, supported conclusions and presentation

**Priority P0 for scaffolding, final integration later. Owner: statistics/paper/demo agent.** Read common matrix, parent TABLES_AND_METRICS, existing canonical metric producers, `robo/eval/paper_pipeline.py`, `robo/roundtrip/milestone_report.py`, and demo utilities. The old milestone reporter is example-specific, not the new matrix source of truth.

## TODO

- [ ] Define input schemas and produce empty/null tables before full jobs finish. They are planning artifacts, not results.
- [ ] Ingest N1/N5 merged ledgers and N2/N3/N6 metrics, retaining every planned method/scope and failure. Extend the canonical paper pipeline, no manual LaTeX numbers or a parallel conflicting metric implementation.
- [ ] Report instance/layout support beside episode support. Distinguish attempted, executed, completed, failed-to-build, abstained and externally unrun units. Never turn missing model output into a measured zero.
- [ ] Bootstrap paired layout clusters then instances and matched resets; report pointwise descriptive uncertainty and the limited independent layout count. Give signed point differences and contingency counts, not only absolute gaps.
- [ ] Check per-task native success and reference difficulty. All-failure pairs with zero gap do not establish preservation. Any noninferiority margin must be declared on DEV, not selected after observing TEST intervals.
- [ ] Keep fixed-prefix replay, full-horizon replay and primary native-policy outcomes in distinct tables/protocols. Relative-marker displacement is not absolute pose error.
- [ ] Build a claim ledger that separates operational software validation from scientific outcome support. A failed hypothesis remains a completed valid experiment, not a reason to change labels or sample until it wins.

## Planned main tables: missing cells remain empty

### T1a — Held-out appearance (separate platform/sensor/scope blocks)

| Method | Scenes available/planned | Views available/planned | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| Source GS before factorization | | | | | |
| B0 composite | | | | | |
| B3 composite | | | | | |
| B4 composite | | | | | |

Native RGB is the held-out target. Do not report infinite native-self PSNR.

### T1b — Geometry and construction cost

| Method | Objects accepted/planned | Geometry matched n | CD cm | F1@20 | Actual stage time/build |
|---|---|---|---|---|---|
| Observed surface / TSDF | | | | | |
| B0 fixed TRELLIS | | | | | |
| B1 fixed proposal priority | | | | | |
| B2 evidence selection | | | | | |
| B3 registration retry | | | | | |
| B4 context verification + repair | | | | | |
| BM same maximum action budget | | | | | |

Conditional geometry is compared on clearly stated common matched support; changed acceptance is not an unconditional gain. Retain native-prior/method-specific compute differences.

### T2 — Paired native manipulation

| Method | Instances/layouts | Executed/planned | Native success/planned | Success/executed | Delta vs REF (pp), 95% CI |
|---|---|---|---|---|---|
| REF_NATIVE | | | | | |
| B0_FIXED_NATIVE | | | | | |
| B3_AGENT_NATIVE | | | | | |
| B4_ROOM_REPAIR_NATIVE | | | | | |
| BM_BUDGET_MATCHED_NATIVE | | | | | |

Add per-native-task breakdown, discordant-success counts and outcome taxonomy in supplementary outputs. Do not invent weighted stage scores. Reference-vs-reconstruction is a complete asset-bundle comparison.

### T3 — What does system verification add?

| Variant | Tasks accepted/planned | Native success/planned | Independently checked room stability | Extra tool calls | Actual time |
|---|---|---|---|---|---|
| B3 logging only | | | | | |
| V1 verify/abstain only, optional arm | | | | | |
| B4 verify + adaptive repair | | | | | |
| BM fixed-order same maximum budget | | | | | |

B4's own check-pass rate is not independent validation. V1 needs its own measured arm for direct repair-vs-rejection conclusions; leave it unmeasured if not run. Room stability must identify actual engine, collision asset and protocol.

### T4 — Scope expansion

| Method / scope | Reconstructed and retained context | Executed/planned | Native success | Replay position error, if valid | Coverage |
|---|---|---|---|---|---|
| B3 L0 target only | | | | | |
| B4 L0 target only | | | | | |
| B3 L1 target + destination | | | | | |
| B4 L1 target + destination | | | | | |
| B3 L2 workspace | | | | | |
| B4 L2 workspace | | | | | |

Reuse L0 from the exact same fixed scope-subset roster. L3 full-room cases are separately labeled with inventory coverage; no L0-to-room headline inference.

## Supplementary evidence

Full-horizon replay vs closed loop with equal budgets; initial pose errors and correspondence-qualified trajectory errors; unchanged-native identity controls; native predicate components and containment checks; native/GS/HC observation block; sensor-regime and policy-training split details; failures and compute. No FID or invented combined simulator metric.

## Conclusion ladder

1. **Feasibility**: current one-target DEV episode only. Phrase as an existence result, not a population rate.
2. **Multi-instance usability**: completed independent TEST instances with known sensor/scope/oracle context and meaningful reference competence.
3. **Agentic benefit**: B3 vs B0 and B4 vs B3 on paired native outcomes with coverage/cost. Geometry-only gains do not establish this.
4. **Decision quality beyond compute**: B4 vs BM with shared action bank and actual cost reporting.
5. **Repair beyond rejection**: B4 vs V1 on successes/planned, not merely better conditional success after abstention.
6. **Closed-loop compensation**: multiple equal-horizon replay/feedback examples; not the old 392-prefix-vs-429 example.
7. **Scope**: destination/workspace/room conclusions require that exact reconstructed scope, not many target-only instances.
8. **Appearance benefit/preservation**: measured B4 native/GS/HC comparison; no inference from uniform-color native success.

Do not promise best-paper selection or manufacture a positive conclusion. Do not silently change the user-approved title; propose a necessary scope/title adjustment with its evidence if results no longer support it.

## Demo deliverables

Immediately produce a clearly labeled ~30-second DEV progress video after real multi-instance results: input RGB-D, reconstructed object/collision, native reference vs reconstructed continuous policy footage, unchanged native success display, exact scope and source IDs. Use actual artifacts, not generated scientific evidence.

Final: 80-95s hero, 30s teaser, 8s loop, poster, subtitles and local-video presentation fallback. Show 3 visually distinct predeclared canonical examples, one genuine selection, one real repair (including failed repair if that is the observation), move-and-reveal GS only when implemented, and synchronized collision/appearance from the SAME state trace. Independent policy episodes may diverge; do not splice them into an apparent single success or force equal state.

Use existing presentation palette and typography. 1080p/30fps master, labels >=28px, safe margins, restrained overlays, no fake chat UI. Archive all planned demo candidates; select representative median cases by a declared rule. A separately labeled best-case montage is allowed only with clear illustrative status and full quantitative reporting. Results come directly from final release JSON, never old 0.783/28.32 figures copied into new-cohort cards.

## Publication and QA

Bind stage commits/assets/configs/checkpoints and actual matrix counts in `release_manifest.json`. Generate `table_provenance.json`, `comparison_statistics.json`, `claim_ledger.csv`, `scope_inventory.csv`, `demo_source_manifest.json` and failure inventory. Update paper measured claims by track: the old ScanNet++ E4 had no policy trials; the native track now does, so avoid a stale project-wide statement in either direction.

When publishing, read the current PDF/tool instructions, compile the actual paper source, render every page, inspect readable tables/figures at final size, check references/fonts and the intended 8-page layout. Do not claim PDF QA merely because LaTeX exited zero. No font files, restricted assets or checkpoints are redistributed in the lightweight bundle.

Acceptance: toy-data aggregation and duplicate/missing-row tests pass; all table fields have source/unit/denominator; every conclusion is supported or explicitly narrowed; real continuous demo sources match release. `STATUS.md` lists implemented producers, completed blocks, scientific findings, remaining unmeasured claims, paper commit and rendered-output paths.
