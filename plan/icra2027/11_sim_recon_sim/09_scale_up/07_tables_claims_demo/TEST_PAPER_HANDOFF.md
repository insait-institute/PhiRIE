# Native TEST paper handoff (preparation, not a scientific release)

## Completed native TEST publication — 2026-09-07

Owner `/root/sr2_import`; state `COMPLETE` for the bounded native table/paper/demo release. Producer branch `agent/icra-n7-publication-3b5e853`, clean source `3b5e853dcf246b6a65cc843624d4ff863c251a0e`. No publication worker remains running. Historical waiting/blocked receipts below are preserved and superseded.

- Final immutable freeze: `20260907-3b5e853-v1`; authoritative tables: `/group/worldcept/code/SimAny/outputs/icra2027/20260907-3b5e853-v1/sim_recon_sim/scale_up/paper_tables`.
- Release manifest SHA256 `dd1ee0c0be35c74225228128282c96b88b60a30345c3065be282c880a9d2e0e0`.
- Existing pipeline validated primary2400/2400 terminal (1350 actual episodes,1050 construction/abstention nonrollouts), separate scope1200/1200 terminal (600 actual episodes), warm86/86 attempts (82 metric receipts,4 exact-RGB identity failures). No pending unit was converted to native failure.
- Ten original prepolicy B0 `CODE_FAILED` outcomes were independently audited into `BUILD_FAILED` through the strict canonical merger. Original collision geometry, thresholds, native outcomes and raw shards remain unchanged; near-threshold local nonconvexity is disclosed, not described as an open mesh. Producer acceptance and native executability remain distinct.
- Guarded generated paper push: `79d003cb80473841c41e37d6deabbc8ee252120c`; separate qualitative abstract correction: final paper main `4eaa780caa9bc80da7493ef8bdb96f7b73b4cd85`, both pushes verified with `git ls-remote`. No experimental value was hand-entered into paper text.
- Final shared `/group/worldcept/code/SimAnyRoom` PDFs rebuilt after the editorial change: conference8pages, extended11pages. All19 pages rasterized, fonts embedded, no LaTeX errors, undefined references, overfull boxes or visible placeholders. Actual conference pages1/7 inspected without overlap. Original approved title and all historical ScanNet++ negative statements/tables preserved.
- Conference PDF SHA256 `6134220ae8e933aa9234b536ef10823b7cc2b70c56b3da89078ca52fc4b12723`; extended PDF SHA256 `7bc339ea33173a65df93d16bb871710e246b6f51cd9e955a50f93d8df787df45`.
- Native video QA PASS:2700 frames,90s,1920x1080, full decode and exact8s loop seam. Strict source closure includes first declared pair, genuine parent-to-retry proof and complete labelled2x repair episode. Video hashes are sealed at release creation, not asserted retrospectively at original acquisition. Native track ready; full-room demo claim remains false.
- Completed CPU publication command: `/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.native_publication watch --plan outputs/final-publication-v4/publication_plan.json --control outputs/final-publication-v4/publication_control --max-hours 1`, cwd `/group/worldcept/code/SimAny-wt/n7-publication-3b5e853`. PID1914182 exited successfully; no Slurm job or expensive rerun required.
- Publisher plan SHA256 `b04284b79cc108881423b33109e6219f444450b96dacf9813d6d5deea0b4de5b`; terminal receipt `outputs/final-publication-v4/publication_control/completion_receipt.json` (`PUBLISHED`). Final editorial QA at `outputs/final-publication-v4/editorial/pdf_qa/qa.json`; tracked paper audit `audit/native_abstract_editorial.json` binds unchanged original bytes and generated evidence.
- Validation:76 integrated table/paper/publication tests PASS3.87s; full real pipeline, PDF QA and native demo QA PASS. Earlier9page publication attempt0886b4e-v4 remains immutable and BLOCKED; compact successor moves complete detail to extended without changing fonts, margins or measurements.
- Claim gates: narrow B3 selection/retry bundle versus fixed service-success improvement is supported by generated paired interval. Preservation, benefit beyond extra compute, adaptive repair advantage, full room/L2 capability, GS occlusion correctness, real-robot transfer and feedback-gain claims remain disabled. B4 coverage loss and L1 destination losses remain explicit. Conditional fidelity common support12/48 and4 failed render identity controls prevent a global fidelity claim.

Final presentation bytes under the same freeze's `sim_recon_sim/scale_up/native_demo`:90s movie7,584,352;30s teaser3,705,738;8s loop152,125;poster101,457;SRT482;source manifest354,919;QA227. SHA bindings are in `publication_control/native_demo_receipt.json`. Frozen source data and outputs remain independent of the later editorial paper commit.


The current approved publication base is `6e7bbcb6ebda059d724f590ff178d7c7bbc3e466`;
`053eb2e21d252f66a2cbe303ee1952218e0f0e00` below is the historical preparation base.
The active code repository is the `main` worktree at `SimAny-wt/icra-main`;
`/group/worldcept/code/SimAny` remains the older E3 worktree and the output root.
Do not confuse its HEAD with the native producer source.

## Implemented publication changes

- Code `8e1cb17`: `robo.eval.native_scale_paper` accepts one declared DEV or TEST target-only primary block. TEST writes `native_test_results.tex`, `native_test_section.tex` and optional `native_test_appearance.tex`; it never overwrites historical DEV files.
- Code `dc99ef1`: the final transfer hook requires every primary outcome terminal before copying any file to the paper. Legitimate build failures and abstentions count as terminal; a pending external outcome does not. Transfer audit is `audit/native_test_transfer.json` or `audit/native_dev_transfer.json`, with the actual tier. No metric/statistics changes.
- Paper `8926a70`: `paper_sections/04_results.tex` prefers the generated TEST section only after that file exists; the original generated DEV section remains separate in the extended paper. With no TEST file, the current DEV manuscript remains active. Limitations retain privileged context and binding-specific native predicates. Approved title and ScanNet negative-result tables are unchanged.
- TEST preview tables mark incomplete snapshots explicitly. Population rates stay null through existing canonical tables; partial pairs cannot become conclusions. Complete paired prose is consumed from the existing `test_conclusion_ledger`, never recomputed or hand-entered. Conditional gain versus service loss remains explicitly described when the measured relation holds.
- Appearance prose only claims byte-exact controls for admitted quality units. Failed controls, missing views and construction failures stay in planned denominators. Native full-frame appearance is not Gaussian/object-only appearance.

## Actual validation

`python -m pytest -q tests/test_native_scale_paper.py tests/test_native_scale_tables.py tests/test_native_scale_evidence.py`: **44 PASS**, including rejected incomplete DEV/TEST transfers, separate TEST/DEV artifacts, mixed-block rejection and preserved negative outcomes.

Actual producer `8e1cb17` was run through `robo.eval.paper_pipeline`, with only `native_paper_section` changed in copied existing configurations:

- Complete DEV source: `20260907-044e56c-v1/sim_recon_sim/scale_up/paper_pipeline.json`.
- Explicitly partial TEST source: `20260907-630ee6c-v1/sim_recon_sim/scale_up/paper_pipeline.json` (historical immutable snapshot006; this is a renderer regression boundary, not the latest status report).
- New immutable validation stage: `/group/worldcept/code/SimAny/outputs/icra2027/20260907-885edc0-v2/sim_recon_sim/scale_up/native_paper_preparation/`.
- Commands: `python -m robo.eval.paper_pipeline --config <stage>/DEV_config.json --out <new-output>` and the same command with `TEST_PARTIAL_config.json`.
- Existing `DEV_tables` is COMPLETE_BLOCKS, 200/200 terminal; `TEST_PARTIAL_tables` is INCOMPLETE, 1228/2400 terminal. All release artifact hashes checked; partial population rates and claim sentences remain unmeasured/disabled. No paper transfer occurred.
- Exact source/config/result hashes: `validation_receipt.json`, `DEV_source.json`, `TEST_PARTIAL_source.json` in that stage. DEV release SHA `2e1cb4950f312a3a9e9dce1849cda436dae987b986be006c1724f972fe78cd81`; TEST preview SHA `41755fdfa5a8b228426a4178efd913bcd8e7cdb087d12934089ecd482c97689c`.

The paper template was compiled using `bash build.sh`: conference **8 pages**, extended **9 pages**. All pages rendered and all fonts embedded; no LaTeX error, undefined reference or overfull box. Conference contact sheet visually checked. QA is in `SimAnyRoom-wt/native-test-template/audit/native-test-template/qa.json`. This verifies the current DEV-active template, **not the pagination of future TEST contents**, which must be checked after the complete release.

## Claims that remain disabled

The generated `native_paper_claim_gates.json` lists exact prohibited positive sentences and required evidence. It never automatically promotes scientific claims:

| Claim | Exact sentence that must stay disabled | Current gate |
|---|---|---|
| Preservation | SimAnyRoom preserves native manipulation capability. | NOT_RUN: no prospective noninferiority/preservation criterion; nonsignificance is insufficient |
| Full room | SimAnyRoom reconstructs complete rooms while preserving manipulation. | NOT_RUN: target-only primary and partial scope diagnostics do not establish full room construction |
| Full L2 | Full workspace reconstruction retains task capability. | NOT_RUN: partial role import is not complete declared workspace support |
| GS occlusion | Gaussian observations preserve correct robot occlusion. | FAIL for the current DEV approximate-depth replay; no fresh native observer or policy admission |
| Feedback | Visual policy feedback compensates for reconstruction errors. | NOT_RUN: singleton/replayed diagnostics do not establish the population feedback effect |
| Sim-real | Native reconstruction transfers policy success to the physical world. | NOT_RUN: genuine matched physical trials required |

`claim_preparation_audit.json` in the validation stage authenticates the new adverse evidence without leaking it into construction:

- First declared TEST r0 scope diagnosis: `20260907-6a60a51-v6/.../test_l1_execution/diagnostics/first_declared_r0/diagnosis.json`. One instance, same engine/reset, failed L1 and measured robot–destination interference. This is not a population scope result.
- GS DEV replay audit: `20260907-c2dc6cf-v4/.../gs_depth_replay_dev/renderer/observations/validation_audit.json`, bound to `observation_invariance.json`. Static public-depth evidence exposes wrong visibility classification, holes and unknown coverage; later dynamic states cannot be compared to unchanged capture depth. Renderer implementation checks are distinct from complete/correct occlusion. No new policy episode was run.
- Existing ScanNet matched-geometry versus isolated-stability negative and the DEV B4 conditional-success/coverage penalty must remain visible; TEST does not erase development diagnostics.

## Final command after the completion gate

The durable producer remains **unchanged** at `/group/worldcept/code/SimAny-wt/n7-final-tables-5cf76bd`, successor PID739136 after the complete-warm-method handoff. It writes no paper files. Its completion receipt will be:

`/group/worldcept/code/SimAny-wt/n7-final-tables-5cf76bd/outputs/final-table-watch-six-methods-v1/control/completion_receipt.json`.

Only after that receipt is `GENERATED`, its bound release is COMPLETE_BLOCKS with 2400/2400 terminal, and the immutable input boundary has been verified:

1. Use the clean merged main producer containing the changes above. Read the final config path from the receipt's `command` (`--config`). Preserve its exact planned ledger, geometry/appearance sources, failure receipts and separate scope boundary.
2. Reserve a **new** output ID with `robo.eval.freeze.reserve_freeze_id`. Copy the final config into that new freeze and set only `native_scale_up.native_paper_section=true`. Record parent config SHA and the new producer SHA; never edit the frozen watcher's config/output.
3. Run the existing final producer, with no alternate table or metric pipeline:

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.paper_pipeline \
  --config <new-freeze>/sim_recon_sim/scale_up/paper_pipeline.json \
  --out <new-freeze>/sim_recon_sim/scale_up/paper_tables \
  --paper-root /group/worldcept/code/SimAnyRoom
```

4. Compile both paper cuts, render every page, validate embedded fonts/references/boxes, and enforce the 8-page conference limit. Reassess exact claim gates; complete primary accounting does not make scope, GS, feedback or physical-trial gates pass. Preserve the complete DEV source in extended text and keep scope controls separate from primary episodes.

No current partial TEST paper transfer or automatic paper promotion is authorized by this preparation artifact.


## Complete warm-fidelity handoff and current final trigger

On 2026-09-07 the bounded warm render population completed:86 planned render units,82 admitted renders with82 canonical metric receipts,4 unchanged-native RGB identity failures and0 unattempted units. New partial release `/group/worldcept/code/SimAny/outputs/icra2027/20260907-a1832f4-v1/sim_recon_sim/scale_up/paper_tables/` was generated by clean source `c60f3e3f0660334642a9ee085b1dd2d5c04e238f`; it binds primary snapshot010 and closed L1 execution snapshot0045 separately. It is **not** the final primary release.

Exact reproduction uses the existing producer and a new output directory:

```bash
cd /group/worldcept/code/SimAny-wt/n7-complete-appearance
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.paper_pipeline \
  --config /group/worldcept/code/SimAny/outputs/icra2027/20260907-a1832f4-v1/sim_recon_sim/scale_up/paper_pipeline.json \
  --out <new-output-directory>
```

No `--paper-root` is used. The frozen config binds all152 geometry method-instance measurements through86 source receipts, final192 construction decisions, and the full pre-render-declared six appearance methods. The prior baseline listed only four methods; the two omitted B4/BM names were restored from the **original frozen render manifest**, without changing inputs, views, thresholds, methods or metrics. The appearance-method declaration does not use native policy outcomes.

Appearance quality is conditional on the same12/48 instances and24/96 views across all six methods. Available instance coverage is29/48 forB0,31/48 forB1/B2/B3 and12/48 forB4/BM. All four failed controls stay outside quality and inside the planned denominator; no raw-RGB substitution. Positive infinite PSNR from exactly equal decoded RGB is preserved. Retained native backgrounds dominate full-frame scores; no global fidelity, object-only, Gaussian, geometry or manipulation claim follows.

Release SHA256 `afee8dfb5a89463389bf5b8a1692f072bbfe1e9b93955a24b8402b3724327a11`;52 artifact hashes verified. Appearance JSON SHA256 `4b009c6e71c3b8fdfcd87f055f6d040a6143ef09afa5a0d83b1b6c883352aef7`; CSV SHA256 `f08038d821b10203ccea39e1c4fd84c15e1786cb9e4747d870d635e5144379e2`. `input_boundary.json` binds all failure receipts and both cohort boundaries.

The old PID413924 was stopped only after verifying that it was waiting, had no child pipeline, had produced no final completion receipt and the primary completion gate still returned unready. Original `outputs/final-table-watch/{plan.json,control/launch.json}` remains historical and unchanged. The unchanged clean `5cf76bd` producer now runs **PID739136**, at most24h, with new immutable configuration under:

`/group/worldcept/code/SimAny-wt/n7-final-tables-5cf76bd/outputs/final-table-watch-six-methods-v1/`.

Its `plan.json` SHA256 is `bef35bdb96af10171b3a7e71a7d6aea8757bc7b2a94cd66431cc624a46659a16`; only the baseline appearance-method list changed. `handoff_requested.json`, `handoff_stopped.json`, `waiting_gate.json`, `control/launch.json` archive parent/new plan, source and config hashes. The current future terminal receipt is **`control/completion_receipt.json` in this successor directory**. No automatic paper transfer or scientific claim promotion is enabled.

## Historical guarded publication v1 (paused; 2026-09-07 04:52 UTC)

The complete-data publication continuation is running as **PID926009**, separate
from primary-only table worker739136. Its pinned code source is
`b26fc6795c99b255b431dcd505d6641345b7b770`; source main may advance while this
worktree remains clean and unchanged. Plan:
`/group/worldcept/code/SimAny-wt/n7-publication-b26fc67/outputs/final-publication-v1/plan.json`
(SHA256 `7a9bcc0bf8c063f536f5df21e5a64bba16e5477812f0a2d24869422779924914`).
The current check receipt is `smoke/completion_receipt.json`: `WAITING`, no paper
mutation. Its final status will appear in `control/completion_receipt.json`.

This successor does not bypass the primary-only watcher: it requires that
watcher's successful complete release, then independently waits for the full
1200-unit scope phase and all86 warm attempts. It binds those immutable
boundaries, regenerates through the existing pipeline in one new freeze, and
uses the exact clean paper base6e7bbcb in a new isolated worktree. Compact scope
rows are B3/B4 by L0/L1 with fresh controls and inherited/additional failures;
paired intervals stay in the extended report, generated scope accounting also
appears in the conference text. Partial scope transfer is rejected.

The existing native-hero command builds90s/30s/8s assets from the same release,
with fixed episode selection and native-track-only readiness. Paper push needs
complete data, original title/negative/DEV bytes unchanged, six disabled claims
still disabled, successful compilation, conference at most8pages, all pages
rendered, embedded fonts, no unresolved references/overfull boxes/placeholders,
and native-demo QA PASS. Any failure preserves exact evidence/generated diff
and ends BLOCKED without an automatic editorial patch. Full TEST page fit is
still unproven. Deployment details: [FINAL_PUBLICATION.md](FINAL_PUBLICATION.md).

## Current strict-media publication successor (2026-09-07 05:11 UTC)

Publication926009 is stopped, with immutable waiting/no-mutation pause receipts.
The live successor is **PID1048710**, pinned clean/pushed source
`04cc514846ece726ab319640c63591944b5bb19c`, worktree
`/group/worldcept/code/SimAny-wt/n7-publication-04cc514`.

Plan `outputs/final-publication-v2/plan.json` has SHA256
`e6b3b7960671f0a6414f0ca0b3fa1a54eb1263c8879fc73709d613e462198164` and binds
its predecessor stop plus real DEV film841437 QA and strict TEST645-source
preflight. Actual check returned WAITING with no paper mutation; future status
is `outputs/final-publication-v2/control/completion_receipt.json`. The separate
primary-only watcher739136 remains unchanged.

The final pipeline now seals each available executed worker video, action file
and trace in release source lineage, and includes the genuine retry registration
and evidence sidecars. Consumers verify these existing release hashes; they may
not certify arbitrary current video bytes themselves or replace the actual
retry parent with a fixed-B0 proxy. The seal is explicitly at release time, not
at original acquisition. A new sealed DEV release passed the actual90s film
regression, and sealed partial TEST release04cc514-v2 authenticated645 sources
while correctly rejecting partial rendering. No prior frozen output was
replaced. Full data, original paper bytes, strict QA, same-release demo and
non-force push guards are otherwise unchanged.
