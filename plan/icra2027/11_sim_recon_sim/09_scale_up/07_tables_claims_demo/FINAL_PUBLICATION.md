# Guarded final native publication

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


`robo.eval.native_publication` is a completion/QA driver for the existing
`native_final_release` and `paper_pipeline`; it neither recomputes measurements
nor changes experiment settings. The existing primary-only final-table watcher
continues independently and remains an input dependency.

The driver waits for its successful final release, all 2,400 terminal primary
units, all 1,200 terminal L1 phase units, and all 86 terminal warm rendering
attempts. Legitimate build failures/abstentions retain null native success;
failed rendering identity controls remain in the attempted denominator. No
pending unit becomes a failure. Existing roster/hash/source validation runs
again before a new immutable freeze is reserved.

A deployment plan must bind:

- `code_commit`: exact clean merged producer commit; it must remain an ancestor
  of local `main`, which may advance for other work.
- `paper_repo`, `paper_remote`, `paper_base_commit`: original paper checkout and
  remote main must both be clean/pinned at the approved base
  `6e7bbcb6ebda059d724f590ff178d7c7bbc3e466`.
- `paper_worktree`, `paper_branch`: new, previously unused isolated destination.
- `qa_python`: installed interpreter with PyMuPDF, currently
  `/group/streetsplat/miniconda3/bin/python`.
- `demo_python`: native environment interpreter for the existing CPU-only
  `interface.demo_agentic native-hero` command.
- `primary_release_receipt`: the six-method primary-only watcher's future
  `control/completion_receipt.json`.
- `experiment_plan`: `{path, sha256}` for that exact six-method experiment plan.

Commands, run from the clean pinned code worktree with different new control
paths for the check and durable watch:

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.native_publication check \
  --plan outputs/publication/plan.json --control outputs/publication/smoke
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.native_publication watch \
  --plan outputs/publication/plan.json --control outputs/publication/control --max-hours 24
```

Only after every measurement gate closes does the driver generate/transfer new
TEST files with the existing pipeline into an isolated paper worktree. Every
previously tracked paper file must remain byte-identical, including the approved
title, legacy negative tables and DEV section. The six disabled scientific
claims must remain disabled. The generated primary section does not turn scope,
GS, feedback or preservation into a positive claim; a compact B3/B4 by L0/L1 scope table and paired intervals use only existing
T4 estimates in the extended version, with generated separate-scope accounting
in the conference version. Incomplete scope transfer is rejected.

The driver invokes the existing `build.sh`, then checks LaTeX errors, undefined
references, overfull boxes, visible placeholders, every rendered page, embedded
fonts, and the eight-page conference limit including references. The extended
version is not subject to the conference page limit. A normal non-force push
occurs only after those checks, the existing native hero/teaser/loop renderer
passes its QA against this exact release, and a fresh paper-base/remote guard.
The CPU-only demo has a bounded 900-second timeout and preserves its fixed
source selection; no alternative episode is chosen on failure. Native-track
readiness does not enable the legacy full-room demo claim. Concurrent
paper changes block publication. On any failure, `completion_receipt.json`
records `BLOCKED`, exact error, preserved tracked diff and copies/hashes of new
untracked generated files. A successful push is separately recorded before
updating the original checkout, so a later local failure cannot conceal an
already-published commit. No automatic editorial correction is attempted.

Smoke coverage includes complete/partial population gates, omitted and failed
appearance controls, duplicate claim gates, actual local Git remote publication
and QA rejection, concurrent remote changes, source ancestry, original negative
text preservation, and page/font/reference/box/placeholder rejection. This is
an implementation gate, not evidence that the eventual complete TEST paper fits
in eight pages. Do not start a worker from an unmerged source; pin the integrated
commit first.
