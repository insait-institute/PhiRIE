## 2026-09-06 11:40 UTC — scoped full evidence transferred to paper

Canonical publication20260906-8cae8a0-v1/source93c9566 and pushed paper83c97b1 now include this task's available full output, preserving all denominators, nulls and failed gates. 389focused/130exactE0PASS;8+8page QA PASS. See E9 STATUS for commands and hashes. Task-specific remaining measurements and limitations below remain open; publication does not promote the scientific gate.

---

## 2026-09-06 08:48 UTC — full E1 aggregation and E2 backgrounds complete; recovery audits active

Live receipt `outputs/icra2027/orchestration-status-20260906T084810Z/status.json` plus sacct/squeue snapshots. Artifact checkout402cda0 and paper18def27 remain unchanged. No common final scientific freeze and no claim that all TODOs are complete.

E1 sourcecd9e8ff/freeze20260906-3e8ab6d-v1: all50ordinary835082–835131 COMPLETED0; canonical aggregation835404 COMPLETED55s, noGPU. Generated `construction/table/construction_table.json`:1871inputs,399controller accepts,394verified exports;44successful scenes,4empty,2preserved room rejections.361/394 independent historical-protocol isolated drops stable; yield0.21058257616247997 and stability0.916243654822335. F1/fullruntime null;valid_for_paper=false. Four other input families remain NOT_RUN in250scene/regime records. Original E3 acceptance probe and E4 room stability are distinct and preserved. Command `python -m run.icra2027.e1_full_drop --config STAGE/execution.json --stage STAGE --expected-commit cd9e8ffdfcae2a32d0d880b8f27e9cad04b2a5c8 --aggregate`; exact launch in STAGE/launchers/aggregate.sbatch.

E2 source8a09ebe/freeze20260906-0be94c7-v1:39new fills/no-ops835033–835071 COMPLETED; independent835077 PASS1m55.40bundles include originalpilot,36cleaned positive scenes and4unchanged zero-object backgrounds;10TRAIN-blocked/80missingTESTviews retained. No new held-out quality metrics yet. Full-render43dcd9e/v2 pilots835438/835439 failed before target render publication: PythonABI overlay mismatch (environment) and missing explicit original VERIFIED_REUSE receipt branch (codebug). New source7a1f85b repairs canonical original-source reuse validation;83focused/130exactE0 PASS; actualE0 4f17532d81db9a7bb6404c49d16778fc8ff68b825636856412542d255caf5cde. Newfreeze20260906-b689700-v1 pilots835460/835461 use unchanged original50/400cameras,36positive/4noop/10blocked scope and correct pinned Python3.10 runtime. No threshold or metric changes; full48followups wait realpilot validation.

E4 original terminal834890 TIMEOUT52m06, extension rejected by Slurm; source18cbd0d/freeze20260906-c3ba498-v1 preserved with no sealed aggregate. New05184e1/freeze20260906-b9bdcf3-v1 uses original-source independent per-scene replays and strict sealed cache composition,73focused/130exactE0 PASS, actualE0 dfb9d98da83d28fdbcb4875eb748502b8bd144a271c389eae212c324207d75f5. Ordinary CPU-only pilot835440/835441/835442 covers original prerequisite/no-tabletop/protected-carve cases; two are already COMPLETE. All50 original outcomes remain fixed, no construction or policy measurement. Full release awaits all3pilot seals.

E9 portable producerb71413d/freeze20260906-313ec4a-v1 completed current paper18def27 archive:311numeric fields,20claim decisions,266members; standalone relocated standard-library verification PASS.64focused/130exactE0 PASS; actualE0 c6990e2bde05809a5531c3a567fd86359f59a89c66750e80a4a7c03e68831857. Canonical `python -m robo.eval.paper_pipeline --bundle-existing ... --paper-root ... --publication-receipt ... --out ...` command in PORTABLE_BUNDLE.md. Archive `SimAnyRoom-audit-18def27.tar.gz`,53253760bytes,SHAee26cd841def89ef577edfb744893d3e8509291269c967e49907e1b7753c212b. Originalf0d3ee7/v2 packaging failure (numeric-leading scene key parser) is preserved; newsource/freeze fixes it. Publication remains20260906-6dcb0e7-v2,8+8pageQA. Paper has not yet been updated with new E1/E2/E4 outputs.

Claim gates: narrow E3 evidence selection/registration retry PASS; broadA4 superiority and E6 predictive applicability FAIL; manipulation/Harmonizer/physicalsuccess/finalhero NOT_RUN; finalsubmission FAIL. Root continues locally after subagent provider quota errors; these are orchestration limits, not scientific failures. No arrays and no silently replaced measurements.

---

# E1 construction-scale status

`AUDIT_COMPLETE=true`  
`READY_FOR_PAPER=false`  
`CLAIM_STATUS=CLAIM_FAILED`

## Decision

The strict 50-scene legacy-evidence audit completed successfully, but the
five-row construction claim is not publication-valid. The two scan-mesh rows
reproduce their audited numeric values, while the splat-fused and single-RGB
rows contain leakage, incoherent snapshots, or denominator changes that make
their current geometry scores inadmissible. All rendered products are marked
`paper_ready=false` and watermarked as preliminary.

No constructor or evaluator fleet was rerun. E1 used CPU-only read access to
existing evidence and wrote only new, no-overwrite artifacts below
`outputs/icra2027/`.

## Frozen implementation and upstream contract

- E1 implementation commit: `ac0dc5a`.
- E0 abbreviated-commit provenance fix: `c22afef`.
- Exact scheduled code commit:
  `c22afefb355cf1f52baaa25f1bb3c9fd7745bc22` (clean worktree).
- Pinned population: 50 unique ScanNet++ v2 `nvs_sem_val` scene IDs in
  `configs/experiments/icra2027/construction_regimes.yaml`.
- Population roster SHA-256:
  `7002a3dd0263ff6fb6895f8a4d424a790c3f6fc125fc682517c328cbee723913`.
- E0 refresh job: `814951`, `COMPLETED`, exit `0:0`, `00:00:08`, CPU-only on
  `hala`.
- E0 contract:
  `outputs/icra2027/preflight-smoke-20260903T075851Z-601696/contract/`.
- E0 contract SHA-256:
  `fd1a6715664dce867cc2a41973161d53d9fa93efc5f719c2e695628ac0e8bcf1`.
- The E0 manifest is clean and its code/config/freeze-inheritance checks all
  match E1. It remains `mode=smoke`, so it is not a paper-mode freeze.

## Slurm runs

| job | purpose | node/resources | result | output |
|---|---|---|---|---|
| `814948` | first clean E0 dependency refresh | `hala`, 8 CPU, 32 GiB, no GPU | `COMPLETED 0:0`, 13 s | `outputs/icra2027/preflight-smoke-20260903T075723Z-594869/contract` |
| `814951` | final E0 refresh at `c22afef` | `hala`, 8 CPU, 32 GiB, no GPU | `COMPLETED 0:0`, 8 s | `outputs/icra2027/preflight-smoke-20260903T075851Z-601696/contract` |
| `814954` | real two-scene/two-regime E1 smoke | `hala`, 8 CPU, 32 GiB, no GPU | `COMPLETED 0:0`, 3 s | `outputs/icra2027/icra2027-contract-v1-e1-c22afef-v1-smoke-814954/construction` |
| `814956` | full 50-scene/five-regime legacy audit | `hala`, 8 CPU, 32 GiB, no GPU | `COMPLETED 0:0`, 3 s | `outputs/icra2027/icra2027-contract-v1-e1-c22afef-v1/construction` |

The accepted smoke used `38d58a7a31` and `c4c04e6d6c`. Its scan-mesh totals
were GT `4/3` and automatic `3/2`; it produced four checked manifests and
correctly remained non-paper.

## Full strict aggregate

Every row uses all 50 planned scenes. `completed` means a terminal success or
report-backed empty scene; `contributing` means the prepared input denominator
is positive.

| regime | completed / contributing | inputs / accepted | yield | F1@20 (weight) | stability | min/scene | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GT segments + scan mesh | 50 / 47 | 789 / 457 | 0.579214 | 0.708191 (457) | 0.772429 | 11.6127 | numeric match, preliminary |
| Auto discovery + scan mesh | 50 / 47 | 1,082 / 671 | 0.620148 | 0.581961 (381) | 0.754098 | 16.9577 | numeric match, preliminary |
| GT segments + splat-fused mesh | 49 / 45 | 678 / 380 | 0.560472 | withheld | withheld | withheld | invalid legacy geometry |
| Auto discovery + splat-fused mesh | 2 / 48 | 934 / 559 | 0.598501 | withheld | withheld | withheld | incoherent snapshots |
| Single RGB + metric depth | 41 / 42 | 406 / 67 | 0.165025 | withheld | withheld | withheld | GT selection leakage |

The automatic splat row has only two report-backed empty scenes counted as
completed because all 50 derived-mesh paths are symlinks and are rejected
without following them; 48 positive-input rows are therefore terminal
`invalid`/`failed`, not silently called successful.

## Differences from the expected paper values

- **GT + splat:** the legacy 0.693 F1 is withheld. The saved run used GT scan
  geometry during registration/evaluation rather than isolating the
  splat-fused mesh. One scene failed; all 50 planned scenes remain visible.
- **Auto + splat:** the current coherent prepared denominator is `934`, with
  `559` accepted. The paper's `930`, 59.0%, and 0.553 combine a later 49-report
  population with only 33 older `eval_vs_gt.json` snapshots. The audit refuses
  that cross-snapshot join.
- **Single RGB + depth:** the presented-input denominator is `406`, including
  17 prepared instances in two generator failures, rather than the scored-only
  denominator `389`. The resulting yield is 16.5%, not 17.2%. The legacy 0.309
  F1 is withheld because representative-frame selection used GT centroids.
- **All rows:** original build commits/manifests and the authoritative dataset
  split-file hash were not recorded at build time. The audit-generated
  manifests preserve this absence rather than assigning the current commit
  retroactively.

## Missing-artifact inventory

- GT scan: independent matching record missing in 50/50; original build commit
  and manifest missing in 50/50.
- Auto scan: legacy GT evaluations exist, but the deterministic versioned
  one-to-one matching protocol is unrecorded in 50/50; original build commit
  and manifest missing in 50/50.
- GT splat: one acceptance/candidate/discovery/registration failure; no valid
  independent GT match, drop result, or runtime snapshot in 50/50.
- Auto splat: one acceptance/candidate/registration failure, 17 missing legacy
  GT-eval files, no drop/runtime snapshots, and 50/50 derived-mesh symlink
  paths rejected without traversal.
- Single RGB: five missing discovery outputs, seven missing candidate/
  registration/GT-match outputs, and no corrected drop/runtime result in
  50/50.

The authoritative detailed report is
`outputs/icra2027/icra2027-contract-v1-e1-c22afef-v1/construction/missing_artifacts.json`.

## Artifact integrity

- `scene_records.csv`: 250 data rows; SHA-256
  `b9168f94943613e250feeb21c829b516d75dd17cc1483e35d0f9bbbf1e6315de`.
- `missing_artifacts.json`: SHA-256
  `d833d4e5288df12cb900859b28d579bd495fc5e2857ac64fb6947023d8792923`.
- `construction_table.json`: SHA-256
  `46dced81824efb92c9dac2b57d970438715ee63ad22d157754243dc062d36d72`.
- `construction_table.csv`: SHA-256
  `0311a9a65657cae9ce8b819c8b6a844ec0dad65ed17dc8c6c2f849efff734984`.
- `construction_table.tex`: SHA-256
  `d2d35b68f510c2f4d198b89ce09a78ed1f8d34f6be94d113bca003c4ae026829`.
- `artifact_hashes.sha256`: 256 entries; the complete `sha256sum -c` check
  passed. List SHA-256:
  `ac11b6f725c873765dd007fa9251651bb794f7b802b7e3cef9a105c637dd4bca`.

## Verification

- Maintained repository suite at the final recorded source state:
  `365 passed, 1 skipped, 2 existing trimesh warnings` in 143.69 s.
- E1 focused suite at the same state: `55 passed`.
- Python compilation, shell syntax, `git diff --check`, normalized-manifest
  checks, table provenance hashes, and the full 256-file artifact hash check
  all passed.

## Required reruns before a paper claim

1. Produce a clean paper-mode E0 freeze with the authoritative dataset split
   file hash and immutable build identities.
2. Version `eval_vs_gt` with deterministic seeds, one-to-one matching, explicit
   input/output population hashes, and no in-place overwrite; rerun the
   automatic scan-mesh evaluation into a new freeze.
3. Implement a documented GT-segment-to-splat-surface transfer, then run the
   two-scene GT-splat pilot before any fleet submission.
4. Run automatic splat construction/evaluation into one coherent new snapshot;
   do not reuse the 33 stale eval files.
5. Run a true no-GT single-RGB pilot with independent held-out evaluation and
   a presented-input denominator that includes generator failures.
6. Add corrected drop tests and runtime manifests where the paper table expects
   those fields.

No current GPU launcher satisfies these gates, so no E1 reconstruction fleet
job was submitted after the audit. This is a scientific blocker, not a
scheduler blocker.

## Acceptance checklist

- [x] One normalized CSV covers all 250 planned scene/regime rows, including
  failures.
- [x] A five-row aggregate is generated by code with no manual table edits.
- [x] Existing numeric matches and every denominator discrepancy are traced to
  source artifacts.
- [x] Every normalized row points to a checked audit manifest.
- [ ] Original build commits/manifests are recoverable for every row.
- [ ] All five regimes have independent, coherent, leak-free geometry evidence.
- [ ] Table I is paper-regenerable from a paper-mode freeze.

## Current full-E3 / E4 input-scope audit (2026-09-06)

Owner `trellis2_environment`; branch `agent/icra-e1-current-source-audit`.
State: AUDITING; experiment/paper gates remain FAIL / NOT_RUN as above.
Source: the task branch commit containing `audit_current_sources.py`; final
machine snapshot binds the exact SHA and dirty state. No model, GPU, Slurm,
new physics or metric execution. No checkpoint changed.

Read-first source audit and metadata API smoke completed. Scoped tests:
`pytest -q tests/test_e1_current_source_audit.py tests/test_construction_metrics.py
 tests/test_construction_inventory.py`: 60 PASS (initial run 9.74 seconds).
Reproducible command and prospective missing-unit protocol:
`CURRENT_SOURCE_PROTOCOL.md`. Machine snapshots/logs are repository-local at
`outputs/e1-current-source-audit/` in the isolated worktree.

The full E3 input family matches Auto discovery + splat-fused mesh, but the
constructor, vocabulary, acceptance, drop/collision and runtime scopes differ
from the historical table. The new metadata audit preserves all 250 cells and
leaves every E1 measurement field null. It does not emit a comparable five-row
result. Current controller values remain separately identified evidence.
Sealed E4 export references are inventory candidates only, never E1 drop or
manipulation measurements; missing/in-flight scenes remain visible.

Pilot/full E1 commands: NOT_RUN pending prospective common protocol review.
Smallest next valid experiment is missing CPU export/drop closure after scope
and source admission, not another 3D generation fleet. Four other regimes need
fresh, source-bound preparation/common-constructor runs. Detailed source/path
leakage, denominator, collision/drop and runtime blockers are enumerated in the
protocol and the machine-readable 250-cell source map. No external credential
is newly required for this bounded audit.
