# E9 — Paper Table Freeze, Claim Audit, and Submission Handoff

**Priority:** P0, final integration  
**Paper output:** all quantitative tables and final claim-safe text  
**Depends on:** E1–E8 as available

## Goal

Generate every paper table from immutable experiment artifacts, copy only generated outputs into the `SimAnyRoom` paper repository, audit every numeric claim, and produce one submission freeze that can be regenerated from a clean checkout.

This task must not improve results, rerun only favorable seeds, or manually fill LaTeX cells.

## Read first

- `robo/eval/paper_pipeline.py`
- `robo/eval/paper_tables.py`
- `configs/experiments/paper_pipeline.template.yaml`
- `agents/eval/make_paper_tables.py`
- `agents/eval/bootstrap.py`
- `agents/eval/power.py`
- all `STATUS.md` files under `plan/icra2027/`
- paper repository: `RunyiYang/SimAnyRoom`

## Input contract

E9 consumes only frozen products:

```text
construction scene records
fidelity manifest and metric JSON/CSV
agentic ablation JSON/CSV
harness config, reset bank, and ledger
audit feature/label CSV and LOSO predictions
Harmonizer visual manifest and metrics
real-world workspace and paired-trial CSVs
root freeze manifest
```

If a task has not passed its claim gate, E9 must remove or narrow the corresponding text. It may not retain the claim with a `TBD` result in a submission-ready freeze.

## Required config

Copy:

```text
configs/experiments/paper_pipeline.template.yaml
    -> configs/experiments/icra2027/paper_pipeline.yaml
```

Add the E3 agentic-ablation input explicitly if `paper_pipeline.py` does not yet support it. Extend the canonical pipeline rather than writing a one-off table script.

The resolved config must contain only paths inside the selected freeze or paths whose hashes are recorded by E0.

## Canonical generation command

With the paper repository checked out beside the code repository:

```bash
FREEZE=<freeze_id>
PAPER_ROOT=../SimAnyRoom

python -m robo.eval.paper_pipeline \
  --config configs/experiments/icra2027/paper_pipeline.yaml \
  --out outputs/icra2027/${FREEZE}/paper_tables \
  --paper-root ${PAPER_ROOT}
```

Expected root product:

```text
outputs/icra2027/<freeze_id>/paper_tables/paper_table_provenance.json
```

Every generated table must list its source files and hashes in that provenance object.

## Required table mapping

| Paper table | Frozen producer/input |
|---|---|
| Automatic construction | `robo.eval.construction_metrics` |
| Appearance and geometry fidelity | `robo.eval.fidelity_metrics` plus fixed mixed-unit formatter |
| Agentic construction ablation | E3 `agentic_ablation.json/csv` |
| Closed-loop manipulation | `robo.eval.main_table` |
| Task-local support | `robo.eval.audit_loso` + `robo.eval.audit_metrics` |
| Harmonizer diagnostics, extended | `robo.eval.harmony_visual_metrics` |
| Real-world construction/trials | `robo.eval.real_world_table` |

Do not let a generated script drop a planned method row because data is missing. Missing values remain explicit until the claim is removed or the experiment is completed.

## Claim ledger

Create:

```text
outputs/icra2027/<freeze_id>/paper_tables/claim_ledger.csv
```

Required columns:

```text
claim_id
paper_file
line_hint
claim_text
claim_type              # measured / derived / qualitative / limitation
source_artifact
source_field
value_in_text
value_in_source
status                  # exact / rounded / unsupported / stale / remove
owner
```

At minimum audit:

- scene/object counts;
- yield, F1, stability, runtime;
- PSNR/SSIM/LPIPS/CD values;
- collapse counts and oracle gap;
- agentic coverage/fidelity statements;
- manipulation episode counts and rates;
- task-support metrics;
- real-room/task/trial counts and sim-real gaps;
- any “all”, “first”, “fully automatic”, “any room”, or speedup wording.

A script should fail when a measured numeric claim has no source artifact or differs beyond its allowed rounding tolerance.

## Abstract claim gate

The abstract may contain a result only when:

- the relevant task `STATUS.md` marks its gate passed;
- the final aggregate is frozen;
- sample/episode/trial counts are stated or inferable nearby;
- the claim does not rely on a proxy mislabeled as real evidence.

Required decisions:

```text
AGENTIC_TITLE_CLAIM: keep / narrow / remove
MANIPULATION_CLAIM: keep / narrow / remove
HARMONIZER_MAIN_CLAIM: keep / move-to-supp / remove
TASK_SUPPORT_CLAIM: keep / move-to-supp / remove
SIM_REAL_CLAIM: keep / remove
```

Write these decisions to `claim_decisions.yaml` and update the paper text accordingly.

## Paper repository edits

Allowed automatic changes:

- generated table `.tex` files;
- generated plots/figures;
- result sentences containing frozen values;
- abstract values;
- limitations/claim removal required by gates.

Do not rewrite method text during the freeze unless an experiment reveals that the implemented method differs from the paper. Such a mismatch must be documented.

Before commit:

```bash
cd ../SimAnyRoom
./build.sh conference.tex
```

Then inspect every rendered page at 200 dpi or higher.

## Submission QA

The final paper must satisfy:

- at most 8 pages including references;
- no `TBD`, `TODO`, red text, placeholder captions, or empty claimed cells;
- no overfull boxes;
- all references and citations resolved;
- embedded fonts;
- readable tables at 100% zoom;
- main teaser uses real experiment imagery, not placeholders;
- figures and tables use the same freeze ID;
- anonymous metadata and author block;
- raw URLs absent unless required by format;
- all reported failure/coverage denominators consistent.

## Reproducibility bundle

Produce:

```text
outputs/icra2027/<freeze_id>/paper_tables/
  paper_table_provenance.json
  claim_ledger.csv
  claim_decisions.yaml
  generated_tables/
  generated_figures/
  bootstrap/
  paper_commit.txt
  paper_build_log.txt
  pdfinfo.txt
  fonts.txt
  page_renders/
  submission_audit.json
```

Also create a lightweight manifest bundle containing configs, hashes, and tables but no restricted datasets/checkpoints.

## Tests

- every generated table has a provenance entry;
- no source path resolves outside the freeze without a recorded hash;
- stale paper values are detected;
- rounding checks work for percentages and decimals;
- missing experiment gates remove claims rather than leave placeholders;
- paper build fails above 8 pages;
- grep for `TBD|TODO|placeholder` is empty in submission mode;
- all expected table labels occur exactly once;
- same freeze ID appears in the table and demo result manifests.

## Fast smoke

Run `paper_pipeline.py` on template/synthetic inputs into `/tmp`, copy into a temporary paper checkout, and compile. The smoke must not modify the real paper repository.

## Acceptance criteria

- [x] One command regenerates all available tables.
- [x] Every numeric claim has a source and rounding audit.
- [x] Failed task gates cause explicit claim edits.
- [x] Paper compiles to at most 8 pages with no overfull boxes/placeholders.
- [x] PDF page renders are visually inspected.
- [x] Paper commit and experiment freeze are cross-referenced.
- [x] The available-paper artifact bundle can be audited without cluster access.

The six checked publication criteria were verified for producer
`fa5a777ba7c56b9e72243da6ad3dca657cb354ed`, publication
`20260906-6dcb0e7-v2`, and pushed paper
`18def2748381f1a7105e2051a5b80e07d82334e0`. Reproduction uses
`python -m robo.eval.paper_pipeline --config configs/experiments/icra2027/paper_full_controller_concise.yaml`
with the output and paper paths recorded in `STATUS.md`. The publication receipt
binds all 40 generated artifacts; `audit/paper_qa_full_controller.json` in the
paper binds both eight-page PDFs and all 16 inspected page renders. This closes
available-table publication, not the final scientific submission. The portable
complete artifact bundle and common final experiment/demo freeze remain open;
`SUBMISSION_FREEZE=FAIL`.

## Handoff

Create `STATUS.md` with the experiment freeze ID, code and paper commits, table provenance hash, claim decisions, PDF path/page count, all remaining limitations, and `SUBMISSION_FREEZE=PASS|FAIL`.

Latest available-result publication supersedes the draft above: producer93c9566, freeze20260906-8cae8a0-v1, pushed paper83c97b1. Nine generated LaTeX tables plus22JSON/CSV sources, four figure assets and four audit files;8+8page/16-render/font QA passes. Portable337-field/22-decision/283-member offline audit passes; exact reproduction and hashes are in STATUS.md and PORTABLE_BUNDLE.md. This closes publication of currently admissible evidence, not missing scientific experiments or the final demo; SUBMISSION_FREEZE=FAIL.
