# Prospective DROID CPU alignment in the paper audit

The optional `engineering_appendix.droid_cpu_alignment` source extends
`robo.eval.paper_pipeline` and consumes the existing canonical
`robo.eval.real_world_records.summarize_prospective` publication. It does not
refit, resample, rerun reconstruction, or define another metric implementation.

```yaml
engineering_appendix:
  droid_cpu_alignment:
    path: /absolute/freeze/full_cpu_summary/workspaces.json
    sha256: EXACT_WORKSPACES_SHA256
    completion_audit:
      path: /absolute/freeze/full_cpu_completion_audit.json
      sha256: EXACT_AUDIT_SHA256
```

The formatter requires the complete original ten-workspace roster, ten terminal
result records, the original config/E0/source binding, and all five canonical
JSON/CSV publication hashes. It checks each row against its authenticated result,
frame plan, and independent alignment split. The existing aggregate is replayed
only as an arithmetic integrity check. A failed construction remains a table row
with its original missing measurements. Very large finite errors are retained;
neither accuracy nor the alignment pass gate selects displayed workspaces.

The generated `droid_cpu_alignment_engineering.tex` displays every workspace's
CPU state and evaluated/planned reference frames before center RMS in centimeters
and median rotation residual in degrees. Two IPRL workspaces retain their original
roster order. All quantitative cells enter the normal claim ledger with exact
JSON field paths; machine-readable workspace rows remain unchanged in copied
sources. `paper_ready` and complete E7 flags stay false. CPU-stage completion
does not establish Gaussian reconstruction, a complete simulator, or physical
policy success. Later Gaussian results require their own authenticated source.

Run through the existing pipeline with an immutable configuration and new output
freeze. No paper table or claim-decision file should be edited by hand.

```bash
PYTHONPATH=. python -m pytest -q tests/test_paper_droid_cpu.py \
  tests/test_paper_engineering_appendix.py tests/test_paper_pipeline_audit.py
```

Negative cases cover omitted/reordered/duplicate workspaces, invented geometry
or simulator completion, hidden construction failures, altered frame counts,
TRAIN/reference leakage, changed source/config/result bytes, partial completion,
and invalid aggregate or claim flags. An all-failed cohort is valid evidence of
failure, not an implementation error or permission to omit its rows.
