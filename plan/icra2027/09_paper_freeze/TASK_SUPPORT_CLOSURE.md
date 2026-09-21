# Optional E6 missing-evidence engineering closure

`robo.eval.paper_task_support_closure.validate_closure(qa_path,
expected_sha256=...)` is a read-only input adapter for `paper_pipeline`. It returns
one machine-readable engineering count row, provenance, null metrics, and a
failed predictive claim gate. It does not write LaTeX or publish paper artifacts.

The accepted original producer is clean `152446d6874155e3a2115759c96e9a369f59236a`.
The caller must bind the closure QA SHA in its own paper E0. Validation checks
both feature and evaluation E0 contracts and metadata, the original 72-query
feature seal, all member/source hashes, the exact 18 public grounding gate
references and source/config identities, and all exact adapted query bundles.
Original public evidence was already validated before feature generation; the
formatter reuses that hash-authenticated proof without replaying geometry,
rendering, discovery, or model jobs. Canonical feature/label/bridge implementation
bytes must equal the pinned original producer. No persistent validation cache is
created, so a subsequent invocation rechecks all input bytes.

Canonical `audit_labels.label_record` verifies the missing-required-evidence
labels. The join preserves original feature columns, failed constructors, and
all 72 unique query keys. Six all-invalid folds must remain unestimable. Explicit
null `NOT_RUN` reference/baseline records are required; added predictions, metric
outputs, changed labels/counts/source/seals, or positive claim flags fail closed.
These are required-evidence validity labels, not measured policy outcomes or
independently measured reference errors. No reference vault is read.

Authoritative input:

- QA: `outputs/icra2027/20260906-047bd5d-v1/validity_independent_qa.json`,
  SHA256 `4207c4ac08c97d06faebd69baa9b02c392b4b1b1c26df0ebb7af443e81fdad89`.
- Feature freeze: `20260906-0e7579a-v2`; seal
  `b1bad494cbfd9e04700d678a580dd5b713a2ef57be8c691f6b9170121a00eaa2`.
- Join: `20260906-047bd5d-v1/audit/validity_closure/label_join_manifest.json`,
  SHA256 `e5f47251f98a07171ce045519d9c1fe39271ae496f6262055095b1943c0acdbb`.

Smoke (use an approved repository-local basetemp):

```bash
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_paper_task_support_closure.py --basetemp=outputs/e9-closure-tests
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/icra2027/preflight.sh --smoke
```

The engineering count row reports 72 queries, 16 failed-constructor queries,
72 invalid queries, and 0 estimable folds out of 6. `status=PASS` means artifact
integrity only; `claim_gate=FAIL`, `loso_status=NOT_RUN`, and all metric cells are
null. It cannot fill the predictive task-support comparison table or enable a
selective-risk improvement claim. Central E9 integration is owned separately.
