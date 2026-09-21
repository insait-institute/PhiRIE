# Task 21 — Reproducibility Release and Submission Audit

**Priority:** P0 final  
**Suggested owner:** senior maintainer not responsible for headline experiments  
**Depends on:** all tasks  
**Blocks:** submission

## Objective
Perform a hostile reproducibility, anonymity, licensing, factuality, and page-budget audit. The audit should attempt to falsify every headline claim before reviewers do.

## Outputs
- `release/REPRODUCE.md`
- `release/MODEL_DATA_LICENSES.md`
- `release/check_release.py`
- `docs/SUBMISSION_AUDIT.md`
- final frozen experiment index

## Audit checklist
- paper compiles under the official ICRA class and fits the current limit including references;
- all citations, titles, release statuses, and reported prior numbers are verified from primary sources;
- no author, cluster path, private token, institution, or identifying metadata leaks;
- every table cell is provenance-linked and sample counts agree across text/table/figure;
- paired comparisons share frozen fields and include failures/coverage;
- phone claim is supported by actual phone rooms, not only ScanNet++;
- full-room collision is used in headline policy runs;
- partial baselines are labeled and licenses permit use/release;
- no main claim depends on a single favourable scene, task, or policy;
- seed/task/policy selection occurred before final outcomes;
- scripts work from a clean checkout with documented external weights/data.

## Required adversarial analyses
1. Remove each task/policy/room and recompute conclusions.
2. Compare pooled and within-task correlations.
3. Inspect all certificate false accepts.
4. Search paper for unsupported “first,” “general,” “any,” and causal wording.
5. Rebuild headline table from raw manifests on a second machine/account.

## Acceptance criteria
- [ ] Audit has no unresolved P0 finding.
- [ ] Known limitations are explicit and numerically scoped.
- [ ] Final code/data release map is complete, even where third-party licenses require download instructions instead of redistribution.
- [ ] Submission PDF/video pass format and anonymity checks.

## Paper artifact unlocked
A defensible final submission rather than an impressive but fragile systems demo.
