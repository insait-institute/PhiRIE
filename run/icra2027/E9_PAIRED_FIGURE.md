# Source-bound paired-contrast figure

With the existing authenticated full-cohort `agentic_paired_uncertainty` source,
set `agentic_paired_uncertainty_figure: true` in a new frozen paper config.
`robo.eval.paper_pipeline` then produces PDF, editable SVG, PNG and a figure
manifest under `generated_figures/`, copying the same bytes to the paper's
`figures/` directory. Their hashes enter `paper_table_provenance.json`.

The four panels show the existing coverage, F1@20, CD and stability-probe
contrasts. All declared contrasts remain visible; the original paired-job,
planned-job and supported-scene counts accompany each point. Conditional
geometry and construction-probe support are labeled separately from coverage
over all planned jobs. Null estimates and intervals remain explicit; the
renderer neither bootstraps nor refits. No sign is flipped to imply improvement.
Pointwise descriptive intervals do not establish independent physical validity
or a population-wide conditional-quality gain.

The normal paired table is still generated, with exact numeric fields in its
claim ledger. The figure manifest maps every plotted point and interval to that
same source row and field. A figure without the full authenticated statistical
source is refused. Rendering failure aborts the atomic paper publication;
there is no placeholder-result fallback.

```bash
PYTHONPATH=. python -m pytest -q tests/test_paper_paired_figure.py \
  tests/test_paper_paired_uncertainty.py
```

Synthetic fixture figures are test artifacts only and must never enter the
paper or presentation. Actual figure generation waits for all construction
seals, independent evaluation, and the canonical paired-statistics audit.
