# Task 22 — Figure Rendering, Readability, and PDF Certification

**Priority:** P0 final  
**Suggested owner:** visualization / paper-quality engineer who did not author the figures  
**Depends on:** Task 20 and a buildable paper commit  
**Blocks:** Task 21 and submission

## Objective

Mechanically certify that every figure and the complete paper remain readable at the exact size used in the ICRA PDF. This task is not a style review. It is a reproducible pass/fail gate for clipping, overlap, tiny fonts, renderer-sensitive glyphs, excessive down-scaling, grayscale ambiguity, missing font embedding, page count, and first-page placement.

## Why this task is load-bearing

A strong method can look weak when the first page is dense or the labels are unreadable. Conversely, a beautiful enlarged source figure may become illegible after `\columnwidth` or `\textwidth` scaling. The only authoritative artifact is the rendered submission PDF at publication size. The phone-scan intuition, task-conditioned certificate, and evidence ladder must be understandable without zooming.

## Required implementation

Create:

- `tools/validate_paper_artifacts.py`
- `configs/paper/figure_qa.yaml`
- `tests/test_paper_artifact_validator.py`
- `tests/fixtures/paper_qa/` with one valid and at least three deliberately broken PDFs
- `paper_artifacts/<paper_commit>/qa/render_manifest.json`
- `paper_artifacts/<paper_commit>/qa/preflight.json`
- `paper_artifacts/<paper_commit>/qa/parity/summary.json`
- rendered PNGs for every page and every standalone figure
- `outputs/validation/task_22/report.json`

The validator must record the paper commit, TeX engine and version, source hashes, page count, media box, embedded fonts, LaTeX warnings, effective figure scale, declared minimum font size, renderer parity status, and hashes of all rendered pages.

## Fixed typography and drawing constraints

These are hard defaults. A deviation requires a visible exception in `render_manifest.json` and manual approval.

- Minimum in-figure text after scaling: **7.5 pt**.
- Main panel/stage labels after scaling: **8.0 pt** or larger.
- Captions use the conference template size; do not shrink captions manually.
- Minimum structural line width after scaling: **0.50 pt**.
- Main arrows and causal links after scaling: **0.65 pt** or larger.
- Do not use `\resizebox` below an effective scale of **0.88** without an explicit readability exception.
- No meaning may be encoded by color alone; labels, shape, line style, or ordering must preserve the distinction in grayscale.
- Red/green success and failure states must remain distinguishable under common color-vision-deficiency simulation.
- No raster screenshot containing text is accepted when vector text can be used.
- The first-page graphical abstract must appear on page 1 and remain within one column unless the final page design explicitly reserves full width.

## Exact validation ladder

Run from a clean checkout of the paper repository.

### Gate 0 — clean official build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error conference.tex
```

Pass conditions:

- exit code 0;
- no undefined control sequence, missing file, multiply defined label, or unresolved reference;
- no `Overfull \\hbox` or `Overfull \\vbox` warning;
- page count satisfies the currently verified ICRA rule including references;
- all new experiment cells remain visibly `TBD` before result freeze and none remain after freeze.

### Gate 1 — PDF preflight and font embedding

```bash
mkdir -p paper_artifacts/qa
python /home/oai/skills/pdfs/scripts/pdf_preflight.py \
  conference.pdf --json > paper_artifacts/qa/preflight.json
pdffonts conference.pdf > paper_artifacts/qa/pdffonts.txt
```

Pass conditions:

- PDF opens without repair, is unencrypted, and is not image-only;
- every used font is embedded or the official template provides a documented exception;
- no malformed glyphs or substituted square boxes appear.

### Gate 2 — publication-scale rendering

```bash
python /home/oai/skills/pdfs/scripts/render_pdf.py \
  conference.pdf --out_dir paper_artifacts/qa/render_pdfium \
  --dpi 220 --engine pdfium
python /home/oai/skills/pdfs/scripts/render_pdf.py \
  conference.pdf --out_dir paper_artifacts/qa/render_pdftoppm \
  --dpi 220 --engine pdftoppm
```

Pass conditions:

- every page is inspected at 100% display scale;
- no clipped captions, labels, legends, tables, or page-edge content;
- all first-page figure text is readable without zooming;
- every multi-panel figure preserves a clear visual hierarchy.

### Gate 3 — renderer parity

```bash
python /home/oai/skills/pdfs/scripts/renderer_parity.py \
  conference.pdf --out_dir paper_artifacts/qa/parity --dpi 220
```

Pass conditions:

- no missing text, shifted labels, broken transparency, or materially different clipping across renderers;
- any anti-aliasing-only pixel difference is documented and automatically separated from structural differences.

### Gate 4 — automated artifact checks

```bash
python tools/validate_paper_artifacts.py \
  --pdf conference.pdf \
  --config configs/paper/figure_qa.yaml \
  --out paper_artifacts/qa/render_manifest.json
pytest -q tests/test_paper_artifact_validator.py
```

The automated validator must at minimum check page count, media size, missing/embedded fonts, LaTeX log errors, overfull boxes, required figure labels, required rendered-page files, file hashes, and declared effective font/line sizes from figure metadata.

### Gate 5 — deliberate negative tests

The fixture suite must contain and correctly reject:

1. a figure with a 6 pt label after scaling;
2. a label or legend clipped outside its bounding box;
3. a non-embedded or missing font/glyph;
4. a red/green-only distinction with no redundant encoding;
5. a `\resizebox` scale below the threshold without an exception.

### Gate 6 — physical-size audit

Print the first page and all full-width figures on letter or A4 at 100% scale, or generate a calibrated on-screen view matching physical dimensions. A second reviewer must confirm that titles, axes, legends, and certificate states are readable at normal reading distance. Record reviewer, date, medium, and findings in the report.

### Gate 7 — handoff validation

```bash
python plan/scripts/validate_task_result.py \
  outputs/validation/task_22/report.json --check-artifacts
```

## Figure-specific assertions

The report must explicitly confirm:

- [ ] The page-1 graphical abstract states the deployment question and shows phone scan → twin → virtual policy → accept/repair/abstain.
- [ ] The detailed pipeline separates metric reconstruction, physics, appearance, policy execution, and certification.
- [ ] The certificate figure distinguishes task-relevant and irrelevant scene regions.
- [ ] The evidence ladder keeps ScanNet++ scale, oracle causality, paired predictivity, and phone deployment as separate protocols.
- [ ] All figure captions state the scientific claim rather than narrating every shape.
- [ ] Qualitative images are selected by a frozen rule and do not hide failures.

## Acceptance criteria

- [ ] Official paper build passes Gates 0–7.
- [ ] No figure uses text below the declared minimum at final scale.
- [ ] Both PDF renderers agree structurally.
- [ ] All pages and standalone figures have hash-traceable QA renders.
- [ ] A second reviewer signs the physical-size audit.
- [ ] Any exception is visible, justified, and linked to the exact figure source line.

## Stop conditions

Stop and return `blocked` when the official class, page-limit rule, or final figure placement is not frozen. Stop and return `fail` when a label can only be made readable by exceeding the page budget; the layout or content must be redesigned rather than globally shrunk.

## Paper artifact unlocked

A certified first page and figure set whose readability can be reproduced by another agent from a clean checkout, plus a defensible final PDF-quality statement in the submission audit.