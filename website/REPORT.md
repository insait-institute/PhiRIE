# PhiRIE organization project page

Prepared 2026-09-21 for `insait-institute/PhiRIE`.

- Nine authors in the supplied order, with the corrected name **Danda Paudel**,
  three affiliations, and Kunyu Peng marked as corresponding author. Authors
  occupy one row; smaller screens can scroll that row horizontally.
- The full paper title is the main large heading. The introductory headline and
  description appear together in a separate highlighted TL;DR box.
- Complete source and all 61 research media/model assets retained with matching
  provenance checksums; 45 visualization views, six videos, three paper figures.
- Three WebGPU experiences: Gaussian scene editing, rigid-body interactions,
  and recorded robot motion replay.
- The `project-page` workflow reads GitHub's Pages base path and deploys from `main`.
  Pages is public at https://insait-institute.github.io/PhiRIE/; the source
  repository retains its existing access settings.

All 12 local browser checks passed with zero browser errors and zero detected axe
WCAG A/AA violations, including all WebGPU interactions, media playback, mobile
layout, and unsupported-WebGPU fallback. The production build and formatting pass;
npm audit reports zero known vulnerabilities. The browser adapter is Chromium
SwiftShader software WebGPU. Hardware GPU performance, Safari, and Firefox are not
measured. The original construction, collider, and scripted replay limitations in
README.md and the page captions remain applicable.

The manuscript PDF remains an unchanged historical snapshot; author metadata was
added to the project page, README, and citation file. The original site delivery
report is retained in REPORT-original.md for historical evidence only.

Exact run output and screenshots are retained in the release validation archive.

The public-page revision was rebuilt and rerun through all 12 browser checks.
The initial run passed the interactions but reported a software WebGPU device
loss; the complete rerun passed with zero browser errors. This page revision
does not retag or replace the published v2.0.1 release archives. Its validation
receipts are recorded separately under `results/project-page-public-20260921`
in the delivery workspace.
