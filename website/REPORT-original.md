# PhiRIE project page delivery

Date: 2026-09-21 UTC. Project: PhiRIE v2.0.1.

## Delivered

- Responsive research website using existing paper figures and project results.
- Three working WebGPU experiences: 3D Gaussian scene editing, reconstructed-object
  physics sandbox, and recorded robot motion replay.
- 45 captioned visualization views with category filters, search, pagination,
  and a keyboard-accessible image dialog.
- Six complete recorded interaction videos and five matched appearance comparisons.
- Three paper-figure exports and the original manuscript PDF snapshot.
- SHA-256 provenance for 61 media/model assets. Original research artifacts remain
  unchanged. All fonts and runtime libraries are served locally with the site.

Site: <https://phirie.ranny-yang.chatgpt.site>. Initial access is owner-private.
Canonical source: `website/` on the PhiRIE repository's `main` branch. Deployment
uses a separate source mirror with only `main`; the v2.0.1 release tag is unchanged.

## Validation

Production browser run: **12/12 checks passed**, **0 failed**, **0 uncaught errors**.

| Check | Result |
|---|---|
| Initial page and deferred 3D downloads | PASS |
| Media and provenance links | PASS: 56 URLs |
| Gallery filter/search/pagination/dialog | PASS |
| Matched-view comparison controls | PASS |
| All six recorded videos and playback | PASS |
| Gaussian rendering, stages, translation, rotation, reset | PASS |
| Gravity, projectile response, object switching and throw | PASS |
| 213-state robot replay, contact jump and view controls | PASS |
| axe WCAG 2 A/AA and WCAG 2.1 AA audit | PASS: 0 detected violations |
| Mobile layout at 390 px | PASS: no horizontal overflow |
| Unavailable WebGPU fallback | PASS |
| Browser runtime | PASS: no exceptions or console errors |

Additional checks: production Vite build passed; all 61 provenance checksums
matched; npm audit reported **0 vulnerabilities** after updating Vite to 6.4.3.
The largest static asset is 24,386,112 bytes, below the 25 MiB hosting limit.

Commands:

```bash
npm ci
npm run build
npm run preview -- --port 4174
BASE_URL=http://127.0.0.1:4174 npm test
npm audit
```

Machine-readable validation and screenshots are retained in
`results/project-page/validation-production/` in the consolidated workspace.
The completed source, build archive, and deployment receipt are retained under
`results/project-page/`.

## Scope and remaining limits

- Rendering was verified in Chromium with the SwiftShader **software WebGPU**
  adapter. Physical GPU performance and Safari/Firefox compatibility are unmeasured.
  Software adapters read back WebGPU-rendered pixels for reliable presentation;
  hardware adapters use direct WebGPU canvas rendering.
- The Gaussian preview uses cropped/subsampled data, SH0 color, and widened
  background kernels. The browser physics uses approximate convex colliders,
  estimated parameters, and Rapier; it does not reproduce the MuJoCo experiment.
- Robot playback uses original MuJoCo states from a scripted cup push. It is not
  learned-policy inference or a claim of successful grasping.
- The supplied manuscript snapshot uses the earlier name SimAnyRoom and may
  precede current TeX. Existing figures are preserved, and a damaged legacy teaser
  JPEG was excluded. Authorship and publication identifiers were not invented.
- Construction and visualization examples retain their GT-assisted, training-view,
  synthetic-input, and recorded-state scope in captions and source records.
- The repository remains private. Public website access requires an explicit
  audience change; no existing sharing policy was broadened.

One non-failing upstream Rapier initialization deprecation warning is recorded
by the browser test. The production site does not use the Vite development server.
