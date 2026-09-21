# PhiRIE project page

Research page with three interactive WebGPU experiences, 45 visualization views,
six recorded videos, three paper-figure exports, and a source manifest.

## Run and build

Node.js 22 is supported. All npm versions are pinned in `package-lock.json`.

```bash
cd website
npm ci
npm run dev
npm run build
npm run preview -- --port 4173
```

The default local build uses `/PhiRIE/`; the Pages workflow obtains the actual
base path from GitHub, including private Pages domains. The current deployment is
https://redesigned-tribble-o8eej69.pages.github.io/ (organization-only).
Open `http://127.0.0.1:4173/PhiRIE/` when using the preview command.
Override the deployment path with `SITE_BASE=/another/path/ npm run build`.
Serve the build over HTTPS (or localhost) with normal binary MIME types. There are no API keys, inference services, Python
servers, or external font/CDN requests at runtime. Assets are loaded on demand;
the initial page does not load the rendering stack or 3D models.

The `project-page` workflow builds and deploys this folder from `main` using
GitHub Pages. Author order and affiliations are recorded in `../AUTHORS.json`;
the manuscript PDF is the original snapshot and has not been rewritten.

## Three experiences

1. **Scene editing:** native WGSL anisotropic Gaussian rendering of the captured
   `c50d2d1d42` desk. Original / removed / completed / reassembled states use
   saved target membership, the actual 476-Gaussian fill slice, and registered
   TRELLIS appearance. Translation and rotation update the replacement in 3D.
   The browser asset is a deterministic spatial crop/subsample with SH0 color.
   Background kernels are widened 1.4× to compensate for subsampling. This is
   a web preview, not the full-resolution research renderer. Registration uses
   GT assistance; this interaction does not run object discovery or generation.
2. **Physics sandbox:** native Three.js WebGPU rendering, Rapier WASM dynamics,
   and three actual SAM3D meshes (cup, headphones, keyboard). Metric meshes are
   decimated and colored from nearest Gaussian SH0 samples. Collisions use one
   convex hull per object, so holes/cavities and concavities are approximated.
   Mass, friction, throwing and projectile speed are interactive. Browser
   dynamics and estimated parameters are not calibrated physical predictions.
3. **Robot replay:** original robot visual geometry and SAM3D cup collision
   geometry positioned from **213 recorded MuJoCo integration states**. Controls
   select original states without interpolation, show the tool trajectory, and
   jump to the first recorded contact. The added support plane is contextual
   browser geometry at the recorded support height. The source controller is a
   scripted physical cup push and retract; task success is unmeasured.

Hardware adapters render directly to the WebGPU canvas. Software adapters use
the same WebGPU shaders, then read the render target into a visible Canvas2D
surface to avoid Linux software swapchain presentation failures. That path is
explicitly labeled **software adapter**. There is no silent WebGL substitution.
An unavailable adapter shows a support message and links to recorded videos.

Reference APIs: [Three.js WebGPURenderer](https://threejs.org/docs/pages/WebGPURenderer.html),
[Rapier JavaScript](https://rapier.rs/docs/user_guides/javascript/getting_started_js/).

## Evidence and reproduction

[public/provenance.json](public/provenance.json) records source paths, SHA-256
identities, conversion operations, and output checksums for 61 media/model files.
[public/gallery.json](public/gallery.json) supplies per-view captions. Original
figures and results remain untouched. No training, model inference, Slurm job,
or new research experiment is part of website construction.

The paper source uses the earlier name **SimAnyRoom**. The existing PDF is
served unchanged as a **manuscript snapshot**, not rebuilt from current TeX.
The qualitative and harmonizer figures come from the paper figure collection;
the uncertainty plot comes directly from the manuscript repository. A damaged
legacy JPEG teaser was excluded. The gallery retains source reconstruction and
completion artifacts; it does not fabricate missing views or claim whole-room
collision accuracy, learned-policy success, or physical-robot transfer.

To rebuild derived research assets, run the exporter in the original consolidated
PhiRIE workspace, which contains the source research data:

```bash
envs/control/.venv/bin/python website/scripts/export_assets.py
# Or one export group:
envs/control/.venv/bin/python website/scripts/export_assets.py --only robot
```

The exporter requires NumPy, Pillow, plyfile, SciPy, trimesh,
fast-simplification, and MuJoCo. Original absolute XML paths are relocated in
memory. Fonts are self-hosted DM Sans and DM Serif Display from Google Fonts;
their SIL Open Font Licenses are included in `public/fonts/`.

## Browser verification

```bash
npx playwright install --with-deps chromium
# In another terminal: npm run preview -- --port 4173
BASE_URL=http://127.0.0.1:4173/PhiRIE/ npm test
```

The test launches Chromium with a software Vulkan/WebGPU adapter for CI.
It checks gallery filters/search/pagination/lightbox, comparison controls,
six videos, rendered changes from Gaussian edits, physical response to a
projectile, robot state playback, mobile overflow, unsupported WebGPU fallback,
and axe WCAG A/AA rules. Results and screenshots go to `test-results/` or
`OUTPUT_DIR`. A software-adapter pass is not a hardware performance measurement.

See [REPORT.md](REPORT.md) for this delivery's checks and remaining limits.
