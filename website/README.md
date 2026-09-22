# PhiRIE project page

[Live page](https://insait-institute.github.io/PhiRIE/) · [Overview video](https://youtu.be/3-YdcBh6Tbw) · [PhiView code](https://github.com/RunyiYang/PhysicalView)

The page includes the overview video directly below the TL;DR, nine PhiView
recordings, a four-setting mass/friction comparison, an image gallery, three
WebGPU playgrounds, and the development version tree and release plan.

## Run locally

Use Node.js 22:

```bash
cd website
npm ci
npm run dev
# http://127.0.0.1:5173/PhiRIE/
```

```bash
npm run format:check
npm run build
npm run preview -- --port 4173
# http://127.0.0.1:4173/PhiRIE/
```

`SITE_BASE` selects the deployment path. The default is `/PhiRIE/`; set
`SITE_BASE=/` for an origin-root deployment. GitHub Pages builds from `main`.

## Content

- `index.html`: page structure, featured video, comparison, and version tree.
- `src/main.js`: gallery, demo picker, and video source controls.
- `public/demos.json`: titles, descriptions, durations, and files for demos 01–09.
- `public/media/`: videos, images, posters, and downloadable screenshots.
- `public/gallery.json`: gallery images and captions.
- `src/lab.js`, `src/gaussian.js`, `src/physics.js`, `src/robot.js`: playgrounds.

The overview embeds YouTube and includes a direct MP4 playback option.
Recordings use HTML video controls and load on demand. WebGPU assets load
only when a playground is launched.

## Browser checks

```bash
npx playwright install --with-deps chromium
# Start the preview server in another terminal.
BASE_URL=http://127.0.0.1:4173/PhiRIE/ npm test
```

Checks cover the header and contact links, video placement and playback, all
nine recordings, parameter comparison, version tree, gallery controls, mobile
layout, accessibility, and the existing WebGPU interactions.

Original recordings remain in the research workspace. Website videos are copied
or compressed for web playback; figures use aspect-preserving image conversion.
