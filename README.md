# ϕ-RIE project page

Source of https://insait-institute.github.io/PhiRIE/. This `website` branch is
independent of `main`, which carries only the released code.

## Edit and preview

Use Node.js 22:

```bash
git switch website
npm ci
npm run dev            # http://127.0.0.1:5173/PhiRIE/
npm run format:check
npm run build          # writes dist/
```

`index.html` holds the page structure (news, overview video, demos, method,
playground, results, paper, versions); `src/` the scripts and styles;
`public/media/` the images and recordings; `public/demos.json` and
`public/gallery.json` the demo and gallery metadata.

## Deploy

Every push to this branch runs the `project-page` workflow, which builds the
page and publishes it with GitHub Pages. The same workflow can be started
manually from the Actions tab.
