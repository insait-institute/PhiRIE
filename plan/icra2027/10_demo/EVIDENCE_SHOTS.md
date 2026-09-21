# Canonical construction evidence shots

`configs/demo/icra2027_evidence.yaml` pins the already completed pilot's
`38d58a7a31/obj_1000` decisions: A0 TRELLIS, A2 ReconViaGen selection, and A3
signed-source-up registration retry. This object was explicitly declared before
rendering. No result-driven replacement or new generation is performed.

The existing twelve-second engineering renderer now supports authenticated
`evidence_comparison` sources. Initial selection displays both raw meshes after
their exact archived registration. Retry displays the original ReconViaGen mesh
under the parent and newly executed registration transforms. Both panels use one
common bounds center, camera and scale. Complete triangle meshes are rendered
by MuJoCo; no simplification, reconstruction, new alignment or simulation is run.
Numeric text is restricted to the cited construction residual, support overlap
and archived settle verdict. These are object evidence, not paper results.

Both the chosen initial and retried proposal remain **UNSUPPORTED**. The retry
improves its archived settle verdict while its support overlap decreases; both
facts remain visible. This does not establish independent physical stability,
policy success, generator superiority or the full agentic claim.

The source manifest retains the original construction freeze
`20260905-33bd974-v1`, even though selected records are copied under the separate
evaluation aggregate `20260905-1577027-v1`. It verifies all selected artifact
hashes, evidence-to-mesh binding, probe/evidence agreement, registration matrices,
proposal identities, retry parent/action, and genuinely changed transforms.
No evaluation metric/reference enters the renderer. The original training-only
RGBA observation is used in the opening shot. The final shot explicitly states
that manipulation footage is unavailable.

## Reproduce

From the committed worktree, with the repository's existing Python interpreter:

```bash
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
export LP_NUM_THREADS=4
export SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python
bash run/demo/preflight_icra2027.sh --config configs/demo/icra2027_evidence.yaml
bash run/demo/render_icra2027.sh --config configs/demo/icra2027_evidence.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-61d8ba4-v3/demo
bash run/demo/assemble_icra2027.sh \
  --demo-dir /group/worldcept/code/SimAny/outputs/icra2027/20260905-61d8ba4-v3/demo
```

The output path is immutable; a rerun requires a new config/freeze. The OSMesa
setup unpacks a Debian shared library locally and does not alter the Python
model environments. An available EGL context is also compatible; renderer
failures propagate instead of being replaced by a fabricated image.

Output remains the honest 12 s / 960x540 / 15 fps engineering draft, plus two
6 s / 1920x1080 / 30 fps static evidence comparison clips and their PNG posters
in `demo/evidence_clips/`. The comparison images intentionally hold still for
reading; they are not presented as a physical trajectory. Every nested clip
and poster is included in the existing package hash manifest.

## Remaining final-demo gates

Final 90/30/8 outputs remain NOT_RUN. Required inputs still include an eligible
hero with at least four accepted objects, clean background and valid full-room
collision; one genuine synchronized MuJoCo/photoreal policy episode; Option C
preservation-certified frames if shown; real-capture construction/alignment;
and E9's final claim-valid common freeze. The current single-scene pilot has
only two A4 accepted objects and does not meet the hero-room rule.

The existing five v18 `scripted_sinusoid` raster recordings are task failures,
not learned-policy success and not synchronized photoreal pairs. They cannot
fill those missing hero shots without explicit engineering-only labeling.
