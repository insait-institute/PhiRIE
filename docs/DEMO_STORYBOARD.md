# SimAnyRoom Demo Storyboard

The authoritative production guide for the current paper is:

## [`plan/icra2027/10_demo/README.md`](../plan/icra2027/10_demo/README.md)

It specifies the 80–95 second hero film, 30 second teaser, 8 second loop, source manifest, rendering/codec requirements, live-presentation fallback, automated QA, and the current agentic story:

```text
capture
→ metric room reconstruction
→ multiple object proposals
→ evidence-based selection or retry
→ shared physics/appearance object identity
→ full-room simulation
→ frozen-policy manipulation
→ real-capture scale
```

The existing `interface/demo_movie.py` remains a useful legacy renderer for scan, discovery, interaction, replay, and assembly shots. New work should preserve it and implement the current film in `interface/demo_agentic.py` as required by the ICRA 2027 demo task.

Do not use the old `SimAny — from a raw scan to an interactive world` title card or the old hard-coded result card for the final submission. All final displayed values must be loaded from the same experiment freeze used by the paper.