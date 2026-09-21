# docs/

Index of the SimAny documentation.

| page | contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | top-level layout: what lives in `agents/`, `models/`, `robo/`, `interface/`, `run/` and how they fit together |
| [PIPELINE.md](PIPELINE.md) | the generation pipeline stage by stage: discover, assets, edit, render, eval |
| [ROBOT.md](ROBOT.md) | robot layer: MuJoCo env, Panda + Robotiq rig, photoreal observations, closed-loop pi0.5 evaluation |
| [DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) | where datasets, checkpoints and caches live, and the env vars that override the paths |
| [ENVIRONMENTS.md](ENVIRONMENTS.md) | why three python environments are unavoidable, plus other environment facts |
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | what the contribution actually is, with the audited numbers behind each claim |
| [BASELINES.md](BASELINES.md) | baselines and concurrent work: what we compare against, what we owe, what we cite |
| [GENERATOR_COMPARISON_CODEX_PROMPT.md](GENERATOR_COMPARISON_CODEX_PROMPT.md) | bounded execution prompt for missing SAM 3D Objects and TRELLIS.2 comparisons, reusing TRELLIS/ReconViaGen outputs; 32 objects, four backends, two concurrent single-GPU jobs and an eight-GPU-hour task limit |
| [PAPER_NOTES.md](PAPER_NOTES.md) | ICRA paper planning notes: title, story, experiment plan |
| [PAPER_REVISIONS.md](PAPER_REVISIONS.md) | revisions owed after reading the July-2026 concurrent work |
| [DEMO_STORYBOARD.md](DEMO_STORYBOARD.md) | shot-by-shot storyboard for the demo video (bilingual, mostly Chinese) |
| [related/](related/) | reference material on related work (fetched arXiv text; PDFs gitignored) |
| [paper/](paper/) | LaTeX source of the paper (root.tex extended/arXiv version with appendices, conference.tex 8-page ICRA cut); build with `paper/build.sh` |
