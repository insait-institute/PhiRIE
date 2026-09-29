# Pipeline blocks

`physicalview blocks --json` exposes all ten tool inventories and their development
branches. Each subfolder describes the matching block's inputs, outputs and calls.
Implementation remains in the installable `physicalview` package and pinned external
backend, preserving existing Python module entry points.

`physicalview plan BLOCK` prints JSON without launching compute. Composed construction
plans require backend source because builders check their stage modules.
`physicalview run BLOCK` writes job specs, logs and results, executes stages in order,
and fails if a stage fails or its declared outputs are missing. Local execution is the
default; `--where slurm` uses GPU targets in your local config.

Use `--out` for isolated results and `--objects 1,2` for object-scoped stages.
Direct module calls (viewer, datasets, reconstruction, paper) accept additional backend
arguments after `--`; composed stages use structured CLI flags.

`full` composes discovery, generation, registration, physical annotation, report, export
and task generation. It requires prepared posed observations and stage environments.
Reconstruction, background editing, viewer launch and robot execution are separate calls.

Shared infrastructure: [runtime](runtime/README.md) and [release](release/README.md).
