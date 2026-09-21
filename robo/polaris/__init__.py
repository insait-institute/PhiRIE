"""robo.polaris: SimAny <-> PolaRiS (arXiv:2512.16881) adapter layer, Task 07.

STATUS (2026-08-16): STATIC EXTRACTION ONLY. See docs/POLARIS_INTEGRATION.md
for the full story. Short version: Isaac Sim 5.1.0's camera-enabled headless
launch segfaults on this cluster (NVIDIA driver 595.x branch is a known-bad
combination with Isaac Sim's RTX renderer; confirmed on both the A6000 and an
H200 node here). Every real PolaRiS gym environment needs
`enable_cameras=True`, so no PolaRiS environment can actually reset/step on
this cluster today. docs/ICRA_RESEARCH_CONTRACT.md Decision 1 has been
updated to BLOCKED and `mujoco_paired` is the permanent (not placeholder)
path for Tasks 07-08.

What *is* here: `import_official.py` extracts the official PolaRiS task
definitions (robot, cameras, initial conditions, rubric, control contract)
directly from PolaRiS source/asset files (Python source + USD text + JSON),
without needing Isaac Sim to launch at all. This is real, checked-in-source
information, not a guess -- but it has NOT been cross-validated against a
live running official environment (that validation is exactly what's
blocked). Treat every extracted value as "what the source code says it
should be," not "what we observed it to be."

`export_simany.py`, `task_adapter.py`, and `validate_pair.py` are
intentionally NOT implemented in this pass -- they would either need a
working official environment to compare against (crashes) or would have to
fabricate the comparison, which would misrepresent Task 07 as further along
than it is. See docs/POLARIS_INTEGRATION.md for the resume plan.
"""
