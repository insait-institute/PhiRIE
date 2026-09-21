# SimAny Paper-to-Code Status

This file maps every quantitative claim in the current ICRA skeleton to an
executable producer. A blank paper cell is allowed only because its frozen
experiment has not yet run, not because the metric or aggregation code is
missing.

## Main quantitative outputs

| Paper evidence | Code producer | Required input | Output |
|---|---|---|---|
| Automatic room construction | `python -m robo.eval.construction_metrics` | per-scene construction records | JSON, CSV, LaTeX |
| Room/object fidelity: PSNR, SSIM, LPIPS, CD, F1@20 | `python -m robo.eval.fidelity_metrics` | held-out render/GT and independent surfaces | JSON, CSV |
| Main manipulation table | `python -m robo.eval.harness_runner` or `python -m robo.eval.main_table` | persisted paired ledger | JSON, CSV, LaTeX + paired CIs |
| Task-local audit | `python -m robo.eval.audit_metrics` | leave-one-scene-out probabilities and labels | JSON, CSV, LaTeX |
| Harmonizer visual/preservation | `python -m robo.eval.harmony_visual_metrics` | aligned images, masks, temporal pairs, latency ledger | JSON, CSV, LaTeX |
| Real-world construction and paired trials | `python -m robo.eval.real_world_table` | workspace and matched-trial records | JSON, CSV, LaTeX |

Template input contracts live in `configs/experiments/` and intentionally contain
no invented numbers.

## Harness implementation

Implemented:

- typed treatment and comparison specifications
- exactly-one-axis comparison validation
- scene-construction, collision, and observation interventions
- one persisted reset bank reused across all treatments
- deterministic reset seeds independent of execution order
- append-only episode ledger and episode-level resume
- explicit success, task failure, build failure, policy timeout, safety
  termination, environment crash, and enhancer failure outcomes
- per-tick action, state, object pose, contact, stage, and latency traces
- per-episode provenance manifest
- complete planned-denominator coverage validation
- automatic main-table generation with reset-paired bootstrap deltas
- a fail-closed external Harmonizer service
- per-camera, per-episode temporal stream isolation
- exact raw robot-core restoration for Option C
- CI tests for treatment isolation, coverage, table denominators, audit metrics,
  fidelity metrics, socket ordering, and robot-core equality

## Remaining execution work, not code TODOs

These items require data, checkpoints, GPU time, or physical robot trials and
cannot be truthfully replaced by code or fabricated values:

1. Fill the paper-scale scene list and frozen policy checkpoint path.
2. Export a genuine private-shim MJCF for the collision ablation.
3. Expose the repository's photoreal composite observer through the
   `source_factory` configured in `harness_paper.yaml`.
4. Run the planned manipulation matrix and keep every scheduled episode.
5. Render held-out room/object views and independent target surfaces.
6. Run the hidden-ground-truth reconstructions and leave-one-scene-out audit
   predictions.
7. Run NVIDIA Harmonizer in its licensed upstream environment and collect
   aligned masks, temporal pairs, and latency.
8. Collect matched physical outcomes before populating real-success columns.

The paper must keep a dash or `TBD` for any item above until the corresponding
frozen artifact exists. DROID demonstration replay alone does not populate a
real-success cell.

## Paper integration

A table is copied into the paper only from a generated `.tex` artifact whose
JSON/CSV source and configuration hash are archived with the run. Do not edit a
generated numeric cell by hand. The final paper build should include:

```text
main_table.tex              -> tables/frozen_policy_main.tex
audit_table.tex             -> tables/task_local_audit_main.tex
harmony_visual_table.tex    -> tables/harmony_visual_main.tex
construction_table.tex      -> tables/automatic_construction_main.tex
real_world_table.tex        -> tables/real_world_main.tex
```

The fidelity table has mixed room and object units; its final LaTeX renderer
should preserve those two blocks rather than pooling the values.
