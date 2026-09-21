# SR5 — Fixed-action physical fidelity

**Owner:** replay/physics agent. **Priority:** P0 after SR2. Read ../PROTOCOL.md and ../TABLES_AND_METRICS.md. This experiment separates physical asset effects from visuomotor-policy perception. It is not a learned-policy benchmark.

## Reference action bank

For each frozen task instance/reset, record a native reference trajectory produced by the compatible policy or a separately labeled state-based diagnostic controller. Save the complete input action stream, control timestamps, robot state, object state, contacts, reset and all controller settings. Choose trajectory IDs by a predeclared rule. Retain failed reference trajectories, and identify the reference-successful subset only as a conditional diagnostic.

If using a demonstration, reproduce its actions in the native simulator first. A dataset's object-pose playback is not a dynamics trace. Ensure the source robot/controller exactly matches. Use the same initial known robot state and PROTOCOL.md's paired object perturbations.

## Replay conditions

Run the identical ROBOT actions in reference, B0, B3 and B4 native environments. Execute engine integration at identical timestep/substeps and control rate. Do not write object poses, force contacts, attach objects, substitute ideal grasps or regenerate a plan in each arm. If a state-based controller replans independently, classify that as a separate state-feedback diagnostic, not fixed-action replay.

Use a fixed action horizon. Native-policy early success does not truncate one comparison's trace selectively. Choose an identical replay prefix for all arms and separately report success at the native horizon. On crash/safety stop, preserve the available prefix and failure event; later errors are missing, never padded with the last good state or zero.

## Metrics

Standard quantities only:

- Position RMSE in centimeters along a common timestamp grid.
- Rotation geodesic error in degrees with explicit quaternion convention; symmetries are handled only by a predeclared object-equivalence rule.
- Final position/orientation errors.
- Native task completion rate under replay.
- Initial room-settle stable count and coverage, separate from action replay.

Use a consistent evaluation object frame. Native link origin, mesh origin and COM can differ. Establish the fixed frame correspondence before comparing trajectories; never align each resulting trajectory to GT or remove initial estimation error in the headline RMSE. Also archive displacement-from-start error as a diagnostic to distinguish placement error from response error. Report its different definition explicitly.

Average over trajectories/task instances/scenes, not all time samples pooled together. Full error curves and contact sequence visualizations are useful diagnostics; no need to invent a weighted simulator score. RMSE on the surviving prefix is conditional and must include retained duration and completed/planned trace counts.

## Small physical probe suite

Use settle, a controlled push by a known robot/tool, grasp-and-lift and release-into-open-container when the frozen task supports them. Do not create automatic tests that move every object directly by a prescribed position and then call this physical validity. A direct external force experiment can be an explicitly labeled additional probe but is not native robot manipulation.

Start with native identity controls U0/U1. Estimate baseline simulator-repeatability tolerance from duplicate native executions on the same hardware. Do not compare a deterministic MuJoCo run against a different PhysX run to infer reconstruction error.

## Proposed implementation

Add `robo/roundtrip/replay.py` as a workload using the canonical episode/trace schema and native adapter. Reuse metric serialization and bootstrap helpers.

```bash
python -m robo.roundtrip.replay --config configs/experiments/sim_recon_sim/replay.yaml --actions "$ACTION_BANK" --out "$OUT/replay"
```

Outputs: per-trajectory manifests, robot actions, state/contact traces, `replay_metrics.csv`, coverage/failure table, identity-control report and synchronized videos. Keys include execution_kind=`action_replay` so SR6 cannot ingest these as learned-policy episodes.

## Acceptance and first pilot

One native action trace, the wrapper/import control, and one reconstructed-object trace are enough for the first interpretable result. Then run the two development layouts. Do not wait for Gaussian or Harmonizer rendering.

Tests: action arrays match byte-for-byte after declared native preprocessing; no object-pose setter after reset; same action count/timebase; crash-prefix handling; world/local-frame unit test; known synthetic rotation/translation metrics; missing target coverage; native success evaluator result reflects live reconstructed geometry.

Handoff exact trajectories/metrics and one causal diagnosis to SR3/SR6. A large error triggers support/pose/collision debugging, not automatic policy retuning. A negative result with valid physics and accounting is an admissible outcome.
