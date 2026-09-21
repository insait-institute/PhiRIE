# Full-input, bounded qualification protocol

This is a new prospective qualification study following the failed compact
pilot. It preserves that pilot's failure and does not change any geometry,
physics, camera, reset, or policy success threshold. It is not a completed
manipulation experiment or permission to rerun the old pilot with easier tasks.

`run.icra2027.e4_full_protocol` authenticates the existing TRAIN discovery
population and uses its existing task-role vocabulary. All 50 scenes and 1871
object jobs remain in the input denominator. All semantic queries are recorded;
a fixed input-only budget takes the first four task IDs per family per scene.
Only those slots are planned for expensive qualification. Budget exclusions and
failed qualification are reported separately, before conditional performance.
Neither controller decisions nor GT, evaluation metrics, or policy outcomes are
read by the protocol builder.

The canonical E4 materializer, candidate qualifier, camera/scorer, reset bank,
and harness must execute and authenticate the declared stages. The selector
accepts a complete view of their boolean gates; it is not a new evaluator.
After qualification, use the first four lexicographic scenes with two valid
tasks in each family, and the first two tasks per family. Five identical reset
states in A0 and A4 give 80 planned episodes per arm for one pinned policy.
The pilot uses the first two scenes and first task in each family. Its episodes
may be retained in full execution only if source, policy, checkpoint and every
frozen field are identical. No task/scene replacement follows policy outcomes.
If fewer than four qualifying rooms exist, the paper-target matrix is NOT_RUN;
no threshold is relaxed. The two families and four-room minimum stay fixed.

The builder and selector are ready; the full-source materialization/qualification
execution contract remains a dependency. Do not launch full qualification until
all 50 E3 scene seals pass their independent construction integrity gate. No
policy job may launch before canonical qualification, camera, scripted-stage,
policy identity, smoke and pilot gates pass. A0/A4 is the construction axis;
collision and observation interventions still require separate frozen blocks.

Reproduce the input-only protocol in a **new** path:

```bash
python -m run.icra2027.e4_full_protocol --out <new-protocol.yaml>
pytest -q tests/test_e4_full_protocol.py
```

The checked-in generated protocol is
`configs/experiments/icra2027/e4_full_protocol.yaml`.
