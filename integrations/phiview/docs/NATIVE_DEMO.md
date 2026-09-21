# Native reconstruction demo

This entry loads a frozen B3 reconstruction into its original RoboCasa kitchen,
using `robo.roundtrip.paired.prepare_paired_adapter` and its estimated placement.
The native renderer, controller and simulator run in the original native Python
environment. PhysicalView's existing viser environment displays colored body
geometry and a textured native camera image. Room context remains native; only
the target is reconstructed.

The worker re-executes the existing episode actions through
`RoboCasaAdapter.step_native_action`. It does not run a new policy or write an
evaluation ledger. The original episode and this presentation replay are
separate artifacts. Pause, Play and Reset control the presentation worker.

The example is the first RECORDED B3 row sorted by unit ID, selected without
conditioning on success:
`unit-00af68df450a157280fb45c0e3c403d2`.
Its original official outcome is failure. It is intentionally retained.

Runtime output: `/group/worldcept/code/PhysicalView/outputs/native-demo-20260908`.
The `source.json`, `import_receipt.json`, `demo_result.json`, continuous MP4,
per-body PLY files and `live.json` describe the actual scene and execution.

Submit an ordinary GPU job with `sbatch run/native_demo.sbatch`. The supplied
launcher uses the existing local environments. A GPU override may be supplied
at submission. Read the job log and tunnel to its allocated node, port 8092.
Native execution must remain inside a Slurm allocation.

On 2026-09-08 job 856118 was cancelled while pending due to the long H200 queue.
Job 856127 uses an idle RTX 6000 on `gcp-eu1-rtx6000-26hb`. The final viewer runs
in an overlapping step at port 8093, focused on the reconstruction:

```bash
ssh -N -L 8093:gcp-eu1-rtx6000-26hb:8093 runyi_yang@hala.slurm.insait.ai
```

Open http://localhost:8093. The live allocation lasts two hours from its start;
saved videos remain available afterward.
