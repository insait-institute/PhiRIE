# Primary sources and code starting points

Checked 2026-09-06. These links identify authoritative starting points, not pinned executable dependencies. SR0/SR7 must resolve actual code, dataset, checkpoint and engine revisions together. Do not install a floating latest release into an existing working environment.

## RoboCasa

- Overview: https://robocasa.ai/docs/introduction/overview.html
- Atomic tasks: https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html
- Policy-learning / evaluation interfaces: https://robocasa.ai/docs/build/html/benchmarking/policy_learning_algorithms.html
- Official multitask checkpoints: https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html
- Release updates: https://robocasa.ai/
- Code: https://github.com/robocasa/robocasa

Observed documentation version: 1.0.1. The project update page records a task-horizon change in May 2026. Match the native policy config, environment version and horizon; do not copy an old evaluation duration. The official algorithm page has separate Diffusion Policy, Openpi and GR00T integrations. A network family name alone does not establish embodiment or normalization compatibility. Resolve actual atomic task IDs from the installed registry, not generic names in this plan.

## BEHAVIOR / OmniGibson

- Concepts and BDDL scope: https://behavior.stanford.edu/getting_started/important_concepts.html
- Pre-sampled task instances: https://behavior.stanford.edu/behavior_components/behavior_tasks.html
- Native task API: https://behavior.stanford.edu/reference/tasks/behavior_task.html
- Object import guide: https://behavior.stanford.edu/omnigibson/objects.html
- Custom asset reference: https://behavior.stanford.edu/reference/objects/usd_object.html
- Installation / quickstart: https://behavior.stanford.edu/getting_started/quickstart.html
- Policy baselines: https://behavior.stanford.edu/challenge/baselines.html
- Code: https://github.com/StanfordVL/BEHAVIOR-1K

The task guide provides pre-sampled instance loading rather than resampling at every run. Object import and task-state support are separate obligations. Generic import does not prove Inside/OnTop/native goals work on a reconstructed asset. The inspected baseline guide uses the v3.9.1 stack and lists provided pi0.5/GR00T N1.7 task checkpoints for turning_on_radio; do not infer checkpoint availability for arbitrary rigid manipulation tasks. Treat custom-subgoal evaluations separately from full native activities. Asset handling must respect the source dataset terms; default release contains no source meshes.

## Rendering hardware

- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

Use the requirements for the Isaac Sim version actually supported by the pinned OmniGibson release. Verify renderer compatibility on the intended GPU; accelerator memory capacity alone is not sufficient evidence. Run official headless/compatibility checks before scheduling the fleet.

## PhiRoom integration reviewed

Code base at planning time: `5bcc6565a55346247f1d8761392a5fe71102c176`.

- `run/run_behavior_recon.sh`: old HDF5-to-3DGS-to-MJCF prototype, not native OmniGibson roundtrip; default GT-depth mesh path.
- `agents/recon/behavior_extract.py`: depth-backprojected initialization, static-clip/multiple-episode considerations, camera-intrinsic restrictions. Setting GT_MESH=0 alone does not make this RGB-only.
- `robo/eval/harness_spec.py`: existing typed treatment enums/axis checks require backward-compatible extension for native reference adapters and labeled joint system contrasts.
- `robo/eval/harness_runner.py`: canonical episode/ledger framework; native loading/stepping must not reuse DROID-specific state or success logic unmodified.
- `agents/orchestrator/`: shared proposals, selection, registration retry and ledger.
- `agents/orchestrator/runtime.py`: isolated single-hull-on-plane probe, not actual room support verification.
- `robo/rendering/`: GS composite, masks, restoration and native observation adapter starting points.
- `robo/eval/fidelity_metrics.py`, `paper_pipeline.py`: standard metrics and table integration; verify current CLI before use.
- `plan/icra2027/PROJECT_LIMITATIONS_AND_TODO.md`: existing project failures and scheduling constraint. Ordinary jobs, no arrays; do not rerun old full experiments unnecessarily.

The paper repository is `RunyiYang/SimAnyRoom`. This plan makes no changes to its title, text, figures or measured tables. New paper claims require new measured results and source-bound generation.
