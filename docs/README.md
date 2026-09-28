# docs/

Index of the PhiRIE documentation. Some pages cite `plan/NN_*.md` task notes;
those are internal planning documents that are not part of this repository.

| page | contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | top-level layout: what lives in `agents/`, `models/`, `robo/`, `interface/`, `run/` and how they fit together |
| [PIPELINE.md](PIPELINE.md) | the generation pipeline stage by stage: discover, assets, edit, render, eval |
| [MODULES.md](MODULES.md) | modular feature contracts, the `phiroom` control CLI and pipeline recipes |
| [PHIVIEW.md](PHIVIEW.md) | the PhiView submodule: checkout, CPU tools, GPU/backend setup, updating the pin |
| [ROBOT.md](ROBOT.md) | robot layer: MuJoCo env, Panda + Robotiq rig, photoreal observations, closed-loop pi0.5 evaluation |
| [DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) | where datasets, checkpoints and caches live, and the env vars that override the paths |
| [ENVIRONMENTS.md](ENVIRONMENTS.md) | why three python environments are unavoidable, plus other environment facts |
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | what the contribution actually is, with the audited numbers behind each claim |
| [BASELINES.md](BASELINES.md) | baselines and concurrent work: what we compare against, what we owe, what we cite |
| [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md) | reproduction ledger for every construction baseline |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | claim, experiment and metric matrix for the paper |
| [ICRA_RESEARCH_CONTRACT.md](ICRA_RESEARCH_CONTRACT.md) | frozen research contract: claims, evidence tracks and gates |
| [PAPER_CODE_STATUS.md](PAPER_CODE_STATUS.md) | map from every quantitative paper claim to its code producer |
| [METRICS.md](METRICS.md) | predictive metrics, hierarchical bootstrap and power analysis |
| [PAIRED_HARNESS.md](PAIRED_HARNESS.md) | the paired robot-evaluation harness behind the main manipulation table |
| [MUJOCO_PAIRED_PROTOCOL.md](MUJOCO_PAIRED_PROTOCOL.md) | the `mujoco_paired` reference-versus-twin protocol |
| [ORACLE_PROTOCOL.md](ORACLE_PROTOCOL.md) | oracle-causal reconstruction benchmark on BEHAVIOR captures |
| [DROID_PROTOCOL.md](DROID_PROTOCOL.md) | DROID workspace reconstruction and FK alignment protocol |
| [GAP_STUDY.md](GAP_STUDY.md) | reconstructed twin versus native simulator gap study |
| [BUILD_AUDIT_SCHEMA.md](BUILD_AUDIT_SCHEMA.md) | the per-build `build_audit.json` schema |
| [POLARIS_INTEGRATION.md](POLARIS_INTEGRATION.md) | PolaRiS / Isaac Sim integration status and blockers |
| [demos/](demos/) | recorded demo packages |
| [releases/](releases/) | release notes for 2.0.0 and 2.0.1 |
