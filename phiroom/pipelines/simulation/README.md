# simulation

Simulation scene export and dynamics.

- Inputs: accepted assets and physical metadata.
- Outputs: MJCF or simulator assets; dynamics traces.
- Tools: MuJoCo, PyBullet, OmniGibson.
- Contract: [phiroom/modules/simulation.json](../../modules/simulation.json).

| Action | Runtime | Implementation |
|---|---|---|
| `mujoco` | `main` | `robo.sim.export_mjcf` |
| `pybullet` | `main` | `robo.sim.s7_sim` |
| `omnigibson` | `main` | `robo.sim.export_omnigibson` |

```bash
bash run/phiroom.sh describe simulation
bash run/phiroom.sh plan simulation mujoco -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run simulation mujoco -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
