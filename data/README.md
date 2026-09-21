# data/

Documented mount point for datasets. Contents are gitignored (only this
README is meant to be tracked); the data itself lives on the cluster, and
the pipeline reads it from there by default:

| dataset | default location | override |
|---|---|---|
| ScanNet++ v2 (frames, meshes, splits) | `/data/ScanNetpp` | `SIMANY_SCANNETPP_ROOT` |
| Per-scene 3D Gaussian splats | `/data/ScanNetppv2_gsplat/splats` | `SIMANY_SPLATS_ROOT` |

Defaults and overrides are defined in [`../run/env.sh`](../run/env.sh); the
pipeline launchers (`run_auto.sh`, `run_factory.sh`, `run_inpaint.sh`,
`run_scene.sh`) source that file, so the main pipeline (the `run_*.sh`
stages) hardcodes no cluster path. Some auxiliary scripts (e.g.
`interface/viewer.py` and a few `run/slurm/fleet_*.sbatch` jobs) still
hardcode cluster paths and ignore these overrides. To use a local copy
with the main pipeline, export the variables before running:

```sh
export SIMANY_SCANNETPP_ROOT=/path/to/ScanNetpp
export SIMANY_SPLATS_ROOT=/path/to/splats
```

See [`../docs/DATA_AND_WEIGHTS.md`](../docs/DATA_AND_WEIGHTS.md) for the
expected directory layout and details.
