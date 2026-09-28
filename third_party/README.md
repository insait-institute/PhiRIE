# third_party/

Vendored external checkouts. Everything in this directory is gitignored
except `bddl_data/` and this README. [`../run/setup_env.sh`](../run/setup_env.sh)
re-clones `TRELLIS/` if it is missing; the rest are documented here so they
can be restored by hand.

| checkout | what it is | used by |
|---|---|---|
| `TRELLIS/` | Microsoft image-to-3D generative model | [`models/s4_trellis.py`](../models/s4_trellis.py) (per-object asset generation) |
| `ReconViaGen/` | multi-view image-to-3D (VGGT-conditioned TRELLIS) | [`models/s4_reconviagen.py`](../models/s4_reconviagen.py), [`agents/assets/factory_hybrid.py`](../agents/assets/factory_hybrid.py) |
| `MaskClustering/` | CVPR 2024 class-agnostic 3D instance discovery | [`agents/baselines/maskclustering.py`](../agents/baselines/maskclustering.py) (discovery baseline; own `.venv-mc` env) |
| `FlashSplat/` | ECCV 2024 2D-mask-to-3DGS segmentation | reference clone only — [`agents/baselines/flashsplat.py`](../agents/baselines/flashsplat.py) reimplements it on gsplat and does not import this checkout |
| `BEHAVIOR-1K/` | OmniGibson simulator + BDDL toolchain | [`robo/sim/export_omnigibson.py`](../robo/sim/export_omnigibson.py), [`robo/sim/omnigibson_bridge/`](../robo/sim/omnigibson_bridge/) |
| `bddl_data/` | BDDL activity definitions + generated data (**tracked in git**) | [`agents/eval/behavior1k_coverage.py`](../agents/eval/behavior1k_coverage.py) |
| `behavior1k_datasets/` | OmniGibson asset datasets, incl. the `phiroom_custom` dataset of our converted twins | [`robo/sim/omnigibson_bridge/convert_assets.py`](../robo/sim/omnigibson_bridge/convert_assets.py) |
| `mujoco_menagerie/` | DeepMind robot model zoo (Franka Panda, Robotiq 2F-85) | [`robo/rigs/pi05_rig.py`](../robo/rigs/pi05_rig.py) |

`bddl_data/` is redistributed from the BDDL project
([StanfordVL/bddl](https://github.com/StanfordVL/bddl)); see that repository for
its license terms.

`BEHAVIOR-1K/` is a manual clone; running
[`robo/sim/omnigibson_bridge/install_omnigibson.sbatch`](../robo/sim/omnigibson_bridge/install_omnigibson.sbatch)
from it installs the `behavior1k` conda env and downloads
`behavior1k_datasets/`.
