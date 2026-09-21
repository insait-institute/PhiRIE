"""Isolated, fast rendering smoke test: does omni.hydra.rtx / VisionSensor
camera capture actually work on this node at all, before committing to the
full (~30+ min) robot_pick_place_demo.py --render run?

Boots OmniGibson WITH rendering enabled (no headless_stubs, matching
robot_pick_place_demo.py's --render code path), builds a minimal scene with
one DatasetObject and one VisionSensor camera, captures a single RGB frame,
and reports success/failure. This is the actual open question flagged
before ever running the full render: whether omni.hydra.rtx (the Hydra
ray-tracing render delegate that segfaults on hala's driver) also fails on
this rtx6000/Blackwell node, or whether it boots clean here.

Run with: OMNIGIBSON_HEADLESS=1 python render_smoke_test.py --out /path/to/frame.png
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=str, required=True)
args = ap.parse_args()

print("[render_smoke] python", sys.version)

import torch as th  # noqa: E402

print("[render_smoke] torch", th.__version__, "cuda available:", th.cuda.is_available())
if th.cuda.is_available():
    print("[render_smoke] gpu:", th.cuda.get_device_name(0))

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.USE_GPU_DYNAMICS = True
print("[render_smoke] gm.HEADLESS =", gm.HEADLESS)

og.launch()
print("[render_smoke] og.launch() OK - Isaac Sim booted with rendering enabled")

from omnigibson.scenes.scene_base import Scene  # noqa: E402
from omnigibson.objects import DatasetObject  # noqa: E402
from omnigibson.sensors import VisionSensor  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402

scene = Scene(use_floor_plane=True, use_skybox=False, include_robots=False)
og.sim.import_scene(scene)
print("[render_smoke] Scene imported")

from omnigibson.utils.asset_utils import get_all_object_category_models  # noqa: E402

models = get_all_object_category_models("plate")
obj = DatasetObject(name="plate", category="plate", model=models[0])
scene.add_object(obj=obj)
obj.set_position_orientation(position=th.tensor([0.6, 0.0, 0.5]))
print("[render_smoke] added a plate DatasetObject")

camera = VisionSensor(
    relative_prim_path="/demo_cam", name="demo_cam",
    modalities=["rgb"], image_height=480, image_width=640,
)
camera.load(scene)
camera.set_position_orientation(
    position=th.tensor([1.8, -0.8, 1.2]),
    orientation=T.euler2quat(th.tensor([1.05, 0.0, 2.05])),
)
camera.initialize()
print("[render_smoke] VisionSensor camera created + initialized")

og.sim.play()
print("[render_smoke] og.sim.play() OK")

for _ in range(30):
    og.sim.step()
print("[render_smoke] 30 physics/render steps OK")

obs = camera.get_obs()[0]
img = obs["rgb"][..., :3].cpu().numpy()
print("[render_smoke] captured rgb frame, shape =", img.shape, "dtype =", img.dtype)

import imageio  # noqa: E402

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
imageio.imwrite(out_path, img)
print(f"[render_smoke] wrote {out_path}")

print("[render_smoke] ALL OK")
