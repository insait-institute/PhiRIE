"""Harness-facing photoreal observer with exact MuJoCo robot masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robo.rendering.composite_obs import build_observer as build_legacy_observer
from robo.rendering.mujoco_masks import render_robot_mask

DEFAULT_CAMERAS = {
    "observation/exterior_image_1_left": "ext_cam",
    "observation/wrist_image_left": "wrist_cam",
}


@dataclass
class HarnessCompositeObserver:
    observer: object
    env: object
    camera_names: dict[str, str]
    robot_roots: tuple[str, ...]

    def reset(self):
        method = getattr(self.observer, "reset", None)
        if callable(method):
            method()

    def _obs(self):
        for name in ("get_obs", "observation", "render_obs"):
            method = getattr(self.observer, name, None)
            if callable(method):
                result = method()
                if isinstance(result, dict):
                    return dict(result)
        if callable(self.observer):
            result = self.observer()
            if isinstance(result, dict):
                return dict(result)
        render = getattr(self.observer, "render", None)
        if callable(render):
            # The maintained pi05 renderer is intentionally camera-scoped;
            # assemble the exact DROID policy observation here rather than
            # demanding a historical dict-valued wrapper method.
            result = {
                image_key: render(camera_name)
                for image_key, camera_name in self.camera_names.items()
            }
            result["observation/joint_position"] = self.env.joint_position()
            result["observation/gripper_position"] = self.env.gripper_position()
            return result
        raise RuntimeError("legacy CompositeObs exposes no dict-valued observation method")

    def get_obs(self):
        obs = self._obs()
        for image_key, camera_name in self.camera_names.items():
            if image_key not in obs:
                continue
            image = np.asarray(obs[image_key])
            mask = render_robot_mask(
                self.env, camera=camera_name,
                width=image.shape[1], height=image.shape[0],
                robot_roots=self.robot_roots)
            obs[image_key.replace("image", "robot_mask")] = mask
        return obs


def build_observer(*, env, suite, factory_dir=None, out_dir=None,
                   render_wh=(640, 360), options=None, **kwargs):
    options = dict(options or {})
    observer = build_legacy_observer(
        env=env, suite=suite, factory_dir=factory_dir, out_dir=out_dir,
        render_wh=render_wh, options=options, **kwargs)
    camera_names = dict(DEFAULT_CAMERAS)
    camera_names.update(options.get("camera_names", {}))
    robot_roots = tuple(options.get(
        "robot_roots", ("panda", "franka", "robot", "robotiq")))
    return HarnessCompositeObserver(observer, env, camera_names, robot_roots)
