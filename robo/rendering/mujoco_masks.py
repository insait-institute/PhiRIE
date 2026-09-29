"""Exact MuJoCo robot masks from segmentation rendering."""
from __future__ import annotations

from typing import Iterable

import numpy as np


class RobotMaskError(RuntimeError):
    pass


def _name(model, object_type, index):
    import mujoco
    return mujoco.mj_id2name(model, object_type, int(index)) or ""


def robot_body_ids(model, roots: Iterable[str] = ("panda", "franka", "robot", "robotiq")):
    """Find robot bodies by explicit root-name substring and all descendants."""
    import mujoco
    roots = tuple(value.lower() for value in roots)
    selected = set()
    for body_id in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id).lower()
        if any(root in name for root in roots):
            selected.add(body_id)
    # Descendant closure. Parent IDs are always earlier in standard MuJoCo
    # models, but iterate to a fixed point for robustness.
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if body_id not in selected and int(model.body_parentid[body_id]) in selected:
                selected.add(body_id)
                changed = True
    if not selected:
        raise RobotMaskError(
            f"no robot body matched roots {roots}; provide explicit robot roots")
    return selected


def robot_geom_ids(model, roots=("panda", "franka", "robot", "robotiq")):
    bodies = robot_body_ids(model, roots)
    ids = {geom_id for geom_id in range(model.ngeom)
           if int(model.geom_bodyid[geom_id]) in bodies}
    if not ids:
        raise RobotMaskError("matched robot bodies contain no geoms")
    return ids


def robot_mask_from_segmentation(segmentation: np.ndarray, geom_ids) -> np.ndarray:
    """Convert MuJoCo segmentation output to an uint8 robot mask."""
    value = np.asarray(segmentation)
    geom_ids = np.asarray(sorted(geom_ids), dtype=np.int64)
    if value.ndim == 3 and value.shape[2] >= 2:
        # MuJoCo returns (object-id, object-type).
        try:
            import mujoco
            geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = (value[..., 1] == geom_type) & np.isin(value[..., 0], geom_ids)
        except ImportError:
            mask = np.isin(value[..., 0], geom_ids)
    elif value.ndim == 2:
        mask = np.isin(value, geom_ids)
    else:
        raise RobotMaskError(f"unexpected segmentation shape {value.shape}")
    return mask.astype(np.uint8) * 255


def render_robot_mask(env, *, camera, width: int, height: int,
                      robot_roots=("panda", "franka", "robot", "robotiq"),
                      scene_option=None):
    """Render a segmentation mask without altering the environment state."""
    try:
        import mujoco
    except ImportError as exc:
        raise RobotMaskError("mujoco is required for exact robot masks") from exc
    renderer = mujoco.Renderer(env.model, height=int(height), width=int(width))
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera=camera, **({"scene_option":scene_option} if scene_option is not None else {}))
        segmentation = renderer.render()
    finally:
        renderer.close()
    return robot_mask_from_segmentation(
        segmentation, robot_geom_ids(env.model, robot_roots))
