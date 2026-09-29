"""Build an explicitly approximate rigid body from selected scene Gaussians."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np


def append_proxy(root, name, proxy):
    if root.find(f".//body[@name='{name}']") is not None:
        raise ValueError(f"A body named {name} already exists")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    mesh_name = f"{name}_clicked_hull"
    ET.SubElement(asset, "mesh", name=mesh_name, file=proxy["mesh"])
    body = ET.SubElement(
        root.find("worldbody"),
        "body",
        name=name,
        pos=" ".join(map(str, proxy["center"])),
    )
    ET.SubElement(body, "freejoint", name=f"{name}_clicked_joint")
    physics = proxy["physics"]
    ET.SubElement(
        body,
        "geom",
        name=mesh_name,
        type="mesh",
        mesh=mesh_name,
        mass=str(physics["mass_kg"]),
        friction=f"{physics['friction']} .005 .0001",
        rgba=".8 .25 .15 1",
    )


def install_proxy(sim, name, points, directory):
    """Compile before changing the live simulation; retain existing poses and reset poses."""
    import mujoco
    import trimesh

    if sim.robot:
        raise ValueError("Create the new simulatable object before placing a robot")
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 24
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "Selection has insufficient finite 3D points for a collision body"
        )
    lo, hi = np.quantile(points, [0.001, 0.999], axis=0)
    points = points[((points >= lo) & (points <= hi)).all(1)]
    if len(points) < 12 or np.any(hi - lo < 0.002):
        raise ValueError(
            "Selection is too thin for a rigid collision proxy; try another view"
        )
    center = (lo + hi) / 2
    points = points[:: max(1, len(points) // 4000)] - center
    mesh = trimesh.convex.convex_hull(points)
    volume = abs(float(mesh.volume))
    if not np.isfinite(volume) or volume <= 1e-8:
        raise ValueError("Selection has no usable collision volume")
    directory.mkdir(parents=True, exist_ok=True)
    mesh_path = directory / "collision.obj"
    mesh.export(mesh_path)
    proxy = {
        "mesh": str(mesh_path.resolve()),
        "center": center.tolist(),
        "approximation": "Convex hull of selected observed Gaussian centers; hidden shape is unknown",
        "physics": {
            "mass_kg": float(np.clip(volume * 500, 0.01, 50)),
            "friction": 0.6,
            "restitution": None,
            "source": "Default density 500 kg/m3 and friction; unmeasured",
        },
        "volume_m3": volume,
    }
    root = ET.parse(sim.xml).getroot()
    append_proxy(root, name, proxy)
    text = ET.tostring(root).decode()
    model = mujoco.MjModel.from_xml_string(text)
    old_model, old_data = sim.model, sim.data
    if model.nq != old_model.nq + 7 or model.nv != old_model.nv + 6:
        raise ValueError("Unexpected model layout while adding a collision body")
    data = mujoco.MjData(model)
    initial_qpos = data.qpos.copy()
    initial_qpos[: old_model.nq] = sim.initial_qpos
    data.qpos[: old_model.nq] = old_data.qpos
    data.qvel[: old_model.nv] = old_data.qvel
    data.ctrl[:] = old_data.ctrl
    data.time = old_data.time
    mujoco.mj_forward(model, data)
    base = ET.fromstring(sim.base_xml)
    append_proxy(base, name, proxy)
    # Persistence is session-local, and source scientific artifacts are untouched.
    (directory / "proxy.json").write_text(json.dumps(proxy, indent=2))
    sim.xml.write_text(text)
    if sim.renderer:
        sim.renderer.close()
        sim.renderer = None
    sim.model, sim.data = model, data
    sim.initial_qpos = initial_qpos
    sim.initial[name] = (center.copy(), np.array([1.0, 0, 0, 0]))
    sim.base_xml = ET.tostring(base)
    sim.state.objects[name].meta["interactive_proxy"] = proxy
    sim.state.objects[name].physics = proxy["physics"]
    sim.enabled.add(name)
    sim.running = False
    return proxy
