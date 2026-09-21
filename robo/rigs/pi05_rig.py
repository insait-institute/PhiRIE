"""DROID-style robot rig for SimAny scenes: Franka Panda + Robotiq 2F-85
assembled into an exported scene.xml via MjSpec, with ZED-like external and
wrist cameras plus a solid table collision box (the exported scene only has
per-object micro-slabs - an arm needs a real tabletop to sweep against).

Conventions (matched to DROID + openpi pi05_droid_jointpos):
- control 15 Hz, physics 1/600 s (robotiq 4-bar needs dt <= 2 ms), 40
  substeps per control tick
- policy actions are ABSOLUTE joint position targets (7) + gripper [0,1]
  (the server-side AbsoluteActions transform already added the state);
  per-tick delta is clamped to 0.2 rad like droid/robot_ik_solver.py
- external camera at DROID-typical (0.05, +/-0.57, 0.66) m in the robot
  base frame; images rendered 1280x720 (left ZED eye equivalent)

Robot mesh paths are absolutized before attach so the scene compiler's
meshdir (absolute path into the scene output dir) does not swallow them.
"""
import os
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MENAGERIE = (
    Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie")
# Compatibility aliases for callers which inspect these constants.  E4 passes
# a commit-verified root at call time instead of relying on this worktree-
# relative directory (which may intentionally be unpopulated).
MENAGERIE = DEFAULT_MENAGERIE
PANDA_XML = DEFAULT_MENAGERIE / "franka_emika_panda" / "panda_nohand.xml"
GRIPPER_XML = DEFAULT_MENAGERIE / "robotiq_2f85" / "2f85.xml"

# DROID reset pose (droid/robot_env.py): [0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0]
PANDA_HOME = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0,
                       3 * np.pi / 5, 0.0])
CONTROL_HZ = 15
PHYSICS_DT = 1.0 / 600.0
SUBSTEPS = int(round(1.0 / CONTROL_HZ / PHYSICS_DT))  # 40
MAX_JOINT_DELTA = 0.2  # rad per 15 Hz tick (droid robot_ik_solver.py)

# SERL mounts base_mount with an extra -90 deg about z inside the Menagerie
# attachment body (which itself is +135 deg from link7)
MOUNT_QUAT = np.array([-1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)


def _absolutize_assets(spec, xml_path):
    """Rewrite relative mesh/texture file paths to absolute ones."""
    base = Path(xml_path).parent / (spec.meshdir or "")
    for m in spec.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str(base / m.file)
    tbase = Path(xml_path).parent / (spec.texturedir or "")
    for t in spec.textures:
        if t.file and not Path(t.file).is_absolute():
            t.file = str(tbase / t.file)
    spec.meshdir = ""
    spec.texturedir = ""


def _delete_keyframes(spec):
    for k in list(spec.keys):
        spec.delete(k)


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """wxyz quat for a MuJoCo camera at pos looking at target (-z forward,
    +y up in camera frame)."""
    pos, target, up = map(np.asarray, (pos, target, up))
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:  # looking straight down/up
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / n
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return rot_to_quat_wxyz(R)


def rot_to_quat_wxyz(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def resolve_menagerie_root(root=None):
    """Resolve a supplied Menagerie checkout and require both rig XMLs."""
    value = root or os.environ.get("SIMANY_MUJOCO_MENAGERIE_ROOT")
    path = Path(value).expanduser() if value else DEFAULT_MENAGERIE
    path = path.resolve(strict=False)
    required = (
        path / "franka_emika_panda" / "panda_nohand.xml",
        path / "robotiq_2f85" / "2f85.xml",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"MuJoCo Menagerie root {path} lacks required rig assets: {missing}")
    return path


def load_rig_spec(menagerie_root=None):
    """Panda (no hand) with the 2F-85 attached at attachment_site."""
    root = resolve_menagerie_root(menagerie_root)
    panda_xml = root / "franka_emika_panda" / "panda_nohand.xml"
    gripper_xml = root / "robotiq_2f85" / "2f85.xml"
    panda = mujoco.MjSpec.from_file(str(panda_xml))
    grip = mujoco.MjSpec.from_file(str(gripper_xml))
    _absolutize_assets(panda, panda_xml)
    _absolutize_assets(grip, gripper_xml)
    _delete_keyframes(panda)
    _delete_keyframes(grip)

    site = panda.site("attachment_site")
    site.quat = MOUNT_QUAT
    # wrist camera on the gripper base, ZED-Mini-ish: centered on the
    # approach axis (pinch site is at z=0.145 in the 2f85 "base" frame),
    # aimed well past the fingertips so the workspace stays in frame.
    # A first attempt offset sideways behind the (wide) actuator housing
    # and aimed just past the pinch point - the housing filled >50% of the
    # frame at every arm pose tested (see docs/DEMO_STORYBOARD-style debug
    # dumps in outputs/pi05_runs/dbg1_*/*_wr224.png), plausibly enough of a
    # distribution shift from real DROID wrist framing to explain the
    # near-zero commanded joint deltas seen in closed-loop eval. Verified
    # against renders at both the home pose and a mid-reach pose.
    base = grip.body("base")
    cam_pos = np.array([0.0, 0.0, 0.02])
    cam_target = np.array([0.0, 0.0, 0.45])
    base.add_camera(name="wrist", pos=cam_pos,
                    quat=lookat_quat(cam_pos, cam_target, up=(0.0, -1.0, 0.0)),
                    fovy=58.0)
    panda.attach(grip, site=site, prefix="2f85/")
    return panda


def _table_geometry(table):
    """Keep the declared top surface without filling occupied space below it."""
    thickness = float(table.get("thickness_m", table["top_z"]))
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("table thickness_m must be finite and positive")
    half = thickness / 2.0
    return ([table["cx"], table["cy"], table["top_z"] - half],
            [table["hx"], table["hy"], half])


def build_scene_model(scene_xml, base_pos, base_yaw, table_box=None,
                      ext_cam=None, exclude_objects=(), xml_dump=None,
                      menagerie_root=None):
    """Assemble scene + rig; returns (model, info dict).

    base_pos: (3,) robot base (link0) position, world; base_yaw: rad about z.
    table_box: dict(cx, cy, hx, hy, top_z), optionally thickness_m for a slab.
        Without thickness_m, retain the legacy solid box from the floor.
    ext_cam: dict(pos(3, base-frame offset), target(3, world), fovy).
    exclude_objects: body names to delete (unstable / excluded from task).
    """
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    for name in exclude_objects:
        b = scene.body(name)
        if b is not None:
            scene.delete(b)

    # option merge: scene (implicitfast) + 2f85 (elliptic cone, impratio 10)
    scene.option.timestep = PHYSICS_DT
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10.0
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    if table_box:
        t = table_box
        table_pos, table_size = _table_geometry(t)
        scene.worldbody.add_geom(
            name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=table_pos,
            size=table_size,
            friction=[0.6, 0.005, 0.0001], rgba=[0.55, 0.45, 0.35, 0.35])

    cy, sy = np.cos(base_yaw), np.sin(base_yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rig = load_rig_spec(menagerie_root)
    frame = scene.worldbody.add_frame(pos=np.asarray(base_pos),
                                      quat=rot_to_quat_wxyz(Rz))
    scene.attach(rig, frame=frame, prefix="robot/")

    if ext_cam is None:
        ext_cam = {"pos": [0.05, 0.57, 0.66], "target": None, "fovy": 68.0}
    if ext_cam.get("mode") == "world":  # explicit world pose (scan frame)
        scene.worldbody.add_camera(
            name="ext_cam", pos=np.asarray(ext_cam["pos"], float),
            quat=np.asarray(ext_cam["quat_wxyz"], float),
            fovy=ext_cam.get("fovy", 68.0))
    else:  # base-frame offset + world lookat target
        cam_world = (np.asarray(base_pos) +
                     Rz @ np.asarray(ext_cam["pos"], float))
        target = ext_cam.get("target")
        if target is None:
            target = np.asarray(base_pos) + Rz @ np.array([0.55, 0.0, 0.10])
        scene.worldbody.add_camera(
            name="ext_cam", pos=cam_world,
            quat=lookat_quat(cam_world, np.asarray(target, float)),
            fovy=ext_cam.get("fovy", 68.0))

    model = scene.compile()
    info = {
        "arm_joints": [f"robot/joint{i}" for i in range(1, 8)],
        "arm_actuators": [f"robot/actuator{i}" for i in range(1, 8)],
        "gripper_actuator": "robot/2f85/fingers_actuator",
        "gripper_driver_joint": "robot/2f85/right_driver_joint",
        "ext_cam": "ext_cam", "wrist_cam": "robot/2f85/wrist",
        "home": PANDA_HOME.copy(),
    }
    # sanity: every name must resolve
    for j in info["arm_joints"]:
        assert model.joint(j) is not None
    for a in info["arm_actuators"]:
        assert model.actuator(a) is not None
    assert model.actuator(info["gripper_actuator"]) is not None
    assert model.camera(info["ext_cam"]) is not None
    assert model.camera(info["wrist_cam"]) is not None
    if xml_dump:
        scene.to_file(str(xml_dump))
    return model, info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--base", nargs=3, type=float, required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()
    model, info = build_scene_model(args.scene_xml, args.base, args.yaw,
                                    xml_dump=args.dump or None)
    print(f"[rig] compiled: nq={model.nq} nu={model.nu} nbody={model.nbody}")
    print(f"[rig] cams: {info['ext_cam']}, {info['wrist_cam']}")
