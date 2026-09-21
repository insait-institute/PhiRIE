"""DROID-convention MuJoCo environment over a SimAny scene, for closed-loop
pi0.5 (pi05_droid_jointpos) evaluation.

Observation dict follows openpi's droid_policy exactly:
  observation/exterior_image_1_left  uint8 (H, W, 3)  - left-ZED-equivalent
  observation/wrist_image_left       uint8 (H, W, 3)
  observation/joint_position         float (7,)  absolute rad
  observation/gripper_position       float (1,)  0=open .. 1=closed
Actions per 15 Hz tick: (8,) = absolute joint position targets (7) +
gripper position [0,1], binarized at 0.5 (openpi examples/droid).
"""
import math

import numpy as np

import mujoco

from robo.manifest.hash import canonical_hash
from robo.rigs import pi05_rig as rig


def _settle_duration_tolerance(duration_s, timestep_s):
    """Floating-point tolerance for an integer-duration physics settle."""
    scale = max(1.0, abs(float(duration_s)), abs(float(timestep_s)))
    return max(1e-12, 32.0 * math.ulp(scale))


def _settle_step_count(duration_s, timestep_s):
    """Return an exact grid-aligned step count without truncation drift.

    MuJoCo's 1/600 second timestep makes ``1.5 / timestep`` evaluate just
    below 900 on this runtime.  Blind ``int`` therefore drops a step, while
    blind ``round`` would silently accept arbitrary off-grid durations.  We
    round only when the reconstructed duration agrees within a small,
    scale-aware floating-point tolerance; otherwise the request is invalid.
    """
    for value, label in ((duration_s, "settle_s"), (timestep_s, "timestep")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{label} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be a finite real number")
    duration = float(duration_s)
    timestep = float(timestep_s)
    if duration < 0.0:
        raise ValueError("settle_s must be non-negative")
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    ratio = duration / timestep
    if not math.isfinite(ratio):
        raise ValueError("settle step ratio must be finite")
    steps = int(round(ratio))
    simulated = steps * timestep
    tolerance = _settle_duration_tolerance(duration, timestep)
    if abs(simulated - duration) > tolerance:
        raise ValueError(
            "settle_s must be an integer multiple of the MuJoCo timestep "
            f"within {tolerance:.3g}s"
        )
    return steps


class DroidSimEnv:
    def __init__(self, scene_xml, base_pos, base_yaw, table_box=None,
                 ext_cam=None, exclude_objects=(), render_wh=(1280, 720),
                 xml_dump=None, menagerie_root=None):
        self.model, self.info = rig.build_scene_model(
            scene_xml, base_pos, base_yaw, table_box=table_box,
            ext_cam=ext_cam, exclude_objects=exclude_objects,
            xml_dump=xml_dump, menagerie_root=menagerie_root)
        self.data = mujoco.MjData(self.model)
        m, i = self.model, self.info
        self._jadr = np.array([m.joint(n).qposadr[0] for n in i["arm_joints"]])
        self._jdadr = np.array([m.joint(n).dofadr[0] for n in i["arm_joints"]])
        self._aids = np.array([m.actuator(n).id for n in i["arm_actuators"]])
        self._grip_aid = m.actuator(i["gripper_actuator"]).id
        self._grip_jadr = m.joint(i["gripper_driver_joint"]).qposadr[0]
        self._ctrl_lo = m.actuator_ctrlrange[self._aids, 0]
        self._ctrl_hi = m.actuator_ctrlrange[self._aids, 1]
        w, h = render_wh
        self.renderer = mujoco.Renderer(m, height=h, width=w)
        self.free_bodies = [m.body(b).name for b in range(m.nbody)
                            if m.body(b).jntnum[0] == 1 and
                            m.body(b).name.startswith("obj_")]
        # finger pad geoms for grasp detection
        self.pad_geoms = {"left": [], "right": []}
        for g in range(m.ngeom):
            name = m.geom(g).name or ""
            if "2f85" in name and "pad" in name:
                side = "left" if "left" in name else "right"
                self.pad_geoms[side].append(g)
        assert self.pad_geoms["left"] and self.pad_geoms["right"], \
            "no gripper pad geoms found"
        self._body_geoms = {}
        for b in self.free_bodies:
            bid = m.body(b).id
            adr, num = m.body_geomadr[bid], m.body_geomnum[bid]
            self._body_geoms[b] = set(range(adr, adr + num))
        self.composite = None  # optional pi05_render.CompositeObs
        # Populated only after a complete reset/settle.  The harness copies
        # this value into each immutable rollout contract before policy use.
        self.last_reset_provenance = None

    @staticmethod
    def _joint_widths(joint_type):
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1

    def _reset_state_snapshot(self):
        """Return the complete named dynamic state relevant to a reset.

        Object joints are separated from robot/fixture joints so a
        construction comparison can permit object-pose differences without
        accidentally permitting robot-home drift.  Together with controls
        and simulation time, these mappings cover every qpos/qvel entry.
        """
        object_joint_owner = {}
        for body_name in self.free_bodies:
            body = self.model.body(body_name)
            first = int(body.jntadr[0])
            for joint_id in range(first, first + int(body.jntnum[0])):
                object_joint_owner[joint_id] = body_name

        object_states = {
            body_name: {
                "world_position_m": [
                    float(value) for value in self.body_pose(body_name)[0]
                ],
                "world_quaternion_wxyz": [
                    float(value) for value in self.body_pose(body_name)[1]
                ],
                "joints": {},
            }
            for body_name in sorted(self.free_bodies)
        }
        non_object_joint_states = {}
        for joint_id in range(self.model.njnt):
            joint = self.model.joint(joint_id)
            name = joint.name or f"__unnamed_joint_{joint_id}"
            qpos_width, qvel_width = self._joint_widths(
                self.model.jnt_type[joint_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            state = {
                "qpos": [float(value) for value in
                         self.data.qpos[qpos_adr:qpos_adr + qpos_width]],
                "qvel": [float(value) for value in
                         self.data.qvel[qvel_adr:qvel_adr + qvel_width]],
            }
            owner = object_joint_owner.get(joint_id)
            if owner is None:
                non_object_joint_states[name] = state
            else:
                object_states[owner]["joints"][name] = state

        actuator_controls = {}
        for actuator_id in range(self.model.nu):
            actuator = self.model.actuator(actuator_id)
            name = actuator.name or f"__unnamed_actuator_{actuator_id}"
            actuator_controls[name] = float(self.data.ctrl[actuator_id])
        return {
            "time_s": float(self.data.time),
            "robot_arm_qpos_rad": [
                float(value) for value in self.data.qpos[self._jadr]
            ],
            "robot_arm_qvel_rad_s": [
                float(value) for value in self.data.qvel[self._jdadr]
            ],
            "non_object_joint_states": non_object_joint_states,
            "object_states": object_states,
            "actuator_controls": actuator_controls,
        }

    # ------------------------------------------------------------- lifecycle
    def reset(self, settle_s=1.5, jitter_body=None, jitter_xy=0.0, rng=None,
              *, jitter_offset_xy=None, jitter_uniform_draw=None,
              reset_seed=None):
        self.last_reset_provenance = None
        mujoco.mj_resetData(self.model, self.data)
        home = self.info["home"]
        self.data.qpos[self._jadr] = home
        self.data.ctrl[self._aids] = home
        self.data.ctrl[self._grip_aid] = 0.0  # open
        # Forward once before perturbation so the recorded nominal body
        # transforms correspond exactly to the reset qpos, not stale xpos.
        mujoco.mj_forward(self.model, self.data)
        nominal_state = self._reset_state_snapshot()

        jitter_xy = float(jitter_xy)
        if not np.isfinite(jitter_xy) or jitter_xy < 0.0:
            raise ValueError("jitter_xy must be a finite non-negative value")
        if reset_seed is not None:
            if isinstance(reset_seed, (bool, np.bool_)):
                raise ValueError("reset_seed must be an integer")
            try:
                reset_seed = int(reset_seed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("reset_seed must be an integer") from exc
            if not 0 <= reset_seed <= np.iinfo(np.uint32).max:
                raise ValueError("reset_seed must be a uint32 value")

        draw_array = None
        if jitter_uniform_draw is not None:
            draw_array = np.asarray(jitter_uniform_draw, dtype=float)
            if (draw_array.shape != (2,) or
                    not np.isfinite(draw_array).all() or
                    np.any(draw_array < 0.0) or np.any(draw_array >= 1.0)):
                raise ValueError(
                    "jitter_uniform_draw must be two values in [0, 1)")
            if reset_seed is not None:
                seeded_draw = np.random.RandomState(reset_seed).random_sample(2)
                if not np.array_equal(draw_array, seeded_draw):
                    raise ValueError(
                        "jitter_uniform_draw differs from reset_seed replay")

        active_jitter = jitter_body is not None and jitter_xy > 0.0
        offset = np.zeros(2, dtype=float)
        if active_jitter:
            bid = self.model.body(jitter_body).id
            adr = self.model.jnt_qposadr[self.model.body(bid).jntadr[0]]
            if jitter_offset_xy is None:
                if draw_array is None:
                    rng = rng or np.random
                    draw_array = np.asarray(rng.random_sample(2), dtype=float)
                offset = (2.0 * draw_array - 1.0) * jitter_xy
            else:
                offset = np.asarray(jitter_offset_xy, dtype=float)
                if offset.shape != (2,) or not np.isfinite(offset).all():
                    raise ValueError("jitter_offset_xy must be two finite values")
                if np.any(np.abs(offset) > jitter_xy + 1e-15):
                    raise ValueError("jitter_offset_xy exceeds jitter_xy bound")
                if draw_array is not None:
                    expected_offset = (2.0 * draw_array - 1.0) * jitter_xy
                    if not np.array_equal(offset, expected_offset):
                        raise ValueError(
                            "jitter_offset_xy differs from jitter_uniform_draw")
            self.data.qpos[adr:adr + 2] += offset
        elif jitter_offset_xy is not None:
            offset_array = np.asarray(jitter_offset_xy, dtype=float)
            if (offset_array.shape != (2,) or
                    not np.isfinite(offset_array).all() or
                    np.any(offset_array != 0.0)):
                raise ValueError(
                    "nonzero jitter_offset_xy supplied for a disabled jitter")
        mujoco.mj_forward(self.model, self.data)
        model_timestep = float(self.model.opt.timestep)
        settle_steps = _settle_step_count(settle_s, model_timestep)
        settle_s = float(settle_s)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        assert np.isfinite(self.data.qpos).all(), "NaN after settle"
        self.start_pose = {b: self.body_pose(b) for b in self.free_bodies}
        jitter = {
            "algorithm": "numpy.random.RandomState.random_sample_then_affine",
            "reset_seed": reset_seed,
            "body": str(jitter_body) if jitter_body is not None else None,
            "max_abs_xy_m": jitter_xy,
            "uniform_draw_0_1": (
                [float(value) for value in draw_array]
                if draw_array is not None else None),
            "offset_xy_m": [float(value) for value in offset],
            "applied": active_jitter,
        }
        settle_protocol = {
            "engine": "mujoco",
            "step_function": "mujoco.mj_step",
            "requested_duration_s": settle_s,
            "model_timestep_s": model_timestep,
            "step_count": settle_steps,
            "simulated_duration_s": float(settle_steps * model_timestep),
        }
        post_settle_state = self._reset_state_snapshot()
        provenance = {
            "schema_version": 1,
            "pre_jitter_nominal_state": nominal_state,
            "pre_jitter_nominal_state_sha256": canonical_hash(nominal_state),
            "jitter": jitter,
            "jitter_sha256": canonical_hash(jitter),
            "settle_protocol": settle_protocol,
            "settle_protocol_sha256": canonical_hash(settle_protocol),
            "post_settle_pre_policy_state": post_settle_state,
            "post_settle_pre_policy_state_sha256": canonical_hash(
                post_settle_state),
        }
        # Canonicalization also rejects any non-finite value that escaped the
        # simulator qpos assertion (for example qvel or actuator state).
        canonical_hash(provenance)
        self.last_reset_provenance = provenance
        return self.get_obs()

    # --------------------------------------------------------------- control
    def apply_action(self, action):
        """One 15 Hz tick: absolute jointpos target (7) + gripper [0,1]."""
        a = np.asarray(action, dtype=float)
        cur = self.data.qpos[self._jadr]
        tgt = cur + np.clip(a[:7] - cur, -rig.MAX_JOINT_DELTA,
                            rig.MAX_JOINT_DELTA)
        self.data.ctrl[self._aids] = np.clip(tgt, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[self._grip_aid] = 255.0 if a[7] > 0.5 else 0.0
        for _ in range(rig.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------------- state
    def joint_position(self):
        return self.data.qpos[self._jadr].copy()

    def gripper_position(self):
        # driver joint range is [0, 0.8] rad; 0.8 ~= fully closed
        return np.array([np.clip(self.data.qpos[self._grip_jadr] / 0.8, 0, 1)])

    def body_pose(self, name):
        bid = self.model.body(name).id
        return (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())

    def body_vel(self, name):
        bid = self.model.body(name).id
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        return v  # [ang(3), lin(3)]

    def render_cam(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render()

    def get_obs(self):
        render = (self.composite.render if self.composite is not None
                  else self.render_cam)
        return {
            "observation/exterior_image_1_left": render(
                self.info["ext_cam"]),
            "observation/wrist_image_left": render(
                self.info["wrist_cam"]),
            "observation/joint_position": self.joint_position(),
            "observation/gripper_position": self.gripper_position(),
        }

    # ------------------------------------------------------------ predicates
    def pad_contacts(self, body_name):
        """Which pad sides are in contact with the body's geoms."""
        geoms = self._body_geoms[body_name]
        sides = set()
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            for side in ("left", "right"):
                pads = self.pad_geoms[side]
                if ((c.geom1 in pads and c.geom2 in geoms) or
                        (c.geom2 in pads and c.geom1 in geoms)):
                    sides.add(side)
        return sides

    def grasped(self, body_name):
        return len(self.pad_contacts(body_name)) == 2

    def lifted(self, body_name, min_dz=0.05):
        z0 = self.start_pose[body_name][0][2]
        return self.data.xpos[self.model.body(body_name).id][2] > z0 + min_dz

    def in_region(self, body_name, region, z_band=0.30):
        p = self.data.xpos[self.model.body(body_name).id]
        return (abs(p[0] - region["cx"]) <= region["hx"] and
                abs(p[1] - region["cy"]) <= region["hy"] and
                region["z"] - 0.05 <= p[2] <= region["z"] + z_band)

    def at_rest(self, body_name, max_lin=0.05):
        return float(np.linalg.norm(self.body_vel(body_name)[3:])) < max_lin
