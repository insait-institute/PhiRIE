"""Damped-least-squares inverse kinematics for the Panda arm in a MuJoCo model.

CONTRACT (robot agent). Pure numpy + mujoco; unit-tested on a Panda-only model built
from third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml.

    solve_ik(model, data, site_or_body, target_pos, target_quat_wxyz | None,
             joint_qpos_adr (7,), joint_dof_adr (7,), *, q_init=None, iters=100,
             damping=1e-2, step=0.5, pos_tol=1e-3, rot_tol=1e-2, joint_limits=True)
        -> IKResult(q (7,), pos_err, rot_err, iters, converged)
    Uses mujoco.mj_jacSite / mj_jacBody for the 6xnv jacobian restricted to the 7 arm
    dofs, DLS update dq = J^T (J J^T + lambda^2 I)^-1 e, orientation error as rotation
    vector, joint limits clamped from model.jnt_range. Must not mutate `data` visibly:
    work on a copy of qpos and restore, or accept a scratch MjData.

    ee_pose(model, data, site_or_body) -> (pos (3,), quat_wxyz (4,))

Implementation notes (not part of the contract):
  * ``site_or_body`` is resolved as a site first, then as a body (so a rig can pass
    "robot/2f85/pinch" and the Panda-only test model can pass "attachment_site").
  * All work happens on a scratch ``mujoco.MjData`` (``scratch=`` keyword, else one is
    allocated per call): the live ``data`` is only read (qpos copied so the other joints —
    gripper, free objects — keep their current values), never written or re-forwarded.
  * The orientation error is the world-frame rotation vector of
    ``q_target * conj(q_current)`` so it lives in the same frame as ``mj_jacSite``'s
    rotational jacobian. Per-iteration steps are capped at ``max_step_rad`` so a badly
    conditioned jacobian near a singularity cannot throw the solution across the
    workspace.
  * ``pos_err``/``rot_err`` in the result are re-evaluated at the returned ``q`` (after
    clamping), so they are the true residuals, not the last iterate's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # mujoco is a hard dependency of everything that calls solve_ik
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    iters: int
    converged: bool


# ---------------------------------------------------------------- quaternions ----

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, angle in [0, pi]) of a unit wxyz quaternion."""
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:  # shortest arc
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[0])
    return v / s * angle


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion (via mujoco for exact conventions)."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def rotation_error(target_quat_wxyz, current_quat_wxyz) -> np.ndarray:
    """World-frame rotation vector w such that exp(w) * q_cur = q_target."""
    q_t = np.asarray(target_quat_wxyz, dtype=float)
    q_c = np.asarray(current_quat_wxyz, dtype=float)
    q_t = q_t / max(np.linalg.norm(q_t), 1e-12)
    q_c = q_c / max(np.linalg.norm(q_c), 1e-12)
    return quat_to_rotvec(quat_mul(q_t, quat_conj(q_c)))


# ------------------------------------------------------------------- resolve -----

def resolve_target(model, site_or_body: str) -> tuple[str, int]:
    """Return ("site" | "body", id) for a name; sites take precedence over bodies."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_or_body)
    if sid >= 0:
        return "site", int(sid)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_or_body)
    if bid >= 0:
        return "body", int(bid)
    raise KeyError(f"no site or body named {site_or_body!r} in the model")


def _pose_of(model, data, kind: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "site":
        return data.site_xpos[idx].copy(), mat_to_quat(data.site_xmat[idx])
    return data.xpos[idx].copy(), data.xquat[idx].copy()


def _jac_of(model, data, kind: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if kind == "site":
        mujoco.mj_jacSite(model, data, jacp, jacr, idx)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, idx)
    return jacp, jacr


def joint_limits_for_qpos(model, joint_qpos_adr) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) per requested qpos address from model.jnt_range; +-inf when unlimited."""
    adr = np.asarray(joint_qpos_adr, dtype=int)
    lo = np.full(adr.shape, -np.inf)
    hi = np.full(adr.shape, np.inf)
    by_adr = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
    for k, a in enumerate(adr):
        j = by_adr.get(int(a))
        if j is None or not model.jnt_limited[j]:
            continue
        lo[k], hi[k] = model.jnt_range[j]
    return lo, hi


# ----------------------------------------------------------------------- API -----

def ee_pose(model, data, site_or_body: str):
    """(pos (3,), quat_wxyz (4,)) of a site or body from the CURRENT data (no forward)."""
    kind, idx = resolve_target(model, site_or_body)
    return _pose_of(model, data, kind, idx)


def solve_ik(model, data, site_or_body: str, target_pos, target_quat_wxyz,
             joint_qpos_adr, joint_dof_adr, *, q_init=None, iters: int = 100,
             damping: float = 1e-2, step: float = 0.5, pos_tol: float = 1e-3,
             rot_tol: float = 1e-2, joint_limits: bool = True,
             scratch=None, max_step_rad: float = 0.5) -> IKResult:
    """Damped least squares IK over the given arm dofs; see module docstring.

    ``target_quat_wxyz=None`` solves position only (rot_err reported as 0.0).
    Never raises for unreachable targets: returns ``converged=False`` with the best
    (finite, limit-respecting) iterate found.
    """
    kind, idx = resolve_target(model, site_or_body)
    jadr = np.asarray(joint_qpos_adr, dtype=int)
    dadr = np.asarray(joint_dof_adr, dtype=int)
    n = len(jadr)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    use_rot = target_quat_wxyz is not None
    if use_rot:
        target_quat = np.asarray(target_quat_wxyz, dtype=float).reshape(4)
        target_quat = target_quat / max(np.linalg.norm(target_quat), 1e-12)

    if scratch is None:
        scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos  # other joints (gripper, objects) stay where they are
    q = (np.array(q_init, dtype=float).reshape(n) if q_init is not None
         else data.qpos[jadr].astype(float).copy())
    lo, hi = joint_limits_for_qpos(model, jadr)
    if joint_limits:
        q = np.clip(q, lo, hi)

    def fk(qv: np.ndarray):
        scratch.qpos[jadr] = qv
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)  # cdof for the jacobians
        return _pose_of(model, scratch, kind, idx)

    def errors(pos, quat):
        e_p = target_pos - pos
        e_r = rotation_error(target_quat, quat) if use_rot else np.zeros(3)
        return e_p, e_r

    best_q, best_cost = q.copy(), np.inf
    best_errs = (np.inf, np.inf)
    it = 0
    converged = False
    for it in range(1, iters + 1):
        pos, quat = fk(q)
        e_p, e_r = errors(pos, quat)
        pe, re_ = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
        cost = pe + 0.1 * re_
        if cost < best_cost:
            best_q, best_cost, best_errs = q.copy(), cost, (pe, re_)
        if pe < pos_tol and (not use_rot or re_ < rot_tol):
            converged = True
            break
        jacp, jacr = _jac_of(model, scratch, kind, idx)
        if use_rot:
            J = np.vstack([jacp[:, dadr], jacr[:, dadr]])
            e = np.concatenate([e_p, e_r])
        else:
            J = jacp[:, dadr]
            e = e_p
        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, e)
        dq *= step
        mx = float(np.max(np.abs(dq))) if dq.size else 0.0
        if mx > max_step_rad:
            dq *= max_step_rad / mx
        if not np.all(np.isfinite(dq)):
            break
        q = q + dq
        if joint_limits:
            q = np.clip(q, lo, hi)

    # Re-evaluate at the returned solution so the reported residuals are exact.
    if converged:
        q_out = q
    else:
        q_out = best_q
    pos, quat = fk(q_out)
    e_p, e_r = errors(pos, quat)
    pe, re_ = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
    converged = bool(pe < pos_tol and (not use_rot or re_ < rot_tol))
    if not np.all(np.isfinite(q_out)):  # pragma: no cover - defensive
        q_out = np.clip(np.nan_to_num(q_out), lo, hi)
    return IKResult(q=np.asarray(q_out, dtype=float), pos_err=pe, rot_err=re_,
                    iters=it, converged=converged)


__all__ = ["IKResult", "ee_pose", "solve_ik", "resolve_target", "rotation_error",
           "joint_limits_for_qpos", "quat_mul", "quat_conj", "quat_to_rotvec", "mat_to_quat"]
