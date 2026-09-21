"""CPU tests for physicalview.ik on a Panda-only MuJoCo model (no viser, no GPU).

Model: third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml (a symlink to the
main checkout when the worktree lacks the gitignored menagerie). End effector: the
``attachment_site`` on the flange (the rig attaches the 2F-85 there).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from physicalview import ik  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda_nohand.xml",
    Path("/group/worldcept/code/SimAny/third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml"),
]
PANDA_XML = next((p for p in _CANDIDATES if p.exists()), None)
EE = "attachment_site"
HOME = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])

pytestmark = pytest.mark.skipif(PANDA_XML is None, reason="menagerie panda_nohand.xml not found")


@pytest.fixture(scope="module")
def panda():
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    data = mujoco.MjData(model)
    names = [f"joint{i}" for i in range(1, 8)]
    jadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    data.qpos[jadr] = HOME
    mujoco.mj_forward(model, data)
    return model, data, jadr, dadr


def _fk(model, jadr, q):
    d = mujoco.MjData(model)
    d.qpos[jadr] = q
    mujoco.mj_forward(model, d)
    return ik.ee_pose(model, d, EE)


def test_ee_pose_matches_mujoco_site(panda):
    model, data, jadr, _ = panda
    pos, quat = ik.ee_pose(model, data, EE)
    sid = model.site(EE).id
    np.testing.assert_allclose(pos, data.site_xpos[sid])
    R = data.site_xmat[sid].reshape(3, 3)
    q_ref = np.empty(4)
    mujoco.mju_mat2Quat(q_ref, R.reshape(9))
    assert abs(abs(float(np.dot(quat, q_ref))) - 1.0) < 1e-9
    # body variant works as well
    bpos, bquat = ik.ee_pose(model, data, "link7")
    np.testing.assert_allclose(bpos, data.body("link7").xpos)


def test_fk_ik_round_trip_from_home(panda):
    model, data, jadr, dadr = panda
    lo, hi = ik.joint_limits_for_qpos(model, jadr)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    rng = np.random.RandomState(0)
    n_ok = 0
    trials = 12
    for _ in range(trials):
        # random joints in the inner 70% of each range (avoid self-folded configs)
        u = rng.uniform(0.15, 0.85, 7)
        q_true = lo + u * (hi - lo)
        pos, quat = _fk(model, jadr, q_true)
        qpos_before = data.qpos.copy()
        res = ik.solve_ik(model, data, EE, pos, quat, jadr, dadr, q_init=HOME, iters=300)
        np.testing.assert_array_equal(data.qpos, qpos_before)  # live data untouched
        assert np.all(np.isfinite(res.q))
        assert np.all(res.q >= lo - 1e-9) and np.all(res.q <= hi + 1e-9)
        if res.converged:
            n_ok += 1
            assert res.pos_err < 2e-3, res
            assert res.rot_err < 0.02, res
            # residuals must be the TRUE residuals at the returned q
            p2, q2 = _fk(model, jadr, res.q)
            assert np.linalg.norm(p2 - pos) < 2e-3
            assert np.linalg.norm(ik.rotation_error(quat, q2)) < 0.02
    # A single-start DLS may miss a few poses that need an elbow flip from home;
    # the arm must still reach the large majority of random reachable poses.
    assert n_ok >= int(0.75 * trials), f"only {n_ok}/{trials} random targets converged"


def test_position_only_ik(panda):
    model, data, jadr, dadr = panda
    pos0, _ = ik.ee_pose(model, data, EE)
    target = pos0 + np.array([0.10, -0.05, 0.05])
    res = ik.solve_ik(model, data, EE, target, None, jadr, dadr, q_init=HOME)
    assert res.converged and res.pos_err < 1e-3 and res.rot_err == 0.0
    p2, _ = _fk(model, jadr, res.q)
    assert np.linalg.norm(p2 - target) < 1e-3


def test_unreachable_target_does_not_raise(panda):
    model, data, jadr, dadr = panda
    target = np.array([2.5, 0.0, 0.5])  # far beyond the 0.855 m reach
    res = ik.solve_ik(model, data, EE, target, np.array([1.0, 0.0, 0.0, 0.0]), jadr, dadr,
                      q_init=HOME, iters=100)
    assert res.converged is False
    assert np.all(np.isfinite(res.q))
    assert res.pos_err > 1.0
    lo, hi = ik.joint_limits_for_qpos(model, jadr)
    assert np.all(res.q >= lo - 1e-9) and np.all(res.q <= hi + 1e-9)


def test_joint_limits_respected_and_optional(panda):
    model, data, jadr, dadr = panda
    lo, hi = ik.joint_limits_for_qpos(model, jadr)
    # Target below the base plane and behind the arm: pulls joints against limits.
    target = np.array([-0.4, 0.0, -0.3])
    res = ik.solve_ik(model, data, EE, target, None, jadr, dadr, q_init=HOME, iters=150)
    assert np.all(res.q >= lo - 1e-9) and np.all(res.q <= hi + 1e-9)
    # Start OUTSIDE the limits: the first thing the solver does is clamp.
    q_bad = hi + 0.5
    res2 = ik.solve_ik(model, data, EE, target, None, jadr, dadr, q_init=q_bad, iters=5)
    assert np.all(res2.q <= hi + 1e-9)
    # joint_limits=False may leave the range (joint4 has range [-3.07, -0.07])
    res3 = ik.solve_ik(model, data, EE, target, None, jadr, dadr, q_init=q_bad, iters=1,
                       joint_limits=False, step=0.0)
    assert np.any(res3.q > hi)


def test_rotation_error_helpers():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    ang = 0.3
    q_z = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
    rv = ik.rotation_error(q_z, q_id)
    np.testing.assert_allclose(rv, [0.0, 0.0, ang], atol=1e-12)
    # sign-flipped quaternion is the same rotation
    np.testing.assert_allclose(ik.rotation_error(-q_z, q_id), [0.0, 0.0, ang], atol=1e-12)
    assert np.linalg.norm(ik.rotation_error(q_z, q_z)) < 1e-12
    with pytest.raises(KeyError):
        m = mujoco.MjModel.from_xml_path(str(PANDA_XML))
        ik.resolve_target(m, "no_such_site")
