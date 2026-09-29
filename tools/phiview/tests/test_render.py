"""CPU parts of physicalview.render: camera conversion (against MuJoCo's own
renderer), look-at helper, compositing rule, and the RenderUnavailable contract."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from physicalview import render as R

_XML = """
<mujoco>
  <worldbody>
    <camera name="cam" pos="0.3 -2 1.2" xyaxes="1 0.1 0 -0.05 0.5 1" fovy="55"/>
    <camera name="tilted" pos="1.5 1.0 2.0" quat="0.85 -0.35 0.2 0.35" fovy="70"/>
    <geom name="ball" type="sphere" size="0.02" pos="0.2 0.1 0.3" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


@pytest.mark.parametrize("cam", ["cam", "tilted"])
def test_mujoco_camera_axes_and_intrinsics(mj, cam):
    mujoco, model, data = mj
    W, H = 320, 180
    w2c, K = R.mujoco_camera_to_w2c_K(model, data, cam, (W, H))
    cid = model.camera(cam).id
    xmat = data.cam_xmat[cid].reshape(3, 3)
    c2w = np.linalg.inv(w2c)
    assert np.allclose(c2w[:3, 3], data.cam_xpos[cid])
    # OpenCV forward (+z) == MuJoCo forward (-z of the camera frame); OpenCV down (+y) == -y
    assert np.allclose(c2w[:3, 2], -xmat[:, 2], atol=1e-12)
    assert np.allclose(c2w[:3, 1], -xmat[:, 1], atol=1e-12)
    assert np.allclose(c2w[:3, 0], xmat[:, 0], atol=1e-12)
    assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-12)
    fy = H / (2 * np.tan(np.radians(model.cam_fovy[cid]) / 2))
    assert K[0, 0] == pytest.approx(fy) and K[1, 1] == pytest.approx(fy)
    assert K[0, 2] == W / 2 and K[1, 2] == H / 2 and K[2, 2] == 1
    # analytic projection through the MuJoCo camera frame (looks along -z, +y up)
    p = np.array([0.2, 0.1, 0.3])
    pc = xmat.T @ (p - data.cam_xpos[cid])
    assert pc[2] < 0  # in front of the camera
    u_exp = W / 2 + fy * pc[0] / (-pc[2])
    v_exp = H / 2 - fy * pc[1] / (-pc[2])
    uv = R.project_points(w2c, K, p[None])[0]
    assert np.allclose(uv, [u_exp, v_exp], atol=1e-9)


@pytest.mark.parametrize("cam", ["cam", "tilted"])
@pytest.mark.gpu
def test_mujoco_camera_matches_renderer_pixels(mj, cam):
    """Project the sphere centre with our (w2c, K) and compare with the centroid of the
    sphere's pixels in MuJoCo's own segmentation render."""
    mujoco, model, data = mj
    W, H = 320, 240
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        renderer = mujoco.Renderer(model, height=H, width=W)
    except Exception as exc:  # noqa: BLE001 - no offscreen GL on this node
        pytest.skip(f"mujoco offscreen rendering unavailable: {exc}")
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
    finally:
        try:
            renderer.close()
        except Exception:  # noqa: BLE001
            pass
    gid = model.geom("ball").id
    ys, xs = np.nonzero(seg[..., 0] == gid)
    assert len(xs) > 3, "sphere not visible in the MuJoCo render"
    centroid = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    w2c, K = R.mujoco_camera_to_w2c_K(model, data, cam, (W, H))
    uv = R.project_points(w2c, K, np.array([[0.2, 0.1, 0.3]]))[0]
    assert np.allclose(uv, centroid, atol=1.5), f"{uv} vs render centroid {centroid}"


def test_mujoco_conversion_matches_pi05_render_formula(mj):
    """Bit-for-bit the same math as robo/rendering/pi05_render.py::_cam_K_w2c."""
    mujoco, model, data = mj
    W, H = 640, 360
    cid = model.camera("tilted").id
    fovy = np.radians(model.cam_fovy[cid])
    fy = H / (2 * np.tan(fovy / 2))
    K_ref = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
    c2w_gl = np.eye(4)
    c2w_gl[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
    c2w_gl[:3, 3] = data.cam_xpos[cid]
    w2c_ref = np.linalg.inv(c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0]))
    w2c, K = R.mujoco_camera_to_w2c_K(model, data, "tilted", (W, H))
    assert np.allclose(w2c, w2c_ref, atol=1e-12) and np.allclose(K, K_ref)


def test_look_at_w2c_conventions():
    eye, target = np.array([2.0, -3.0, 1.5]), np.array([0.0, 0.0, 0.5])
    w2c = R.look_at_w2c(eye, target)
    K = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1]])
    assert np.allclose(R.project_points(w2c, K, target[None])[0], [64, 48])   # principal point
    above = R.project_points(w2c, K, (target + [0, 0, 0.2])[None])[0]
    assert above[1] < 48                                                       # world up -> smaller v
    c2w = np.linalg.inv(w2c)
    assert np.allclose(c2w[:3, 3], eye)
    assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(w2c[:3, :3]) == pytest.approx(1.0)


def test_render_unavailable_without_cuda_or_bg(tmp_path):
    torch = pytest.importorskip("torch")
    from physicalview.scene_state import ResultSet, SceneState
    rs = ResultSet(name="x", scene_id="x", kind="other", out_dir=tmp_path, scene_dir=None,
                   splat_ply=None, mesh_ply=None)
    st = SceneState(result_set=rs, K=None, W=0, H=0, cameras={}, splat_gs=None, clean_bg_gs=None,
                    mesh=None, objects={}, tasks=None, scene_xml=None, sim_export=tmp_path,
                    timings={}, report=None)
    if not torch.cuda.is_available():
        with pytest.raises(R.RenderUnavailable, match="cuda"):
            R.Renderer(st)
    else:
        # GPU present: the missing background must still surface as RenderUnavailable
        with pytest.raises(R.RenderUnavailable):
            R.Renderer(st)
    assert issubclass(R.RenderUnavailable, RuntimeError)


def test_composite_rule_sim_over_splat(monkeypatch):
    """composite_with_robot copies robot pixels (mask>0) over the splat render and
    resizes the splat to the raster when sizes differ (pi05_render.py rule)."""
    r = R.Renderer.__new__(R.Renderer)
    splat = np.zeros((20, 30, 3), dtype=np.uint8)
    splat[..., 2] = 200
    monkeypatch.setattr(r, "render", lambda w2c, K, wh, object_poses=None, background=None: splat, raising=False)
    robot = np.zeros((20, 30, 3), dtype=np.uint8)
    robot[..., 0] = 255
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 7:12] = 255
    out = r.composite_with_robot(np.eye(4), np.eye(3), (30, 20), {}, robot, mask)
    assert out.shape == (20, 30, 3)
    assert np.array_equal(out[mask > 0], robot[mask > 0])
    assert np.array_equal(out[mask == 0], splat[mask == 0])
    # size mismatch: the splat is resized to the raster
    robot_big = np.zeros((40, 60, 3), dtype=np.uint8)
    mask_big = np.zeros((40, 60), dtype=bool)
    mask_big[:4, :4] = True
    out2 = r.composite_with_robot(np.eye(4), np.eye(3), (30, 20), {}, robot_big, mask_big)
    assert out2.shape == (40, 60, 3) and out2[0, 0, 0] == 0 and out2[20, 30, 2] == 200


def test_project_points_basic():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    uv = R.project_points(np.eye(4), K, np.array([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0]]))
    assert np.allclose(uv, [[320, 240], [570, 115]])


REAL_XML = Path(__file__).resolve().parents[1] / "outputs" / "c50d2d1d42_factory" / "sim_export" / "scene.xml"


@pytest.mark.skipif(not REAL_XML.exists(), reason="exported scene.xml not available")
def test_real_exported_scene_cameras_convert(mj):
    mujoco, _, _ = mj
    try:
        model = mujoco.MjModel.from_xml_path(str(REAL_XML))
    except Exception as exc:  # noqa: BLE001 - assets referenced by the xml may be missing
        pytest.skip(f"scene.xml not loadable here: {exc}")
    if model.ncam == 0:
        pytest.skip("no cameras in exported scene")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for cid in range(model.ncam):
        name = model.camera(cid).name or cid
        w2c, K = R.mujoco_camera_to_w2c_K(model, data, name, (640, 360))
        assert np.isfinite(w2c).all() and np.isfinite(K).all()
        assert np.allclose(np.linalg.inv(w2c)[:3, 3], data.cam_xpos[cid])
