"""CPU tests for physicalview.streaming (server-render display mode).

* viser camera -> (w2c, K) conversion against render.look_at_w2c / project_points
* stream frame sizing (caps, aspect, moving scale, even sizes)
* MuJoCo free-camera round trip on a tiny model (azimuth/elevation/lookat/fovy)
* debounce + coalescing of the per-client render loop with a fake client and clock
* ServerRenderStream without a renderer logs once and pushes nothing; with a fake
  renderer it attaches, streams, honours show_objects and clears on stop
No viser, no GPU: viser is never imported by physicalview.streaming.
"""
from __future__ import annotations

import logging
import threading
import time
import types

import numpy as np
import pytest

from physicalview import render as R
from physicalview import streaming as ST
from physicalview.config import load_config
from physicalview.splats import matrix_to_quat_wxyz


def _viser_cam_from_w2c(w2c: np.ndarray):
    c2w = np.linalg.inv(w2c)
    return c2w[:3, 3].copy(), matrix_to_quat_wxyz(c2w[:3, :3])


def _wait(cond, timeout: float = 3.0, step: float = 0.01) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return cond()


# ------------------------------------------------------------------ camera math --

@pytest.mark.parametrize("eye,target", [
    ([2.0, -3.0, 1.5], [0.0, 0.0, 0.5]),
    ([-1.0, 4.0, 2.0], [0.3, 0.2, 0.9]),
    ([0.5, 0.5, 3.0], [0.5, 0.6, 0.0]),
])
def test_viser_camera_matches_look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    w2c_ref = R.look_at_w2c(eye, target)
    position, wxyz = _viser_cam_from_w2c(w2c_ref)
    fov = np.radians(50.0)
    W, H = 1280, 720
    w2c, K = ST.viser_camera_to_w2c_K(position, wxyz, fov, (W, H))
    assert np.allclose(w2c, w2c_ref, atol=1e-9)
    f = (H / 2) / np.tan(fov / 2)
    assert np.allclose(K, [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    # look-at target on the principal point; world-up above it -> smaller v
    assert np.allclose(R.project_points(w2c, K, target[None])[0], [W / 2, H / 2], atol=1e-6)
    above = R.project_points(w2c, K, (target + [0, 0, 0.2])[None])[0]
    assert above[1] < H / 2
    # vertical fov: a point on the top image edge at depth d sits d*tan(fov/2) along camera -Y
    c2w = np.linalg.inv(w2c)
    d = 2.0
    p_top = c2w[:3, 3] + d * c2w[:3, 2] - d * np.tan(fov / 2) * c2w[:3, 1]
    assert R.project_points(w2c, K, p_top[None])[0][1] == pytest.approx(0.0, abs=1e-6)
    p_right = c2w[:3, 3] + d * c2w[:3, 2] + d * np.tan(fov / 2) * (W / H) * c2w[:3, 0]
    assert R.project_points(w2c, K, p_right[None])[0][0] == pytest.approx(W, abs=1e-6)


def test_viser_camera_rejects_bad_fov():
    with pytest.raises(ValueError):
        ST.viser_camera_to_w2c_K([0, 0, 0], [1, 0, 0, 0], 0.0, (64, 64))
    with pytest.raises(ValueError):
        ST.viser_camera_to_w2c_K([0, 0, 0], [1, 0, 0, 0], np.pi, (64, 64))


def test_quat_round_trip_with_splats_helpers():
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        Rm = ST.quat_to_matrix(q)
        q2 = matrix_to_quat_wxyz(Rm)
        assert np.allclose(ST.quat_to_matrix(q2), Rm, atol=1e-9)


def test_stream_size_caps_aspect_and_scale():
    assert ST.stream_size(1920, 16 / 9, 1280) == (1280, 720)
    assert ST.stream_size(1024, 16 / 9, 1280) == (1024, 576)          # narrower canvas than the cap
    assert ST.stream_size(1920, 16 / 9, 1280, scale=0.5) == (640, 360)  # moving scale
    assert ST.stream_size(1920, 16 / 9, None) == (1920, 1080)          # native
    assert ST.stream_size(1920, 16 / 9, 1920) == (1920, 1080)
    assert ST.stream_size(0, 0.0, 1280) == (1280, 720)                 # unknown canvas / aspect
    w, h = ST.stream_size(1001, 1.0, None)
    assert w % 2 == 0 and h % 2 == 0 and w == h == 1000
    assert ST.stream_size(5, 1.0, None) == (16, 16)


def test_settings_from_config_defaults():
    cfg = load_config()
    s = ST.StreamSettings.from_config(cfg)
    assert (s.max_width, s.jpeg_quality, s.max_fps, s.moving_scale) == (1280, 80, 15.0, 0.5)
    assert cfg.display_mode == "server"
    s2 = ST.StreamSettings.from_config(types.SimpleNamespace(stream={"max_width": 0, "jpeg_quality": 200,
                                                                    "max_fps": 0, "moving_scale": 5}))
    assert s2.max_width is None and s2.jpeg_quality == 100 and s2.max_fps == 0.1 and s2.moving_scale == 1.0


# ------------------------------------------------------------- MuJoCo free camera --

def test_free_camera_params_invert_forward_formula():
    for az, el in [(30.0, -20.0), (-120.0, 45.0), (90.0, -45.0), (179.0, 5.0)]:
        f = ST.forward_from_azimuth_elevation(az, el)
        c2w = np.eye(4)
        z = f / np.linalg.norm(f)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, np.cross(z, x), z, [1.0, 2.0, 3.0]
        lookat, az2, el2, d = ST.free_camera_params(np.linalg.inv(c2w), distance=0.7)
        assert d == 0.7
        assert np.allclose(lookat, [1.0, 2.0, 3.0] + 0.7 * z)
        assert (az2 - az + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)
        assert el2 == pytest.approx(el, abs=1e-9)


def test_free_camera_round_trip_mujoco():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    scn = mujoco.MjvScene(m, maxgeom=10)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    for eye, target in [([2.0, -3.0, 1.5], [0.0, 0.0, 0.5]), ([-1.0, 4.0, 2.0], [0.3, 0.2, 0.9]),
                        ([0.5, 0.5, 3.0], [0.5, 0.6, 0.0]), ([1.0, 1.0, 0.2], [0.0, 0.0, 1.5])]:
        w2c = R.look_at_w2c(np.asarray(eye), np.asarray(target))
        ST.apply_free_camera(cam, w2c, distance=0.8)
        assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        c2w = np.linalg.inv(w2c)
        fwd = np.asarray(scn.camera[0].forward)
        up = np.asarray(scn.camera[0].up)
        pos = (np.asarray(scn.camera[0].pos) + np.asarray(scn.camera[1].pos)) / 2   # stereo pair
        assert np.allclose(fwd, c2w[:3, 2], atol=1e-6)      # OpenCV +Z forward
        assert np.allclose(up, -c2w[:3, 1], atol=1e-6)      # OpenCV +Y down (no roll, world +Z up)
        assert np.allclose(pos, c2w[:3, 3], atol=1e-6)
    # the vertical fov of a free camera comes from model.vis.global_.fovy
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 180.0], [0, 0, 1]])
    fovy = ST.vertical_fov_deg(K, 360)
    assert fovy == pytest.approx(np.degrees(2 * np.arctan(180 / 500)))
    m.vis.global_.fovy = fovy
    mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    c0 = scn.camera[0]
    assert c0.frustum_top / c0.frustum_near == pytest.approx(np.tan(np.radians(fovy) / 2), rel=1e-5)


def test_mask_from_segmentation_geom_type_filter():
    mujoco = pytest.importorskip("mujoco")
    seg = np.zeros((4, 5, 2), np.int32)
    seg[..., 0] = np.arange(20).reshape(4, 5)
    seg[..., 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    seg[0, :, 1] = int(mujoco.mjtObj.mjOBJ_BODY)      # first row: not geoms
    mask = ST.mask_from_segmentation(seg, [1, 2, 7, 19])
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert mask[0].sum() == 0                          # ids 1,2 are in row 0 but typed as bodies
    assert mask[1, 2] == 255 and mask[3, 4] == 255 and mask[2, 2] == 0
    assert np.array_equal(ST.mask_from_segmentation(seg[..., 0], [7]) > 0, seg[..., 0] == 7)


# ------------------------------------------------------------- fake viser client --

class _FakeScene:
    def __init__(self) -> None:
        self.calls: list = []

    def set_background_image(self, image, format="auto", jpeg_quality=None, **_kw):
        self.calls.append((None if image is None else tuple(image.shape), format, jpeg_quality))


class _FakeCamera:
    def __init__(self) -> None:
        w2c = R.look_at_w2c(np.array([2.0, -3.0, 1.5]), np.array([0.0, 0.0, 0.5]))
        self.position, self.wxyz = _viser_cam_from_w2c(w2c)
        self.fov = float(np.radians(60.0))
        self.aspect = 16 / 9
        self.image_width, self.image_height = 1600, 900
        self.callbacks: list = []

    def on_update(self, cb):
        self.callbacks.append(cb)
        return cb


class _FakeClient:
    def __init__(self, cid: int = 7) -> None:
        self.client_id = cid
        self.scene = _FakeScene()
        self.camera = _FakeCamera()


def _render_fn(record: list):
    def fn(client, scale):
        record.append(scale)
        w, h = ST.stream_size(client.camera.image_width, client.camera.aspect, 1280, scale)
        return np.full((h, w, 3), 128, np.uint8)
    return fn


def test_debounce_moving_scale_then_full_frame_and_coalescing():
    clock = {"t": 0.0}
    settings = ST.StreamSettings(max_fps=15, moving_scale=0.5, settle_s=0.15, jpeg_quality=77)
    client = _FakeClient()
    scales: list = []
    cs = ST._ClientStream(client, _render_fn(scales), settings, clock=lambda: clock["t"])
    assert cs.step() is False                                  # nothing pending
    cs.camera_changed()
    assert cs.step() is True and scales == [0.5]               # moving -> half resolution
    assert client.scene.calls[-1] == ((360, 640, 3), "jpeg", 77)
    assert cs.wait_timeout(clock["t"]) == pytest.approx(0.15 + 1e-3)   # a full frame is owed
    clock["t"] = 0.05
    for _ in range(4):
        cs.camera_changed()                                    # burst: one pending render
    assert cs.step() is True and scales == [0.5, 0.5] and cs.coalesced == 3
    assert cs.step() is False                                  # still moving, nothing new
    clock["t"] = 0.05 + 0.10
    assert cs.step() is False                                  # not settled yet (< 150 ms)
    clock["t"] = 0.05 + 0.16
    assert cs.step() is True and scales[-1] == 1.0             # settled: final full-res frame
    assert client.scene.calls[-1][0] == (720, 1280, 3)
    assert cs.step() is False and cs.wait_timeout(clock["t"]) is None
    # scene change while idle -> exactly one full-res frame
    cs.invalidate()
    cs.invalidate()
    assert cs.step() is True and scales[-1] == 1.0 and cs.coalesced == 4
    assert cs.step() is False
    # scene change while the camera moves -> moving scale, then the owed full frame
    clock["t"] = 1.0
    cs.camera_changed()
    assert cs.step() and scales[-1] == 0.5
    cs.invalidate()
    assert cs.step() and scales[-1] == 0.5
    clock["t"] = 1.2
    assert cs.step() and scales[-1] == 1.0
    assert cs.frames == len(scales) == 7 and cs.errors == 0
    # moving_scale 1.0 disables the two-pass behaviour
    cs.settings.moving_scale = 1.0
    clock["t"] = 2.0
    cs.camera_changed()
    assert cs.step() and scales[-1] == 1.0 and cs.wait_timeout(2.0) is None


def test_render_errors_are_counted_not_raised(caplog):
    settings = ST.StreamSettings()
    client = _FakeClient()

    def boom(_client, _scale):
        raise RuntimeError("gpu hiccup")
    cs = ST._ClientStream(client, boom, settings, clock=lambda: 0.0)
    cs.invalidate()
    with caplog.at_level(logging.WARNING, logger="studio.stream"):
        assert cs.step() is False
    assert cs.errors == 1 and cs.frames == 0 and "gpu hiccup" in caplog.text
    cs2 = ST._ClientStream(client, lambda c, s: None, settings, clock=lambda: 0.0)   # renderer idle
    cs2.invalidate()
    assert cs2.step() is False and cs2.errors == 0 and client.scene.calls == []


def test_client_thread_respects_max_fps_and_settles():
    settings = ST.StreamSettings(max_fps=20, moving_scale=0.5, settle_s=0.25)
    client = _FakeClient()
    scales: list = []
    cs = ST._ClientStream(client, _render_fn(scales), settings)
    cs.start()
    t0 = time.monotonic()
    n_updates = 0
    while time.monotonic() - t0 < 0.4:
        cs.camera_changed()
        n_updates += 1
        time.sleep(0.005)
    assert _wait(lambda: scales and scales[-1] == 1.0, timeout=3.0)
    time.sleep(0.2)
    cs.stop()
    assert not cs.alive
    assert n_updates > cs.frames >= 1                     # frames were dropped/coalesced
    assert cs.frames <= 20 * 0.7 + 3                      # never above max_fps
    assert scales[-1] == 1.0 and all(s == 0.5 for s in scales[:-1])
    assert client.scene.calls[-1][0] == (720, 1280, 3)


# ----------------------------------------------------------- ServerRenderStream --

def _ctx(renderer=None, robot=None):
    from physicalview.app import Context
    from physicalview.gpu import GpuInfo
    ctx = Context(server=None, config=load_config(), gpu=GpuInfo(present=False), jobs=None)
    ctx.renderer = renderer
    ctx.robot = robot
    return ctx


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls: list = []
        self.state = types.SimpleNamespace(objects={}, edited_poses={})
        self.hidden: set = set()

    def render(self, w2c, K, wh, object_poses=None, background=None):
        self.calls.append((tuple(int(v) for v in wh), object_poses))
        return np.full((int(wh[1]), int(wh[0]), 3), 90, np.uint8)


def test_server_stream_without_renderer_logs_once_and_pushes_nothing(caplog):
    ctx = _ctx(None)
    stream = ST.ServerRenderStream(ctx, lambda: ctx.renderer, lambda: None, ctx.config)
    client = _FakeClient()
    with caplog.at_level(logging.WARNING, logger="studio.stream"):
        assert stream._render_client(client, 1.0) is None
        assert stream._render_client(client, 0.5) is None
    assert sum("no GPU renderer" in rec.message for rec in caplog.records) == 1
    with pytest.raises(R.RenderUnavailable):
        stream.render_for_camera(client.camera.position, client.camera.wxyz, client.camera.fov,
                                 client.camera.aspect, (64, 36))
    assert client.scene.calls == []


def test_server_stream_attach_events_show_objects_and_stop():
    fr = _FakeRenderer()
    ctx = _ctx(fr)
    stream = ST.ServerRenderStream(ctx, lambda: ctx.renderer, lambda: ctx.robot, ctx.config)
    assert stream.settings.max_width == 1280 and stream.jpeg_quality == 80
    stream.start()                                     # no viser server: subscribes events only
    client = _FakeClient(cid=3)
    cs = stream.attach_client(client)
    assert stream.attach_client(client) is cs          # idempotent
    assert client.camera.callbacks == [cs.camera_changed]
    assert _wait(lambda: client.scene.calls)           # first frame right after attach
    shape, fmt, q = client.scene.calls[0]
    assert shape == (720, 1280, 3) and fmt == "jpeg" and q == 80   # 1600 px canvas capped to 1280
    assert fr.calls[-1] == ((1280, 720), None)         # objects at rest/edited poses
    n = len(client.scene.calls)
    ctx.events.publish("robot.tick", {"body_poses": {}})           # a scene topic -> re-render
    assert _wait(lambda: len(client.scene.calls) > n)
    stream.show_objects = False
    stream.jpeg_quality = 60
    stream.max_width = 640
    n = len(client.scene.calls)
    ctx.events.publish("display.invalidate")
    assert _wait(lambda: len(client.scene.calls) > n)
    assert client.scene.calls[-1] == ((360, 640, 3), "jpeg", 60)
    assert fr.calls[-1] == ((640, 360), {})            # background only
    # direct camera motion through the registered callback: half-res frame first
    n = len(client.scene.calls)
    client.camera.callbacks[0](client.camera)
    assert _wait(lambda: len(client.scene.calls) > n)
    assert client.scene.calls[n][0] == (180, 320, 3)
    assert _wait(lambda: client.scene.calls[-1][0] == (360, 640, 3))   # settled full-res frame
    st = stream.stats()[3]
    assert st["frames"] >= 4 and st["wh"] == (640, 360) and st["errors"] == 0
    stream.stop()
    assert not cs.alive and not stream.active
    assert client.scene.calls[-1] == (None, "auto", None)             # background cleared
    stream.detach_client(client)                       # no-op after stop


def test_render_for_camera_matches_manual_conversion():
    fr = _FakeRenderer()
    ctx = _ctx(fr)
    stream = ST.ServerRenderStream(ctx, lambda: fr, lambda: None, ctx.config)
    cam = _FakeCamera()
    img = stream.render_for_camera(cam.position, cam.wxyz, cam.fov, cam.aspect, (320, 180))
    assert img.shape == (180, 320, 3)
    assert fr.calls == [((320, 180), None)]


def test_object_poses_follow_live_bodies():
    """With a robot session the objects follow MuJoCo bodies (frame_transform relative to
    the reset pose); objects without a body keep their edited/aligned pose."""
    from physicalview import splats as SP
    fr = _FakeRenderer()
    T1 = np.eye(4)
    T1[:3, 3] = [1.0, 0.0, 0.5]
    T2 = np.diag([2.0, 2.0, 2.0, 1.0])
    fr.state.objects = {"obj_01": types.SimpleNamespace(accepted=True, T_world=T1),
                        "obj_02": types.SimpleNamespace(accepted=True, T_world=T2),
                        "obj_03": types.SimpleNamespace(accepted=False, T_world=T1)}
    fr.state.edited_poses = {"obj_02": np.diag([3.0, 3.0, 3.0, 1.0])}
    p0, q0 = np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])
    p1 = np.array([1.2, 0.1, 0.5])

    class Sess:
        closed = False

        def live_body_poses(self):
            return {"obj_01": (p1, q0)}, {"obj_01": (p0, q0)}
    ctx = _ctx(fr, Sess())
    stream = ST.ServerRenderStream(ctx, lambda: fr, lambda: ctx.robot, ctx.config)
    stream.show_robot = False                          # no MuJoCo pass (needs EGL); poses only
    poses = stream._object_poses(fr, ctx.robot, None)
    assert set(poses) == {"obj_01", "obj_02"}
    assert np.allclose(poses["obj_01"], SP.frame_transform(T1, p1, q0, p0, q0))
    assert np.allclose(poses["obj_01"][:3, 3], p1)
    assert np.allclose(poses["obj_02"], fr.state.edited_poses["obj_02"])
    stream.show_objects = False
    assert stream._object_poses(fr, ctx.robot, None) == {}
