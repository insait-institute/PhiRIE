"""Server-side rendering display mode: gsplat renders streamed to the browser as JPEG
background images, so the browser never holds Gaussian data.

Why. The Scene tab used to send every Gaussian (up to millions of background splats plus
the per-object splats) to the browser via ``add_gaussian_splats`` and let WebGL render
them; laptops ran out of memory. In *server* mode the GPU renderer (``render.Renderer``)
renders exactly the view the browser camera looks at and the frame is pushed with
``client.scene.set_background_image(rgb, format="jpeg")``. The 3D scene graph then only
carries lightweight helpers (object frames, highlight box, gizmos, a camera frustum).

Contract
    ServerRenderStream(ctx, renderer_getter, robot_getter, config)
        start()             register viser client connect/disconnect hooks (once), attach
                            every connected client, subscribe to the scene events below
        stop()              stop the per-client threads and clear their background images
        invalidate()        request a re-render for every client (coalesced)
        clear_background()  remove the background image of every client
        render_for_camera(position, wxyz, fov, aspect, wh) -> uint8 RGB [H, W, 3]
                            the complete path (background + posed objects + robot
                            composite) without a browser; used by smoke.py and tests
        show_objects / show_robot / jpeg_quality / max_width (None = native width)
        stats() -> {client_id: {...}}
    Scene events that trigger a re-render (SCENE_TOPICS): scene.loaded,
    scene.objects_changed, scene.pose_edited, selection.changed, robot.tick,
    inpaint.version_selected, display.invalidate.

Per client (``_ClientStream``) there is ONE daemon render thread. Camera ``on_update``
callbacks and scene events only set flags and wake it, so at most one render is pending
per client and intermediate requests are dropped (coalesced). While the camera changed
within the last ``settle_s`` (150 ms) frames are rendered at ``moving_scale`` of the target
size; one final full-resolution frame follows once the camera settles. The loop never
exceeds ``max_fps``. Target size: min(camera.image_width, max_width) and the matching
height for the client's aspect ratio.

Camera math (tests/test_streaming.py checks it against render.look_at_w2c)
    viser cameras follow OpenCV conventions: ``wxyz`` is the camera-to-world rotation
    (+X right, +Y down, +Z forward) and ``position`` the camera origin in world, so
    w2c = inv([R(q) | p]) and K = [[f, 0, W/2], [0, f, H/2], [0, 0, 1]] with
    f = (H/2) / tan(fov/2) (viser's ``fov`` is the vertical field of view in radians).
    Robot pass: MuJoCo is rendered from the same view with a free camera
    (``mjCAMERA_FREE``: lookat = p + d*forward, distance d, azimuth/elevation from
    forward = (cos el cos az, cos el sin az, sin el)) and ``model.vis.global_.fovy`` set to
    the same vertical fov; roll is ignored, which is exact for viser's orbit camera
    (world +Z up, like MuJoCo's free camera). The robot mask comes from a segmentation
    render with the same camera (robot geoms = bodies named ``robot/...``), then
    ``Renderer.composite_with_robot`` puts robot pixels over the splat render. MuJoCo is
    only touched under the RobotSession lock (``session.locked()``).

Graceful degradation: without a GPU renderer (``ctx.renderer is None``) the stream logs
once and pushes nothing; the client-splat display mode stays available. viser and mujoco
are only imported inside methods, so the module imports in CPU test environments.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from physicalview import splats as SP

log = logging.getLogger("studio.stream")

SCENE_TOPICS: tuple[str, ...] = (
    "scene.loaded", "scene.objects_changed", "scene.pose_edited", "selection.changed",
    "robot.tick", "inpaint.version_selected", "display.invalidate",
)
RESOLUTION_CHOICES: dict[str, int | None] = {"720p": 1280, "1080p": 1920, "native": None}
_MIN_SIDE = 16
_MJ_RENDERER_CACHE = 4
_ERROR_LOG_PERIOD_S = 5.0


@dataclass
class StreamSettings:
    """Runtime stream parameters (mutable copies of config ``viewer.stream``)."""
    max_width: int | None = 1280
    jpeg_quality: int = 80
    max_fps: float = 15.0
    moving_scale: float = 0.5
    settle_s: float = 0.15          # camera considered "moving" this long after its last change
    robot_distance: float = 1.0     # free-camera lookat distance (any positive value works)

    @classmethod
    def from_config(cls, config) -> "StreamSettings":
        src = getattr(config, "stream", None)
        kw: dict[str, Any] = {}
        for key in ("max_width", "jpeg_quality", "max_fps", "moving_scale"):
            val = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
            if val is not None:
                kw[key] = val
        s = cls(**kw)
        s.max_width = int(s.max_width) if s.max_width else None
        s.jpeg_quality = int(np.clip(int(s.jpeg_quality), 1, 100))
        s.max_fps = max(float(s.max_fps), 0.1)
        s.moving_scale = float(np.clip(float(s.moving_scale), 0.1, 1.0))
        return s


# ----------------------------------------------------------------------- camera math --

def quat_to_matrix(wxyz) -> np.ndarray:
    return SP.quat_wxyz_to_matrix(np.asarray(wxyz, dtype=np.float64).reshape(1, 4))[0]


def viser_camera_to_w2c_K(position, wxyz, fov: float, wh) -> tuple[np.ndarray, np.ndarray]:
    """viser camera (OpenCV: wxyz = camera-to-world rotation, position = origin, vertical
    fov in radians) -> (OpenCV w2c 4x4, K 3x3) for an image of size wh=(W, H)."""
    W, H = int(wh[0]), int(wh[1])
    fov = float(fov)
    if not (0.0 < fov < np.pi) or not np.isfinite(fov):
        raise ValueError(f"vertical fov must be in (0, pi) radians, got {fov}")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_to_matrix(wxyz)
    c2w[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    f = (H / 2.0) / np.tan(fov / 2.0)
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.linalg.inv(c2w), K


def stream_size(image_width, aspect, max_width, scale: float = 1.0) -> tuple[int, int]:
    """(W, H) of a streamed frame: min(canvas width, max_width) scaled by ``scale``, height
    from the canvas aspect; both even and >= 16 px. Unknown canvas -> max_width (or 1280),
    unknown aspect -> 16:9."""
    w = int(image_width) if (image_width and image_width > 0) else int(max_width or 1280)
    if max_width:
        w = min(w, int(max_width))
    a = float(aspect) if (aspect and np.isfinite(aspect) and aspect > 0) else 16.0 / 9.0
    w = max(int(round(w * float(scale))), _MIN_SIDE)
    h = max(int(round(w / a)), _MIN_SIDE)
    return (w // 2) * 2, (h // 2) * 2


def vertical_fov_deg(K: np.ndarray, H: int) -> float:
    return float(np.degrees(2.0 * np.arctan(H / (2.0 * float(K[1, 1])))))


def forward_from_azimuth_elevation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """MuJoCo free-camera forward vector (engine_vis_visualize.c: mjv_updateCamera)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def free_camera_params(w2c: np.ndarray, distance: float = 1.0) -> tuple[np.ndarray, float, float, float]:
    """(lookat, azimuth_deg, elevation_deg, distance) of a MuJoCo free camera whose eye and
    forward direction match an OpenCV w2c (roll is dropped)."""
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    p = c2w[:3, 3]
    f = c2w[:3, 2] / (np.linalg.norm(c2w[:3, 2]) + 1e-12)
    d = float(distance) if distance and distance > 0 else 1.0
    azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))
    return p + d * f, azimuth, elevation, d


def apply_free_camera(cam, w2c: np.ndarray, distance: float = 1.0):
    """Configure a ``mujoco.MjvCamera`` as a free camera matching ``w2c``; returns it."""
    import mujoco
    lookat, az, el, d = free_camera_params(w2c, distance)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = lookat
    cam.distance = d
    cam.azimuth = az
    cam.elevation = el
    return cam


def mask_from_segmentation(seg: np.ndarray, geom_ids) -> np.ndarray:
    """uint8 {0,255} mask of the given geom ids from a MuJoCo segmentation render
    ([H,W,2] = (object id, object type) or a plain [H,W] id image). Same rule as
    robo.rendering.mujoco_masks.robot_mask_from_segmentation."""
    seg = np.asarray(seg)
    ids = np.asarray(sorted(int(g) for g in geom_ids), dtype=np.int64)
    if seg.ndim == 3 and seg.shape[2] >= 2:
        try:
            import mujoco
            geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = (seg[..., 1] == geom_type) & np.isin(seg[..., 0], ids)
        except ImportError:
            mask = np.isin(seg[..., 0], ids)
    elif seg.ndim == 2:
        mask = np.isin(seg, ids)
    else:
        raise ValueError(f"unexpected segmentation shape {seg.shape}")
    return mask.astype(np.uint8) * 255


def _resize(img: np.ndarray, wh: tuple[int, int], nearest: bool = False) -> np.ndarray:
    w, h = int(wh[0]), int(wh[1])
    if img.shape[1] == w and img.shape[0] == h:
        return img
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    except Exception:  # noqa: BLE001
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.NEAREST if nearest else Image.BILINEAR))


def camera_state(client) -> dict | None:
    """Snapshot of a viser client's camera, or None before its first camera message
    (viser asserts on reads until then)."""
    cam = client.camera
    try:
        st = {
            "position": np.asarray(cam.position, dtype=np.float64).reshape(3).copy(),
            "wxyz": np.asarray(cam.wxyz, dtype=np.float64).reshape(4).copy(),
            "fov": float(cam.fov), "aspect": float(cam.aspect),
            "image_width": int(cam.image_width), "image_height": int(cam.image_height),
        }
    except Exception:  # noqa: BLE001 - AssertionError before the first ViewerCameraMessage
        return None
    if not (0.0 < st["fov"] < np.pi) or not np.all(np.isfinite(st["position"])):
        return None
    return st


# --------------------------------------------------------------------- per client --

class _ClientStream:
    """One render thread per browser client; see the module docstring for the policy.
    ``render_fn(client, scale) -> uint8 RGB | None`` does the actual work. A ``clock``
    can be injected for deterministic tests (``step(now)`` drives one scheduling step)."""

    def __init__(self, client, render_fn: Callable[[Any, float], np.ndarray | None],
                 settings: StreamSettings, *, clock: Callable[[], float] = time.monotonic,
                 name: str = "client", on_pushed: Callable[[], None] | None = None) -> None:
        self.client = client
        self.settings = settings
        self.name = name
        self._render_fn = render_fn
        self._clock = clock
        self._on_pushed = on_pushed
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._cam_dirty = False
        self._scene_dirty = False
        self._pending_full = False
        self._last_cam_change = -np.inf
        self._thread: threading.Thread | None = None
        self._last_error_log = -np.inf
        self.frames = 0
        self.coalesced = 0          # requests merged into an already pending render
        self.errors = 0
        self.last_wh: tuple[int, int] | None = None
        self.last_scale: float | None = None
        self.last_ms: float | None = None
        self._push_times: deque[float] = deque(maxlen=30)

    # --- requests (any thread) ------------------------------------------------------
    def camera_changed(self, _handle=None) -> None:
        with self._lock:
            if self._cam_dirty:
                self.coalesced += 1
            self._cam_dirty = True
            self._last_cam_change = self._clock()
        self._wake.set()

    def invalidate(self) -> None:
        with self._lock:
            if self._scene_dirty:
                self.coalesced += 1
            self._scene_dirty = True
        self._wake.set()

    # --- scheduling -------------------------------------------------------------------
    def decide(self, now: float) -> float | None:
        """Render scale for a frame due now, or None when nothing is pending."""
        s = self.settings
        with self._lock:
            moving = (now - self._last_cam_change) < s.settle_s
            if self._cam_dirty or self._scene_dirty:
                self._cam_dirty = self._scene_dirty = False
                if moving and s.moving_scale < 1.0:
                    self._pending_full = True
                    return float(s.moving_scale)
                self._pending_full = False
                return 1.0
            if self._pending_full and not moving:
                self._pending_full = False
                return 1.0
            return None

    def wait_timeout(self, now: float) -> float | None:
        """How long the loop may sleep: until the camera settles when a full-resolution
        frame is owed, else indefinitely (None)."""
        with self._lock:
            if self._pending_full:
                return max(self.settings.settle_s - (now - self._last_cam_change), 0.0) + 1e-3
        return None

    def step(self, now: float | None = None) -> bool:
        """One scheduling step: render + push when a frame is due. True when pushed."""
        now = self._clock() if now is None else now
        scale = self.decide(now)
        if scale is None:
            return False
        return self._render_and_push(scale)

    def _render_and_push(self, scale: float) -> bool:
        t0 = self._clock()
        try:
            img = self._render_fn(self.client, scale)
        except Exception as exc:  # noqa: BLE001 - keep streaming after a bad frame
            self.errors += 1
            self._log_error(f"{self.name}: render failed: {type(exc).__name__}: {exc}")
            return False
        if img is None:
            return False
        try:
            self.client.scene.set_background_image(img, format="jpeg",
                                                   jpeg_quality=int(self.settings.jpeg_quality))
        except Exception as exc:  # noqa: BLE001 - client may be gone
            self.errors += 1
            self._log_error(f"{self.name}: push failed: {type(exc).__name__}: {exc}")
            return False
        self.frames += 1
        self.last_wh = (int(img.shape[1]), int(img.shape[0]))
        self.last_scale = float(scale)
        self.last_ms = (self._clock() - t0) * 1000.0
        self._push_times.append(time.monotonic())
        if self._on_pushed is not None:
            try:
                self._on_pushed()
            except Exception:  # noqa: BLE001
                pass
        return True

    def _log_error(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= _ERROR_LOG_PERIOD_S:
            self._last_error_log = now
            log.warning("%s\n%s", msg, traceback.format_exc(limit=3))

    def fps(self) -> float:
        if len(self._push_times) < 2:
            return 0.0
        span = self._push_times[-1] - self._push_times[0]
        return (len(self._push_times) - 1) / span if span > 0 else 0.0

    # --- thread -----------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"stream-{self.name}")
        self._thread.start()

    def stop(self, join: bool = True, timeout: float = 3.0) -> None:
        self._stopped.set()
        self._wake.set()
        t = self._thread
        if join and t is not None and t is not threading.current_thread():
            t.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        next_ok = 0.0
        while not self._stopped.is_set():
            now = self._clock()
            if now < next_ok:                       # max_fps limiter
                time.sleep(min(next_ok - now, 1.0))
                continue
            pushed = False
            try:
                pushed = self.step(now)
            except Exception:  # noqa: BLE001
                self._log_error(f"{self.name}: scheduling error")
            if pushed:
                next_ok = self._clock() + 1.0 / max(float(self.settings.max_fps), 0.1)
                continue
            timeout = self.wait_timeout(self._clock())
            self._wake.wait(timeout if timeout is not None else 1.0)
            self._wake.clear()


# ------------------------------------------------------------------------ the stream --

class ServerRenderStream:
    def __init__(self, ctx, renderer_getter: Callable[[], Any], robot_getter: Callable[[], Any],
                 config, settings: StreamSettings | None = None) -> None:
        self.ctx = ctx
        self._renderer = renderer_getter
        self._robot = robot_getter
        self.settings = settings or StreamSettings.from_config(config)
        self.show_objects = True
        self.show_robot = True
        self.on_frame: Callable[[dict], None] | None = None   # <= 1 Hz stats callback
        self._clients: dict[int, _ClientStream] = {}
        self._lock = threading.Lock()
        self._active = False
        self._hooked = False
        self._unsubscribe: list[Callable[[], None]] = []
        self._warned_no_renderer = False
        self._last_robot_error = -np.inf
        self._mj_lock = threading.Lock()
        self._mj: "OrderedDict[tuple, tuple[Any, Any]]" = OrderedDict()   # (id(model), w, h) -> (model, Renderer)
        self._free_cam = None
        self._last_report = 0.0

    # --- settings ---------------------------------------------------------------------
    @property
    def jpeg_quality(self) -> int:
        return int(self.settings.jpeg_quality)

    @jpeg_quality.setter
    def jpeg_quality(self, value: int) -> None:
        self.settings.jpeg_quality = int(np.clip(int(value), 1, 100))

    @property
    def max_width(self) -> int | None:
        return self.settings.max_width

    @max_width.setter
    def max_width(self, value: int | None) -> None:
        self.settings.max_width = int(value) if value else None

    @property
    def active(self) -> bool:
        return self._active

    # --- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        """Idempotent: hook client connect/disconnect once, attach connected clients,
        subscribe to the scene topics, and request a first frame."""
        self._active = True
        server = getattr(self.ctx, "server", None)
        if server is not None:
            if not self._hooked:
                self._hooked = True
                server.on_client_connect(self._on_connect)      # also fires for connected clients
                server.on_client_disconnect(self._on_disconnect)
            else:
                try:
                    for client in server.get_clients().values():
                        self.attach_client(client)
                except Exception as exc:  # noqa: BLE001
                    log.warning("stream: get_clients failed: %s", exc)
        events = getattr(self.ctx, "events", None)
        if events is not None and not self._unsubscribe:
            for topic in SCENE_TOPICS:
                self._unsubscribe.append(events.subscribe(topic, self._make_handler(topic)))
        self.invalidate()

    def stop(self) -> None:
        """Stop every client thread, clear their background images, free MuJoCo renderers.
        Hooks stay registered but ignore new clients until start() is called again."""
        self._active = False
        with self._lock:
            streams = list(self._clients.values())
            self._clients.clear()
        for cs in streams:
            cs.stop()
            self._clear_client(cs.client)
        self._close_mj_renderers()

    def _make_handler(self, topic: str):
        def handler(_payload) -> None:
            if topic == "scene.loaded":
                self._close_mj_renderers()   # a new scene means a new MuJoCo model
            self.invalidate()
        return handler

    def _on_connect(self, client) -> None:
        if self._active:
            self.attach_client(client)

    def _on_disconnect(self, client) -> None:
        self.detach_client(client)

    def attach_client(self, client) -> _ClientStream:
        cid = int(client.client_id)
        with self._lock:
            cs = self._clients.get(cid)
            if cs is not None:
                return cs
            cs = _ClientStream(client, self._render_client, self.settings, name=f"client{cid}",
                               on_pushed=self._report)
            self._clients[cid] = cs
        client.camera.on_update(cs.camera_changed)
        cs.start()
        cs.invalidate()   # first frame as soon as the camera is known
        log.info("stream: client %s attached (server render)", cid)
        return cs

    def detach_client(self, client) -> None:
        with self._lock:
            cs = self._clients.pop(int(client.client_id), None)
        if cs is not None:
            cs.stop(join=False)
            log.info("stream: client %s detached after %d frames", client.client_id, cs.frames)

    def invalidate(self) -> None:
        with self._lock:
            streams = list(self._clients.values())
        for cs in streams:
            cs.invalidate()

    def clear_background(self) -> None:
        with self._lock:
            streams = list(self._clients.values())
        for cs in streams:
            self._clear_client(cs.client)

    @staticmethod
    def _clear_client(client) -> None:
        try:
            client.scene.set_background_image(None)
        except Exception:  # noqa: BLE001 - client gone
            pass

    def stats(self) -> dict[int, dict]:
        with self._lock:
            streams = dict(self._clients)
        return {cid: {"frames": cs.frames, "coalesced": cs.coalesced, "errors": cs.errors,
                      "fps": round(cs.fps(), 1), "wh": cs.last_wh, "scale": cs.last_scale,
                      "ms": None if cs.last_ms is None else round(cs.last_ms, 1)}
                for cid, cs in streams.items()}

    def _report(self) -> None:
        if self.on_frame is None:
            return
        now = time.monotonic()
        if now - self._last_report < 1.0:
            return
        self._last_report = now
        try:
            self.on_frame(self.stats())
        except Exception:  # noqa: BLE001
            pass

    # --- rendering --------------------------------------------------------------------
    def _render_client(self, client, scale: float) -> np.ndarray | None:
        renderer = self._renderer()
        if renderer is None:
            if not self._warned_no_renderer:
                self._warned_no_renderer = True
                log.warning("stream: no GPU renderer (ctx.renderer is None) — server render idle; "
                            "switch the Scene tab to client splats if you need a picture")
            return None
        self._warned_no_renderer = False
        cam = camera_state(client)
        if cam is None:
            return None
        wh = stream_size(cam["image_width"], cam["aspect"], self.settings.max_width, scale)
        return self.render_for_camera(cam["position"], cam["wxyz"], cam["fov"], cam["aspect"], wh)

    def render_for_camera(self, position, wxyz, fov: float, aspect: float, wh) -> np.ndarray:
        """Full frame for a viser-convention camera at size wh (aspect is informational: the
        image size fixes K's principal point; fov is vertical)."""
        from physicalview.render import RenderUnavailable
        renderer = self._renderer()
        if renderer is None:
            raise RenderUnavailable("no GPU renderer for the server render stream")
        w2c, K = viser_camera_to_w2c_K(position, wxyz, fov, wh)
        return self.render_view(renderer, w2c, K, (int(wh[0]), int(wh[1])))

    def render_view(self, renderer, w2c: np.ndarray, K: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
        session = self._robot()
        if session is not None and getattr(session, "closed", False):
            session = None
        robot = None
        if session is not None and self.show_robot:
            try:
                robot = self._robot_pass(session, w2c, K, wh)
            except Exception as exc:  # noqa: BLE001 - robot overlay is best effort
                now = time.monotonic()
                if now - self._last_robot_error >= _ERROR_LOG_PERIOD_S:
                    self._last_robot_error = now
                    log.warning("stream: robot pass failed (%s: %s); rendering splats only\n%s",
                                type(exc).__name__, exc, traceback.format_exc(limit=3))
        poses = self._object_poses(renderer, session, robot)
        if robot is None:
            return renderer.render(w2c, K, wh, object_poses=poses)
        rgb, mask = robot[0], robot[1]
        return renderer.composite_with_robot(w2c, K, wh, poses, rgb, mask)

    def _object_poses(self, renderer, session, robot) -> dict[str, np.ndarray] | None:
        """None -> renderer rest/edited poses; {} -> background only; else live poses
        following the MuJoCo bodies (same rule as the scene panel's robot.tick handler)."""
        if not self.show_objects:
            return {}
        body_poses = reset = None
        if robot is not None:
            body_poses, reset = robot[2], robot[3]
        elif session is not None:
            try:
                body_poses, reset = session.live_body_poses()
            except Exception:  # noqa: BLE001
                body_poses = None
        if not body_poses:
            return None
        state = renderer.state
        poses: dict[str, np.ndarray] = {}
        for obj_id, rec in state.objects.items():
            if not rec.accepted or rec.T_world is None:
                continue
            base_T = state.edited_poses.get(obj_id, rec.T_world)
            if obj_id in body_poses:
                pos, quat = body_poses[obj_id]
                if reset and obj_id in reset:
                    p0, q0 = reset[obj_id]
                    poses[obj_id] = SP.frame_transform(base_T, pos, quat, p0, q0)
                else:
                    poses[obj_id] = SP.body_pose_transform(base_T, pos, quat)
            else:
                poses[obj_id] = np.asarray(base_T, dtype=np.float64)
        return poses

    # --- MuJoCo robot pass ----------------------------------------------------------------
    def _mj_renderer_for(self, model, w: int, h: int):
        """Cached mujoco.Renderer per (model, size); call under _mj_lock."""
        import mujoco
        key = (id(model), int(w), int(h))
        hit = self._mj.get(key)
        if hit is not None and hit[0] is model:
            self._mj.move_to_end(key)
            return hit[1]
        while len(self._mj) >= _MJ_RENDERER_CACHE:
            _, (_, old) = self._mj.popitem(last=False)
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        r = mujoco.Renderer(model, height=int(h), width=int(w))
        self._mj[key] = (model, r)
        return r

    def _close_mj_renderers(self) -> None:
        with self._mj_lock:
            items = list(self._mj.values())
            self._mj.clear()
        for _, r in items:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass

    def _robot_pass(self, session, w2c: np.ndarray, K: np.ndarray, wh: tuple[int, int]):
        """(robot_rgb, robot_mask, body_poses, reset_poses) rendered by MuJoCo from the same
        camera; everything touching the model/data runs under the session lock."""
        import mujoco
        W, H = int(wh[0]), int(wh[1])
        with session.locked():
            if getattr(session, "closed", False):
                return None
            model, data = session.model, session.data
            rw = min(W, int(model.vis.global_.offwidth))
            rh = min(H, int(model.vis.global_.offheight))
            if rw <= 0 or rh <= 0:
                return None
            if self._free_cam is None:
                self._free_cam = mujoco.MjvCamera()
            cam = apply_free_camera(self._free_cam, w2c, self.settings.robot_distance)
            old_fovy = float(model.vis.global_.fovy)
            with self._mj_lock:
                r = self._mj_renderer_for(model, rw, rh)
                model.vis.global_.fovy = vertical_fov_deg(K, H)
                try:
                    r.disable_segmentation_rendering()
                    r.update_scene(data, camera=cam)
                    rgb = np.array(r.render(), copy=True)
                    r.enable_segmentation_rendering()
                    r.update_scene(data, camera=cam)
                    seg = np.array(r.render(), copy=True)
                finally:
                    r.disable_segmentation_rendering()
                    model.vis.global_.fovy = old_fovy
            geom_ids = session.robot_geom_ids()
            body_poses, reset = session.live_body_poses()
        mask = mask_from_segmentation(seg, geom_ids)
        if (rw, rh) != (W, H):
            rgb = _resize(rgb, (W, H))
            mask = _resize(mask, (W, H), nearest=True)
        return rgb, mask, body_poses, reset


__all__ = ["ServerRenderStream", "StreamSettings", "SCENE_TOPICS", "RESOLUTION_CHOICES",
           "viser_camera_to_w2c_K", "stream_size", "vertical_fov_deg", "free_camera_params",
           "apply_free_camera", "forward_from_azimuth_elevation", "mask_from_segmentation",
           "camera_state"]
