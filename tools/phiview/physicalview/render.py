"""CUDA renders for Studio (gsplat via agents.core.common.render_view).

CONTRACT (scene/render agent). Requires torch+gsplat in the studio env; every function
raises RenderUnavailable (not ImportError) with a helpful message when CUDA/gsplat is
missing so the UI can degrade to client-side splats only.

    class Renderer:
        __init__(state: SceneState, device="cuda", use_clean_bg=True)
            moves the background gaussians to the GPU once; caches per-object canonical
            gaussians on the GPU (object_canonical_gs).
        render(w2c, K, wh, object_poses: dict[str, 4x4] | None = None, background=None)
            -> uint8 RGB [H,W,3]. object_poses maps obj_id -> world T for the canonical
            gaussians (None = objects at their aligned rest T; {} = background only).
            Objects listed in `hidden` (set attribute) are omitted.
        render_camera(frame_name, scale=0.5, **kw)  uses state.cameras[frame_name], state.K
        composite_with_robot(w2c, K, wh, object_poses, robot_rgb, robot_mask)
            -> RGB where robot pixels (mask>0) are taken from the MuJoCo raster (same rule
            as robo/rendering/pi05_render.py: sim-over-splat).
        thumbnail(obj_id, source=None, size=192) -> uint8 [size,size,3] turntable-ish
            render of one canonical proposal on white (for proposal cards).
        set_background(kind: "raw"|"clean", force=False) ; kind switches which gaussians are
            used; force=True re-uploads the current kind (state.clean_bg_gs was replaced).

    def mujoco_camera_to_w2c_K(model, data, cam_name, wh) -> (w2c 4x4 OpenCV, K 3x3)
        so the photoreal view matches the MuJoCo camera exactly (reuse the conversion in
        robo/rendering/pi05_render.py if present; MuJoCo cameras look down -z with +y up).

Implementation notes: torch and gsplat are imported lazily inside methods so this module
imports in CPU-only environments; ``mujoco_camera_to_w2c_K`` is pure numpy (needs only a
forwarded ``MjData``) and is unit-tested against MuJoCo's own renderer.
When ``object_poses`` is None the renderer honours ``state.edited_poses`` (gizmo edits)
over the aligned rest transforms, so snapshots match what the viewer shows.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from physicalview.scene_state import SceneState, object_canonical_gs

log = logging.getLogger("studio.render")


class RenderUnavailable(RuntimeError):
    pass


def _require_cuda(device: str):
    """Import torch/gsplat lazily; translate every failure into RenderUnavailable."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RenderUnavailable(f"torch import failed: {exc}") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RenderUnavailable("torch.cuda.is_available() is False: photoreal renders need a GPU node "
                                "(client-side splats still work)")
    try:
        import gsplat  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RenderUnavailable(f"gsplat import failed ({exc}); use the studio/mini-viewer env") from exc
    return torch


def _to_device(gs: dict, device: str) -> dict:
    out = dict(gs)
    for k in ("means", "quats", "scales", "opacities", "sh"):
        out[k] = gs[k].to(device)
    return out


def _resize_rgb(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = int(wh[0]), int(wh[1])
    if img.shape[1] == w and img.shape[0] == h:
        return img
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up: np.ndarray = (0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV w2c (x right, y down, z forward) for a camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z) + 1e-12
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return np.linalg.inv(c2w)


class Renderer:
    def __init__(self, state: SceneState, device: str = "cuda", use_clean_bg: bool = True):
        torch = _require_cuda(device)
        self.state = state
        self.device = device
        self.hidden: set[str] = set()
        self._lock = threading.Lock()
        self._canon: dict[tuple[str, str], dict] = {}
        self._bg_kind: str | None = None
        self._bg: dict | None = None
        if state.splat_gs is None and state.clean_bg_gs is None:
            raise RenderUnavailable(f"scene {state.result_set.name} has no background splat loaded")
        kind = "clean" if (use_clean_bg and state.clean_bg_gs is not None) else "raw"
        if kind == "raw" and state.splat_gs is None:
            kind = "clean"
        self.set_background(kind)
        for obj_id, rec in state.objects.items():
            if not rec.accepted:
                continue
            try:
                self._canonical(obj_id, None)
            except Exception as exc:  # noqa: BLE001 - a missing proposal must not kill the renderer
                log.warning("renderer: %s canonical gaussians unavailable: %s", obj_id, exc)
        torch.cuda.empty_cache()

    # --- gaussians -------------------------------------------------------------------
    @property
    def sh_degree(self) -> int:
        return int(self._bg["sh_degree"]) if self._bg is not None else 0

    def set_background(self, kind: str, force: bool = False) -> None:
        if kind not in ("raw", "clean"):
            raise ValueError(f"background kind must be 'raw' or 'clean', got {kind!r}")
        src = self.state.clean_bg_gs if kind == "clean" else self.state.splat_gs
        if src is None:
            raise RenderUnavailable(f"no {kind} background gaussians loaded for this scene")
        if kind == self._bg_kind and not force:
            return
        import torch
        with self._lock:
            self._bg = None
            torch.cuda.empty_cache()
            self._bg = _to_device(src, self.device)
            self._bg_kind = kind
            # canonical gaussians are padded to the background's SH degree
            for key, gs in list(self._canon.items()):
                from agents.core.common import pad_sh
                self._canon[key] = pad_sh(gs, max(self.sh_degree, int(gs["sh_degree"])))

    @property
    def background_kind(self) -> str | None:
        return self._bg_kind

    def _canonical(self, obj_id: str, source: str | None) -> dict:
        from agents.core.common import pad_sh
        rec = self.state.objects[obj_id]
        src = source or rec.chosen_source or "trellis"
        key = (obj_id, src)
        gs = self._canon.get(key)
        if gs is None:
            cpu = object_canonical_gs(self.state, obj_id, source)
            gs = pad_sh(_to_device(cpu, self.device), max(self.sh_degree, int(cpu["sh_degree"])))
            self._canon[key] = gs
        return gs

    def invalidate(self, obj_ids=None) -> None:
        """Drop cached GPU gaussians (after a job re-generated an object)."""
        with self._lock:
            if obj_ids is None:
                self._canon.clear()
            else:
                for key in list(self._canon):
                    if key[0] in obj_ids:
                        self._canon.pop(key, None)

    def _rest_poses(self) -> dict[str, np.ndarray]:
        poses = {}
        for obj_id, rec in self.state.objects.items():
            if not rec.accepted or rec.T_world is None:
                continue
            poses[obj_id] = self.state.edited_poses.get(obj_id, rec.T_world)
        return poses

    def _compose(self, object_poses: dict[str, np.ndarray] | None) -> dict:
        from agents.core.common import cat_gaussians, transform_gaussians
        if self._bg is None:
            raise RenderUnavailable("no background gaussians on the GPU")
        parts = [self._bg]
        poses = self._rest_poses() if object_poses is None else object_poses
        for obj_id, T in poses.items():
            if obj_id in self.hidden or obj_id not in self.state.objects:
                continue
            try:
                gs = self._canonical(obj_id, None)
            except Exception as exc:  # noqa: BLE001
                log.debug("skip %s: %s", obj_id, exc)
                continue
            parts.append(transform_gaussians(gs, np.asarray(T, dtype=np.float64)))
        return parts[0] if len(parts) == 1 else cat_gaussians(parts)

    # --- renders ---------------------------------------------------------------------
    def render(self, w2c: np.ndarray, K: np.ndarray, wh: tuple[int, int],
               object_poses: dict[str, np.ndarray] | None = None,
               background=None) -> np.ndarray:
        from agents.core.common import render_view
        w, h = int(wh[0]), int(wh[1])
        with self._lock:
            gs = self._compose(object_poses)
            rgb, _, _ = render_view(gs, np.asarray(w2c, dtype=np.float64), np.asarray(K, dtype=np.float64),
                                    w, h, background=background)
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def render_camera(self, frame_name: str, scale: float = 0.5, **kw) -> np.ndarray:
        st = self.state
        if st.K is None or frame_name not in st.cameras:
            raise KeyError(f"camera {frame_name!r} not available (K loaded: {st.K is not None}, "
                           f"{len(st.cameras)} frames)")
        K = np.asarray(st.K, dtype=np.float64).copy()
        w, h = int(round(st.W * scale)), int(round(st.H * scale))
        K[:2] *= scale
        return self.render(st.cameras[frame_name], K, (w, h), **kw)

    def composite_with_robot(self, w2c, K, wh, object_poses, robot_rgb, robot_mask) -> np.ndarray:
        splat = self.render(w2c, K, wh, object_poses)
        robot_rgb = np.asarray(robot_rgb)
        if robot_rgb.ndim == 3 and robot_rgb.shape[2] == 4:
            robot_rgb = robot_rgb[..., :3]
        mask = np.asarray(robot_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if splat.shape[:2] != robot_rgb.shape[:2]:
            splat = _resize_rgb(splat, (robot_rgb.shape[1], robot_rgb.shape[0]))
        if mask.shape[:2] != robot_rgb.shape[:2]:
            mask = _resize_rgb(mask.astype(np.uint8)[..., None].repeat(3, axis=2),
                               (robot_rgb.shape[1], robot_rgb.shape[0]))[..., 0]
        out = splat.copy()
        m = mask > 0
        out[m] = robot_rgb[m]
        return out

    def thumbnail(self, obj_id: str, source: str | None = None, size: int = 192) -> np.ndarray:
        """3/4 view of one canonical proposal (TRELLIS frame: z-up, ~[-0.5,0.5]^3) on white."""
        from agents.core.common import render_view
        with self._lock:
            gs = self._canonical(obj_id, source)
            means = gs["means"]
            lo, hi = means.min(0).values.cpu().numpy(), means.max(0).values.cpu().numpy()
            center = (lo + hi) / 2.0
            radius = float(np.linalg.norm(hi - lo)) / 2.0 + 1e-3
            fov = np.radians(40.0)
            dist = radius / np.sin(fov / 2.0) * 1.05
            direction = np.array([1.0, -1.0, 0.75])
            direction /= np.linalg.norm(direction)
            w2c = look_at_w2c(center + direction * dist, center)
            f = size / (2.0 * np.tan(fov / 2.0))
            K = np.array([[f, 0, size / 2.0], [0, f, size / 2.0], [0, 0, 1.0]])
            rgb, _, _ = render_view(gs, w2c, K, size, size, background=[1.0, 1.0, 1.0])
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def save_png(self, img: np.ndarray, path: Path) -> Path:
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(str(path))
        return path


def mujoco_camera_to_w2c_K(model, data, cam_name: str, wh: tuple[int, int]):
    """MuJoCo camera -> (OpenCV w2c 4x4, K 3x3) for an image of size wh=(W, H).

    Same conversion as robo/rendering/pi05_render.py: MuJoCo/OpenGL cameras look along
    -z with +y up; OpenCV looks along +z with +y down, so c2w_cv = c2w_gl @ diag(1,-1,-1).
    fovy is MuJoCo's vertical field of view (degrees); pixels are square (fx = fy)."""
    W, H = int(wh[0]), int(wh[1])
    cid = model.camera(cam_name).id if isinstance(cam_name, str) else int(cam_name)
    fovy = np.radians(float(model.cam_fovy[cid]))
    fy = H / (2.0 * np.tan(fovy / 2.0))
    K = np.array([[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    c2w_gl = np.eye(4, dtype=np.float64)
    c2w_gl[:3, :3] = np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    c2w_gl[:3, 3] = np.asarray(data.cam_xpos[cid], dtype=np.float64)
    c2w_cv = c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w_cv), K


def project_points(w2c: np.ndarray, K: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """World points [N,3] -> pixel coords [N,2] (u right, v down) with an OpenCV w2c/K."""
    p = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    pc = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.clip(pc[:, 2:3], 1e-9, None)
    uv = pc[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
