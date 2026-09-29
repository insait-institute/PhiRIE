"""Photoreal observations for pi0.5 eval: gsplat composite per control tick.

Per camera and tick: background splat (clean_background.ply when inpainted,
else the raw scene splat) + per-object canonical gaussians posed from the
live MuJoCo state, with the MuJoCo-rendered robot masked in on top via
segmentation (SIMPLER-style green-screening, but sim-over-splat).

This is what closes the visual real2sim gap: PolaRiS showed visual mismatch
costs ~25% success while sim dynamics cost ~0 - and photoreal renders are
exactly what SimAny's asset pipeline provides.

Needs GPU (gsplat/torch) + MUJOCO_GL=egl. Run inside the SimAny venv
with SIMANY_SCENE/SIMANY_OUT set for the scene being evaluated.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from agents.core import common as C


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_bound_background(out_dir: Path) -> tuple[Path, dict]:
    """Resolve a materialized scene's frozen source splat, never a global one."""
    materialization = out_dir / "materialization_manifest.json"
    if materialization.is_file():
        value = json.loads(materialization.read_text())
        identity = value.get("source_scene", {}).get("source_scene_gaussian")
        if not isinstance(identity, dict) or set(identity) != {
            "path", "sha256", "size_bytes"
        }:
            raise ValueError(
                "materialized factory has no exact source_scene_gaussian identity"
            )
        path = Path(identity["path"])
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid materialized source scene Gaussian: {path}")
        size = path.stat().st_size
        if size != identity["size_bytes"] or _sha256_file(path) != identity["sha256"]:
            raise ValueError("materialized source scene Gaussian identity drift")
        return path, {
            "path": str(path),
            "sha256": identity["sha256"],
            "size_bytes": size,
            "semantics": "frozen_raw_scene_splat_with_original_capture_objects",
        }
    clean = out_dir / "inpaint" / "clean_background.ply"
    path = clean if clean.exists() else C.SPLAT_PLY
    return path, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "semantics": "legacy_clean_background" if clean.exists() else "legacy_raw_scene_splat",
    }


def depth_test_robot(splat_rgb, splat_depth, splat_alpha, robot_rgb,
                     robot_depth, robot_mask, *, alpha_min=0.95):
    """Approximate visibility against GS expected camera-z depth.

    Unobserved/low-opacity pixels are explicitly unknown; they never authorize
    robot pasting. This does not recover surfaces hidden by the capture robot.
    """
    splat_rgb=np.asarray(splat_rgb);robot_rgb=np.asarray(robot_rgb)
    depth=np.asarray(splat_depth);alpha=np.asarray(splat_alpha)
    rz=np.asarray(robot_depth);mask=np.asarray(robot_mask,dtype=bool)
    shape=splat_rgb.shape[:2]
    if (splat_rgb.ndim!=3 or splat_rgb.shape[2]!=3 or robot_rgb.shape!=splat_rgb.shape
            or any(x.shape!=shape for x in (depth,alpha,rz,mask))):
        raise ValueError('RGB/depth/alpha/robot masks require the same calibrated pixel grid')
    if splat_rgb.dtype!=np.uint8 or robot_rgb.dtype!=np.uint8:
        raise ValueError('robot restoration requires byte RGB')
    if not 0<alpha_min<=1:raise ValueError('opacity threshold must be in (0,1]')
    known=np.isfinite(depth)&(depth>0)&np.isfinite(alpha)&(alpha>=alpha_min)&(alpha<=1)
    robot_valid=np.isfinite(rz)&(rz>0)
    visible=mask&known&robot_valid&(rz<=depth)
    unknown=(~known)|(mask&~robot_valid)
    out=splat_rgb.copy();out[visible]=robot_rgb[visible]
    return out,{'robot_visible':visible,'robot_occluded':mask&known&robot_valid&~visible,
                'unknown_coverage':unknown,'depth_semantics':'GS expected camera-z vs MuJoCo metric camera-z'}


class CompositeObs:
    """Renders photoreal (splat + robot overlay) images for env cameras."""

    def __init__(self, env, out_dir, width=640, height=360, scale=1.0,
                 *, robot_compositing="legacy_mask", alpha_min=0.95,
                 robot_roots=("panda", "franka", "robot", "robotiq"), scene_option=None):
        import mujoco
        self.env = env
        self.W, self.H, self.scale = width, height, scale
        if robot_compositing not in ("legacy_mask", "expected_depth_v1"):
            raise ValueError('unknown robot compositing protocol')
        if robot_compositing=='expected_depth_v1' and scale!=1.0:
            raise ValueError('depth compositing requires the exact native camera grid')
        if not 0<alpha_min<=1:raise ValueError('opacity threshold must be in (0,1]')
        self.robot_compositing=robot_compositing;self.alpha_min=alpha_min
        self.last_visibility=None
        self.scene_option=scene_option
        self.last_layers=None
        out_dir = Path(out_dir)

        bg_src, self.background_identity = _scene_bound_background(out_dir)
        self.bg = C.load_gaussians(bg_src)
        self.canon, self.scales = {}, {}
        objects = json.loads((out_dir / "objects" / "objects.json").read_text())
        for m in objects:
            name = f"obj_{m['index']:02d}"
            odir = out_dir / "objects" / name
            al = odir / "aligned.json"
            if not al.exists() or not (odir / "trellis_gs.ply").exists():
                continue
            al_d = json.loads(al.read_text())
            if al_d.get("rejected"):
                continue
            if name not in env.free_bodies:  # excluded from the sim
                continue
            self.canon[name] = C.pad_sh(
                C.load_gaussians(odir / "trellis_gs.ply"),
                self.bg["sh_degree"])
            self.scales[name] = C.decompose_similarity(
                np.array(al_d["T"]))[0]

        # robot mask renderer (segmentation) + robot RGB renderer share the
        # env's model; use a dedicated renderer at obs resolution
        self.mj_renderer = mujoco.Renderer(env.model, height=height,
                                           width=width)
        self._robot_geoms = set()
        for g in range(env.model.ngeom):
            gname = env.model.geom(g).name or ""
            bid = env.model.geom_bodyid[g]
            bname = env.model.body(bid).name or ""
            if bname.startswith("robot/") or gname.startswith("robot/"):
                self._robot_geoms.add(g)

        if robot_compositing=='expected_depth_v1':
            from robo.rendering.mujoco_masks import robot_geom_ids
            self._robot_geoms=robot_geom_ids(env.model,robot_roots)

    # camera helpers -------------------------------------------------------
    def _cam_K_w2c(self, cam_name):
        m, d = self.env.model, self.env.data
        cid = m.camera(cam_name).id
        fovy = np.radians(m.cam_fovy[cid])
        fy = self.H / (2 * np.tan(fovy / 2))
        K = np.array([[fy, 0, self.W / 2], [0, fy, self.H / 2], [0, 0, 1]])
        c2w_gl = np.eye(4)
        c2w_gl[:3, :3] = d.cam_xmat[cid].reshape(3, 3)
        c2w_gl[:3, 3] = d.cam_xpos[cid]
        # MuJoCo/OpenGL camera (-z forward, +y up) -> OpenCV (+z fwd, +y down)
        c2w_cv = c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
        return K, np.linalg.inv(c2w_cv)

    def render(self, cam_name):
        K, w2c = self._cam_K_w2c(cam_name)
        parts = [self.bg]
        for name, gs in self.canon.items():
            pos, quat = self.env.body_pose(name)
            T = np.eye(4)
            T[:3, :3] = self.scales[name] * C.quat_to_rot_wxyz(quat)
            T[:3, 3] = pos
            parts.append(C.transform_gaussians(gs, T))
        depth_mode=self.robot_compositing=='expected_depth_v1'
        splat,gs_depth,gs_alpha = C.render_view(C.cat_gaussians(parts), w2c, K,
                              self.W, self.H, scale=self.scale,
                              render_mode="RGB+ED" if depth_mode else "RGB")
        splat = (np.clip(splat, 0, 1) * 255).astype(np.uint8)

        scene_kwargs={} if getattr(self,'scene_option',None) is None else {'scene_option':self.scene_option}
        self.mj_renderer.update_scene(self.env.data, camera=cam_name, **scene_kwargs)
        rgb = self.mj_renderer.render().copy()
        robot_depth=None
        if depth_mode:
            self.mj_renderer.enable_depth_rendering()
            try:
                self.mj_renderer.update_scene(self.env.data,camera=cam_name, **scene_kwargs)
                robot_depth=self.mj_renderer.render().copy()
            finally:
                self.mj_renderer.disable_depth_rendering()
        self.mj_renderer.enable_segmentation_rendering()
        self.mj_renderer.update_scene(self.env.data, camera=cam_name, **scene_kwargs)
        seg = self.mj_renderer.render()
        self.mj_renderer.disable_segmentation_rendering()
        geom_ids = seg[..., 0]
        mask = np.isin(geom_ids, list(self._robot_geoms))
        if depth_mode:
            from robo.rendering.mujoco_masks import robot_mask_from_segmentation
            mask=robot_mask_from_segmentation(seg,self._robot_geoms)>0
            out,self.last_visibility=depth_test_robot(splat,gs_depth,gs_alpha,rgb,
                robot_depth,mask,alpha_min=self.alpha_min)
            self.last_layers={'world_rgb':splat,'world_depth':gs_depth,'world_alpha':gs_alpha,
                              'robot_rgb':rgb,'robot_depth':robot_depth,'robot_mask':mask}
            return out
        if splat.shape[:2] != rgb.shape[:2]:
            import cv2
            splat = cv2.resize(splat, (rgb.shape[1], rgb.shape[0]))
        out = splat.copy()
        out[mask] = rgb[mask]
        return out
