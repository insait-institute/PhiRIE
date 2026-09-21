"""Shared utilities for the SimAny pipeline.

World frame everywhere: the ScanNet++ mesh/colmap frame (metric, z-up, floor
near z=0). Camera poses: OpenCV w2c straight from dslr/colmap/images.txt,
paired with PINHOLE intrinsics from dslr/nerfstudio/transforms_undistorted.json
and images from dslr/resized_undistorted_images (verified recipe, PSNR 33+).

TRELLIS canonical object frame: z-up, roughly [-0.5, 0.5]^3.
Per-object registration: x_world = s * R @ x_canonical + t, stored as a single
4x4 matrix T with the isotropic scale folded into the upper-left 3x3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_LEGACY_PREFIX = "SIMF_"  # pre-rename prefix, honoured for one release


def env(name: str, default: str | None = None) -> str | None:
    """Read SIMANY_<name>, falling back to the pre-rename legacy prefix.

    The legacy prefix is still honoured so queued jobs and archived launchers
    keep working; it warns so the fallback does not go unnoticed.
    """
    v = os.environ.get(f"SIMANY_{name}")
    if v is None:
        v = os.environ.get(f"{_LEGACY_PREFIX}{name}")
        if v is not None:
            print(f"[simany] {_LEGACY_PREFIX}{name} is deprecated; "
                  f"use SIMANY_{name}", file=sys.stderr)
    return default if v is None else v


_SAM3_SNAPSHOT_REL = (
    "models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")


def resolve_sam3_ckpt() -> str:
    """Resolve the SAM3 checkpoint, honoring a strict explicit override first.

    ``SIMANY_SAM3_CKPT`` is used by frozen/reproducible jobs and therefore must
    name an absolute, non-empty, regular non-symlink file.  Without that
    override, retain the historical cache search for interactive runs.
    """
    override = env("SAM3_CKPT")
    if override is not None:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError("SIMANY_SAM3_CKPT must be an absolute path")
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                "SIMANY_SAM3_CKPT must name a non-empty regular non-symlink "
                f"file: {path}"
            )
        return str(path)

    candidates = [
        os.path.expanduser(f"~/.cache/huggingface/hub/{_SAM3_SNAPSHOT_REL}"),
        f"/group/worldcept/hf_cache/hub/{_SAM3_SNAPSHOT_REL}",
    ]
    if os.environ.get("HF_HOME"):
        candidates.insert(
            0, os.path.join(os.environ["HF_HOME"], "hub", _SAM3_SNAPSHOT_REL))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"SAM3 checkpoint not found in any of: {candidates}. "
        "Copy it to /group/worldcept/hf_cache/hub/models--facebook--sam3/ "
        "or set HF_HOME to a shared location.")


# Repo root is derived from this file, so the tree can be moved or cloned.
ROOT = Path(env("ROOT") or Path(__file__).resolve().parents[2])
TRELLIS_DIR = Path(env("TRELLIS_DIR") or ROOT / "third_party" / "TRELLIS")

SCENE_ID = env("SCENE", "c50d2d1d42")
_DATA_ROOT = Path(env("SCANNETPP_ROOT") or "/data/ScanNetpp")
_SPLATS_ROOT = Path(env("SPLATS_ROOT") or "/data/ScanNetppv2_gsplat/splats")

SCENE_DIR = _DATA_ROOT / "data" / SCENE_ID
SPLAT_PLY = _SPLATS_ROOT / f"{SCENE_ID}.ply"
IMAGES_DIR = SCENE_DIR / "dslr" / "resized_undistorted_images"
TRANSFORMS_JSON = SCENE_DIR / "dslr" / "nerfstudio" / "transforms_undistorted.json"
COLMAP_IMAGES_TXT = SCENE_DIR / "dslr" / "colmap" / "images.txt"
MESH_PLY = SCENE_DIR / "scans" / "mesh_aligned_0.05.ply"
SEGMENTS_JSON = SCENE_DIR / "scans" / "segments.json"
SEGMENTS_ANNO_JSON = SCENE_DIR / "scans" / "segments_anno.json"

# SIMANY_OUT overrides the output tree (factory mode writes next to the
# zero-shot outputs without touching them); scene inputs stay SIMANY_SCENE
_out = env("OUT")
OUT = Path(_out) if _out else ROOT / "outputs" / SCENE_ID

# Pipeline-internal geometry (raycasting for discovery/registration/plane-fit,
# physics collision) vs. GT-evaluation geometry (MESH_PLY, above) are
# DELIBERATELY separate constants. GT instance annotations (segments_anno.json)
# are indexed by vertex position into MESH_PLY specifically - swapping that
# path would silently corrupt load_gt_instances(). Ablations that replace the
# scan mesh with a self-derived one (SIMANY_MESH_SRC=derived) must only affect
# PIPELINE_MESH_PLY; evaluation always scores against the true MESH_PLY.
_mesh_src = env("MESH_SRC", "")
if _mesh_src == "derived":
    PIPELINE_MESH_PLY = OUT / "derived_mesh.ply"
elif _mesh_src:
    PIPELINE_MESH_PLY = Path(_mesh_src)
else:
    PIPELINE_MESH_PLY = MESH_PLY

# One greppable line per process so every stage log records what resolved.
print(f"[simany] scene={SCENE_ID} out={OUT} "
      f"auto={1 if env('AUTO') == '1' else 0} "
      f"mesh_src={_mesh_src or 'gt'}", file=sys.stderr)

# -------- simulatable vocabulary (everything except the building shell) ---
# prompt/label -> max plausible extent (m). Used by factory_prepare (GT
# whitelist + size gate) and auto_segment (SAM3 prompts + lift gate).
VOCAB_SMALL = {
    "bottle": 0.45, "plastic bottle": 0.45, "glass bottle": 0.45,
    "water bottle": 0.45, "cup": 0.25, "mug": 0.25, "paper cup": 0.2,
    "box": 0.9, "cardboard box": 1.2, "carboard box": 1.2,
    "storage box": 0.9, "tissue box": 0.4, "book": 0.5, "notebook": 0.45,
    "keyboard": 0.6, "mouse": 0.2, "computer mouse": 0.2, "telephone": 0.4,
    "headphones": 0.45, "headphone": 0.45, "shoe": 0.4, "shoes": 0.5,
    "bag": 0.8, "backpack": 0.8, "toy": 0.6, "bowl": 0.4, "plate": 0.4,
    "can": 0.25, "jar": 0.35, "remote": 0.3, "remote control": 0.3,
    "pen": 0.25, "pencil": 0.25, "marker": 0.25, "whiteboard marker": 0.25,
    "stapler": 0.3, "scissors": 0.3, "plant pot": 0.8, "flower pot": 0.8,
    "vase": 0.6, "kettle": 0.4, "laptop": 0.6, "tablet": 0.4,
    "cellphone": 0.25, "phone": 0.3, "basket": 0.7, "tray": 0.6,
    "folder": 0.5, "charger": 0.3, "laptop charger": 0.3,
    "wireless charger": 0.25, "pen holder": 0.3, "speaker": 0.6,
    "clock": 0.5,
}
VOCAB_FURNITURE = {
    "table": 2.6, "desk": 2.6, "office table": 2.6, "coffee table": 1.6,
    "chair": 1.4, "office chair": 1.4, "armchair": 1.6, "stool": 1.0,
    "sofa": 2.8, "couch": 2.8, "monitor": 1.2, "computer monitor": 1.2,
    "tv": 1.8, "television": 1.8, "shelf": 2.6, "bookshelf": 2.6,
    "cabinet": 2.4, "drawer": 1.6, "nightstand": 1.0, "trash can": 0.9,
    "trash bin": 0.9, "bin": 0.9, "lamp": 1.8, "floor lamp": 1.9,
    "whiteboard": 2.6, "bed": 2.6, "bench": 2.2, "printer": 0.9,
    "computer tower": 0.7, "pc": 0.7, "suitcase": 0.9, "ladder": 2.4,
    "fan": 0.8, "heater": 1.2, "radiator": 1.6, "picture": 1.5,
    "picture frame": 1.5, "mirror": 2.0, "pillow": 0.9, "cushion": 0.9,
    "blanket": 1.5, "towel": 1.2, "curtain": 2.8, "rug": 3.0, "carpet": 3.0,
}
VOCAB = {**VOCAB_SMALL, **VOCAB_FURNITURE}
# building shell / layout: never simulated, never removed
STRUCTURAL_EXCLUDE = {
    "wall", "floor", "ceiling", "door", "doorframe", "door frame", "window",
    "windowsill", "window frame", "window sill", "beam", "column", "pillar",
    "stairs", "staircase", "railing", "glass pane", "ventilation", "vent",
    "light", "ceiling light", "ceiling lamp", "socket", "light switch",
    "pipe", "duct", "sprinkler", "smoke detector",
}

# SAM3 concept prompt -> ScanNet++ GT labels it may correspond to (for eval
# matching and background carving only; the pipeline itself never reads GT
# labels of course).
TARGET_PROMPTS = {
    "bottle": ["plastic bottle", "glass bottle", "bottle"],
    "mug": ["mug", "cup"],
    "computer mouse": ["mouse"],
    "keyboard": ["keyboard"],
    "headphones": ["headphones"],
    "telephone": ["telephone"],
    "box": ["box", "cardboard box", "carboard box", "storage box"],
}

# Fallback physics table when the VLM annotation fails to parse.
FALLBACK_PHYSICS = {
    "bottle": (0.5, 0.5), "mug": (0.35, 0.5), "computer mouse": (0.09, 0.6),
    "keyboard": (0.9, 0.6), "headphones": (0.25, 0.5), "telephone": (1.2, 0.6),
    "box": (0.4, 0.6),
}


# ---------------------------------------------------------------- cameras --

def load_intrinsics(transforms_json: Path = TRANSFORMS_JSON):
    meta = json.loads(transforms_json.read_text())
    K = np.array(
        [[meta["fl_x"], 0.0, meta["cx"]],
         [0.0, meta["fl_y"], meta["cy"]],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, int(meta["w"]), int(meta["h"]), meta


def load_colmap_w2c(images_txt: Path = COLMAP_IMAGES_TXT):
    """Parse colmap images.txt -> {image_name: 4x4 w2c (OpenCV convention)}."""
    w2c = {}
    lines = [ln.strip() for ln in images_txt.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    for ln in lines[::2]:  # meta line, then 2D-points line
        parts = ln.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = quat_to_rot_wxyz(np.array([qw, qx, qy, qz]))
        M[:3, 3] = [tx, ty, tz]
        w2c[name] = M
    return w2c


# ------------------------------------------------------------ quaternions --

def quat_to_rot_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rot_to_quat_wxyz(R):
    """Convert (..., 3, 3) rotations without an untracked local utils3d shim."""
    from scipy.spatial.transform import Rotation

    matrices = np.asarray(R, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape (..., 3, 3)")
    batch = matrices.shape[:-2]
    if matrices.size == 0:
        return np.empty(batch + (4,), dtype=np.float64)
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    quaternions = xyzw[:, [3, 0, 1, 2]]
    # Preserve the old shim's convention: the largest component is positive.
    dominant = np.argmax(np.abs(quaternions), axis=1)
    signs = np.where(quaternions[np.arange(len(quaternions)), dominant] < 0, -1, 1)
    return (quaternions * signs[:, None]).reshape(batch + (4,))


def quat_mul_wxyz(a, b):
    """Hamilton product, batched on b: a (4,), b (...,4) -> (...,4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def decompose_similarity(T):
    """4x4 with isotropic scale folded into R -> (s, R, t)."""
    T = np.asarray(T, dtype=np.float64)
    A = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-12)))
    return s, A / s, T[:3, 3].copy()


# ------------------------------------------------------------ GT instances --

def load_gt_instances():
    """ScanNet++ instance annotations -> list of dicts with world-frame stats."""
    from plyfile import PlyData

    seg_idx = np.asarray(json.loads(SEGMENTS_JSON.read_text())["segIndices"])
    anno = json.loads(SEGMENTS_ANNO_JSON.read_text())
    ply = PlyData.read(str(MESH_PLY))
    v = ply["vertex"]
    verts = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)

    out = []
    for g in anno["segGroups"]:
        mask = np.isin(seg_idx, np.asarray(g["segments"]))
        idx = np.nonzero(mask)[0]
        if len(idx) < 10:
            continue
        pts = verts[idx]
        out.append({
            "object_id": g["objectId"],
            "label": g["label"],
            "vert_idx": idx,
            "centroid": pts.mean(axis=0),
            "aabb": np.stack([pts.min(axis=0), pts.max(axis=0)]),
        })
    return out


def load_auto_instances(*, instances_path=None, mesh_path=None):
    """Instances discovered by auto_segment.py (no GT semantics used).
    Same schema as load_gt_instances; object_id = 1000 + index.

    vert_idx here indexes whatever mesh auto_segment.py actually raycasted
    against (PIPELINE_MESH_PLY - the true MESH_PLY by default, or a
    self-derived mesh under SIMANY_MESH_SRC ablations) - NOT the GT mesh, so
    this must resolve against PIPELINE_MESH_PLY or indices silently point
    at unrelated vertices in a different mesh."""
    from plyfile import PlyData

    d = np.load(instances_path if instances_path is not None else OUT / "auto_instances.npz", allow_pickle=False)
    ply = PlyData.read(str(mesh_path if mesh_path is not None else PIPELINE_MESH_PLY))
    v = ply["vertex"]
    verts = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)
    out = []
    for k, label in enumerate(d["labels"]):
        idx = d[f"vert_idx_{k}"]
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or not len(idx) or np.any(idx < 0) or np.any(idx >= len(verts)):
            raise ValueError("automatic segmentation indices do not match construction mesh")
        pts = verts[idx]
        out.append({"object_id": 1000 + k, "label": str(label),
                    "vert_idx": idx, "centroid": pts.mean(axis=0),
                    "aabb": np.stack([pts.min(axis=0), pts.max(axis=0)]),
                    "score": float(d["scores"][k])})
    return out


def load_instances():
    """Dispatch: SIMANY_AUTO=1 -> automated (image+pose+mesh only), else GT."""
    if env("AUTO") == "1":
        return load_auto_instances()
    return load_gt_instances()


def load_mesh_o3d():
    """Pipeline-internal mesh (raycasting/registration/plane-fit/collision) -
    NOT the GT-evaluation mesh. See PIPELINE_MESH_PLY vs MESH_PLY above."""
    import open3d as o3d
    return o3d.io.read_triangle_mesh(str(PIPELINE_MESH_PLY))


def make_raycast_scene():
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(PIPELINE_MESH_PLY))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def match_gt_instance(centroid, label, gts, max_dist=0.5):
    """Nearest independent GT instance by centroid (+ label agreement bonus).

    Used to score any discovered/registered object against a GT instance it
    was NOT derived from (unlike the in-pipeline 'own extraction' points),
    so ablations that change the pipeline's internal mesh/segmentation are
    all scored on the same yardstick. Returns None if nothing is close.
    """
    best, best_d = None, max_dist
    for g in gts:
        d = float(np.linalg.norm(np.asarray(centroid) - g["centroid"]))
        if label and g.get("label") == label:
            d -= 0.15  # same-label tie-break bonus, not a hard filter
        if d < best_d:
            best, best_d = g, d
    return best


def mesh_zdepth(scene, K, w2c, width, height, stride=1):
    """Ray-cast the scan mesh -> z-depth map (OpenCV convention).

    Rays are built with camera-frame z-component 1 so t_hit equals z-depth.
    Returns (H',W') float32, inf where no hit, for the strided pixel grid.
    """
    import open3d as o3d

    us = np.arange(0, width, stride, dtype=np.float64) + 0.5
    vs = np.arange(0, height, stride, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(us, vs)
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                      np.ones_like(uu)], axis=-1)
    c2w = np.linalg.inv(w2c)
    d_world = d_cam @ c2w[:3, :3].T
    o_world = np.broadcast_to(c2w[:3, 3], d_world.shape)
    rays = o3d.core.Tensor(
        np.concatenate([o_world, d_world], axis=-1).reshape(-1, 6).astype(np.float32))
    ans = scene.cast_rays(rays)
    return ans["t_hit"].numpy().reshape(uu.shape).astype(np.float32)


def unproject(K, depth, mask, c2w):
    """Pixels where mask & finite depth -> world points (N,3)."""
    vv, uu = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[vv, uu].astype(np.float64)
    x = (uu + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vv + 0.5 - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=1)
    return pts_cam @ c2w[:3, :3].T + c2w[:3, 3]


# ------------------------------------------------------------- gaussians --

def load_gaussians(ply_path=SPLAT_PLY, device="cuda"):
    """Load an Inria-format 3DGS ply (any SH degree, including 0).

    Returns dict: means [N,3], quats [N,4] wxyz normalized, scales [N,3]
    (exp-activated), opacities [N] (sigmoid-activated), sh [N,K,3], sh_degree.
    """
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    names = {p.name for p in v.properties}

    def cols(cnames):
        return np.stack([np.asarray(v[n], dtype=np.float32) for n in cnames], axis=1)

    means = cols(["x", "y", "z"])
    f_dc = cols([f"f_dc_{i}" for i in range(3)])
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    if n_rest:
        k_rest = n_rest // 3
        f_rest = cols([f"f_rest_{i}" for i in range(n_rest)])
        f_rest = f_rest.reshape(-1, 3, k_rest).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], f_rest], axis=1)
    else:
        sh = f_dc[:, None, :]
    sh_deg = int(np.sqrt(sh.shape[1]) - 1)

    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    scales = np.exp(cols([f"scale_{i}" for i in range(3)]))
    quats = cols([f"rot_{i}" for i in range(4)])
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return {"means": t(means), "quats": t(quats), "scales": t(scales),
            "opacities": t(opac.reshape(-1)), "sh": t(sh), "sh_degree": sh_deg}


def pad_sh(gs, target_degree):
    """Zero-pad SH rest coefficients up to target_degree (in place)."""
    import torch
    K_t = (target_degree + 1) ** 2
    sh = gs["sh"]
    if sh.shape[1] < K_t:
        pad = torch.zeros(sh.shape[0], K_t - sh.shape[1], 3,
                          dtype=sh.dtype, device=sh.device)
        gs["sh"] = torch.cat([sh, pad], dim=1)
    gs["sh_degree"] = target_degree
    return gs


def transform_gaussians(gs, T):
    """Apply similarity transform (scale folded in 4x4) to a gaussian dict."""
    import torch
    s, R, t = decompose_similarity(T)
    dev = gs["means"].device
    Rt = torch.from_numpy(R.astype(np.float32)).to(dev)
    tt = torch.from_numpy(t.astype(np.float32)).to(dev)
    out = dict(gs)
    out["means"] = gs["means"] * s @ Rt.T + tt
    out["scales"] = gs["scales"] * s
    qR = rot_to_quat_wxyz(R)
    q = quat_mul_wxyz(qR, gs["quats"].cpu().numpy().astype(np.float64))
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    out["quats"] = torch.from_numpy(q.astype(np.float32)).to(dev)
    return out


def cat_gaussians(gs_list):
    import torch
    deg = max(g["sh_degree"] for g in gs_list)
    gs_list = [pad_sh(dict(g), deg) for g in gs_list]
    return {
        "means": torch.cat([g["means"] for g in gs_list]),
        "quats": torch.cat([g["quats"] for g in gs_list]),
        "scales": torch.cat([g["scales"] for g in gs_list]),
        "opacities": torch.cat([g["opacities"] for g in gs_list]),
        "sh": torch.cat([g["sh"] for g in gs_list]),
        "sh_degree": deg,
    }


def render_view(gs, w2c, K, width, height, render_mode="RGB", scale=1.0,
                background=None):
    """gsplat full-SH rasterization -> rgb [H,W,3] in [0,1] (+depth if RGB+ED)."""
    import torch
    from gsplat import rasterization

    device = gs["means"].device
    K = np.asarray(K, dtype=np.float64).copy()
    if scale != 1.0:
        K[:2] *= scale
        width, height = int(round(width * scale)), int(round(height * scale))
    with torch.no_grad():
        viewmat = torch.from_numpy(np.asarray(w2c)).float().to(device)[None]
        Kt = torch.from_numpy(K).float().to(device)[None]
        bg = None
        if background is not None:
            # gsplat expects (C,3); it appends the depth column itself for ED
            bg = torch.tensor(background, dtype=torch.float32, device=device)[None]
        colors, alphas, _ = rasterization(
            means=gs["means"], quats=gs["quats"], scales=gs["scales"],
            opacities=gs["opacities"], colors=gs["sh"],
            viewmats=viewmat, Ks=Kt, width=width, height=height,
            sh_degree=gs["sh_degree"], render_mode=render_mode,
            backgrounds=bg, packed=False, near_plane=0.01, far_plane=100.0)
    out = colors[0]
    alpha = alphas[0, ..., 0].cpu().numpy()
    if render_mode == "RGB+ED":
        return (out[..., :3].clamp(0, 1).cpu().numpy(),
                out[..., 3].cpu().numpy(), alpha)
    return out.clamp(0, 1).cpu().numpy(), None, alpha


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def save_json(path, obj):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=default))
