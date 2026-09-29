"""SimAny demo movie segments.

Subcommands (env SIMANY_SCENE / SIMANY_OUT select the scene):
  scan            point-cloud -> splat radial reveal        (mini-viewer, GPU)
  discover        object glow + labels + crossfade to twin  (mini-viewer, GPU)
  interact-sim    scripted physics session -> poses json    (venv, CPU)
  interact-render poses json -> captioned photoreal mp4     (mini-viewer, GPU)
  assemble        title + segments + outro -> final mp4     (CPU)
"""
import argparse
import json
import os

import numpy as np

from agents.core import common as C

FPS = 30
# render scale vs intrinsics: 0.5 -> 876x584, 1.0 -> 1752x1168 (full res)
SCALE = float(C.env("DEMO_SCALE", "0.5"))
# object appearance in twin/interact segments: "asset" renders the generated
# TRELLIS/RVG gaussians; "orig" keeps every object's ORIGINAL scene gaussians
# (assets still drive physics; moving objects are rigidly re-posed originals)
OBJ_SRC = C.env("DEMO_OBJ_SRC", "asset")
CYAN = np.array([0.25, 0.95, 1.0])
GREEN = np.array([0.35, 1.0, 0.45])
C0 = 0.2820947917738781


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return 3 * t * t - 2 * t ** 3


def _font(size):
    from PIL import ImageFont
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_u8(rgb):
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def overlay_caption(u8, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(u8).convert("RGBA")
    W, H = im.size
    s = W / 876.0  # font/bar geometry authored at 876px width
    bh = int(34 * s)
    bar = Image.new("RGBA", (W, bh), (0, 0, 0, 135))
    im.alpha_composite(bar, (0, H - bh))
    d = ImageDraw.Draw(im)
    d.text((int(12 * s), H - int(29 * s)), text, font=_font(int(16 * s)),
           fill=(240, 240, 240, 235))
    return np.asarray(im.convert("RGB"))


def draw_labels(u8, labels):
    """labels: list of (u, v, text, alpha) in pixel coords."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(u8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    s = im.size[0] / 876.0
    f = _font(int(15 * s))
    for u, v, txt, a in labels:
        a = float(np.clip(a, 0, 1))
        if a <= 0.02:
            continue
        A = int(255 * a)
        d.line([(u, v), (u, v - 30 * s)], fill=(255, 255, 255, A),
               width=max(2, int(2 * s)))
        tw = d.textlength(txt, font=f)
        d.rectangle([u - tw / 2 - 5 * s, v - 52 * s,
                     u + tw / 2 + 5 * s, v - 30 * s],
                    fill=(0, 0, 0, int(150 * a)))
        d.text((u - tw / 2, v - 49 * s), txt, font=f,
               fill=(255, 255, 255, A))
    im.alpha_composite(ov)
    return np.asarray(im.convert("RGB"))


def writer_for(path):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=FPS, macro_block_size=None, quality=8)


# ------------------------------------------------------------- gaussians --

def sub(gs, mask):
    """Row-subset of a gaussian dict (torch bool mask)."""
    return {"means": gs["means"][mask], "quats": gs["quats"][mask],
            "scales": gs["scales"][mask], "opacities": gs["opacities"][mask],
            "sh": gs["sh"][mask], "sh_degree": gs["sh_degree"]}


def tint(gs, color, w):
    """Clone + mix base color toward `color` (0..1) with weight w."""
    import torch
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in gs.items()}
    c = torch.tensor(color, dtype=out["sh"].dtype, device=out["sh"].device)
    base = out["sh"][:, 0, :] * C0 + 0.5
    out["sh"][:, 0, :] = ((1 - w) * base + w * c - 0.5) / C0
    if out["sh"].shape[1] > 1:
        out["sh"][:, 1:, :] *= (1 - w)
    return out


def load_mesh_points(max_pts=900_000):
    """Scan-mesh vertices + colors via plyfile (no open3d in render env)."""
    from plyfile import PlyData
    v = PlyData.read(str(C.MESH_PLY))["vertex"]
    pts = np.stack([np.asarray(v[a], np.float32) for a in "xyz"], axis=1)
    names = {p.name for p in v.properties}
    if {"red", "green", "blue"} <= names:
        col = np.stack([np.asarray(v[n], np.float32) / 255.0
                        for n in ("red", "green", "blue")], axis=1)
    else:
        col = np.full_like(pts, 0.55)
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts, col = pts[idx], col[idx]
    return pts, col


def points_as_gaussians(pts, col, device="cuda", radius=0.004, opac=0.92):
    import torch
    n = len(pts)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.astype(np.float32))).to(device)
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    return {"means": t(pts), "quats": t(quats),
            "scales": t(np.full((n, 3), radius, np.float32)),
            "opacities": t(np.full(n, opac, np.float32)),
            "sh": t(((col - 0.5) / C0).reshape(n, 1, 3)), "sh_degree": 0}


# --------------------------------------------------------------- cameras --

def slerp_w2c(w2c_a, w2c_b, t):
    from scipy.spatial.transform import Rotation, Slerp
    c2w_a, c2w_b = np.linalg.inv(w2c_a), np.linalg.inv(w2c_b)
    rots = Rotation.from_matrix([c2w_a[:3, :3], c2w_b[:3, :3]])
    R = Slerp([0, 1], rots)([t])[0].as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return np.linalg.inv(c2w)


def pick_views(k=3, min_dist=0.9):
    """Views seeing the most discovered objects, spread >= min_dist apart."""
    K, W, H, _ = C.load_intrinsics()
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    cents = np.array([m["centroid"] for m in objects], dtype=np.float64)
    scored = []
    for fn, w2c in sorted(C.load_colmap_w2c().items()):
        pc = cents @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.clip(pc[:, 2], 1e-6, None)
        u = pc[:, 0] / z * K[0, 0] + K[0, 2]
        v = pc[:, 1] / z * K[1, 1] + K[1, 2]
        n = int(((z > 0.3) & (u > 0) & (u < W) & (v > 0) & (v < H)).sum())
        scored.append((n, fn, w2c))
    scored.sort(key=lambda x: -x[0])
    picks = [scored[0]]
    for n, fn, w2c in scored[1:]:
        if all(np.linalg.norm(np.linalg.inv(w2c)[:3, 3]
                              - np.linalg.inv(pw)[:3, 3]) > min_dist
               for _, _, pw in picks):
            picks.append((n, fn, w2c))
        if len(picks) == k:
            break
    return [w for _, _, w in picks]


def cam_path(keys, T):
    """Piecewise slerp through key w2cs, T frames total."""
    keys = list(keys)
    if len(keys) == 1:
        return [keys[0]] * T
    out = []
    for i in range(T):
        u = i / max(T - 1, 1) * (len(keys) - 1)
        a = min(int(u), len(keys) - 2)
        out.append(slerp_w2c(keys[a], keys[a + 1], u - a))
    return out


def project(pt, w2c, K):
    pc = np.asarray(pt) @ w2c[:3, :3].T + w2c[:3, 3]
    if pc[2] < 0.2:
        return None
    u = pc[0] / pc[2] * K[0, 0] + K[0, 2]
    v = pc[1] / pc[2] * K[1, 1] + K[1, 2]
    return u * SCALE, v * SCALE


_HOLE_PSNR_CACHE = {}


def _hole_psnr_by_object():
    """{obj_idx: worst-view hole-fill PSNR} joining inpaint/obj_XX/views.json
    (frame list per object) against inpaint/fill_stats.json (per-frame hole
    PSNR). Asset registration F1 and hole-fill PSNR are independent failure
    modes - a perfectly-aligned asset (F1~0.99) can still sit over a badly
    filled removal hole, which F1 alone can't see. Uses min (not mean)
    across the object's views: one bad frame is enough to show a visible
    blotch in a rendered fly-through, an average would paper over it."""
    if C.OUT in _HOLE_PSNR_CACHE:
        return _HOLE_PSNR_CACHE[C.OUT]
    out = {}
    fs_path = C.OUT / "inpaint" / "fill_stats.json"
    if fs_path.exists():
        by_frame = {}
        for r in json.loads(fs_path.read_text()):
            by_frame.setdefault(r["frame"], []).append(
                r["hole_psnr_vs_inpainted"])
        for vj in (C.OUT / "inpaint").glob("obj_*/views.json"):
            idx = int(vj.parent.name.split("_")[1])
            vals = [p for v in json.loads(vj.read_text())
                    for p in by_frame.get(v["frame"], [])]
            if vals:
                out[idx] = min(vals)
    _HOLE_PSNR_CACHE[C.OUT] = out
    return out


def demo_ok(m, al):
    """Demo curation: drop assets below the F1 gate, below the hole-fill
    PSNR gate, or explicitly excluded (SIMANY_DEMO_EXCLUDE=idx,idx). Excluded
    objects stay as original scene gaussians instead - covers bad assets
    AND their removal residue."""
    excl = C.env("DEMO_EXCLUDE", "").split(",")
    if str(m["index"]) in excl:
        return False
    f1 = al.get("eval", {}).get("f1@20mm", {}).get("f1", 1.0)
    if f1 < float(C.env("DEMO_F1MIN", "0.70")):
        return False
    hole = _hole_psnr_by_object().get(int(m["index"]))
    if hole is not None and hole < float(
            C.env("DEMO_HOLEMIN", "15.0")):
        return False
    return True


def load_assets(bg_deg):
    """Kept generated assets (scene frame, static) + demo-excluded metas."""
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    placed, metas, excluded = [], [], []
    for m in objects:
        odir = C.OUT / "objects" / f"obj_{int(m['index']):02d}"
        aj = odir / "aligned.json"
        gp = odir / "trellis_gs.ply"
        if not (aj.exists() and gp.exists()):
            continue
        al = json.loads(aj.read_text())
        if al.get("rejected"):
            continue
        if not demo_ok(m, al):
            excluded.append(m)
            continue
        gs = C.pad_sh(C.load_gaussians(gp), bg_deg)
        placed.append(C.transform_gaussians(gs, np.array(al["T"])))
        metas.append(m)
    return placed, metas, excluded


def instance_geometry():
    """Mesh points + GT/auto instance dicts keyed by object_id."""
    from plyfile import PlyData
    # vert_idx from load_instances() indexes the pipeline mesh (== MESH_PLY
    # unless SIMANY_MESH_SRC=derived), so read the same one
    v = PlyData.read(str(getattr(C, "PIPELINE_MESH_PLY", C.MESH_PLY)))["vertex"]
    mesh_pts = np.stack([np.asarray(v[a], np.float64) for a in "xyz"], axis=1)
    return mesh_pts, {g["object_id"]: g for g in C.load_instances()}


def splat_mask_near(gs, pts, r=0.03):
    """Bool mask of gaussians within r of the given points."""
    import torch
    from scipy.spatial import cKDTree
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 4000,
                                                  replace=False)]
    dist, _ = cKDTree(pts).query(gs["means"].cpu().numpy(), k=1,
                                 distance_upper_bound=r)
    return torch.from_numpy(np.isfinite(dist)).to(gs["means"].device)


def restore_excluded(orig, excluded, mesh_pts, insts):
    """Original scene gaussians of demo-excluded objects (to paste back)."""
    parts = []
    for m in excluded:
        g = insts.get(int(m["gt_object_id"]))
        if g is None:
            continue
        mask = splat_mask_near(orig, mesh_pts[g["vert_idx"]])
        if mask.any():
            parts.append(sub(orig, mask))
    return parts


def orig_object_gaussians(orig, metas, mesh_pts, insts):
    """{obj_XX: original scene gaussians} for the given object metas."""
    out = {}
    for m in metas:
        g = insts.get(int(m["gt_object_id"]))
        if g is None:
            continue
        mask = splat_mask_near(orig, mesh_pts[g["vert_idx"]])
        if mask.any():
            out[f"obj_{int(m['index']):02d}"] = sub(orig, mask)
    return out


def pose_mat(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = C.quat_to_rot_wxyz(quat_wxyz)
    T[:3, 3] = pos
    return T


# ------------------------------------------------------------------ scan --

def seg_scan(args):
    import torch
    gs = C.load_gaussians()
    # cull saturated color floaters (magenta blobs near bright windows)
    base = gs["sh"][:, 0, :] * C0 + 0.5
    ok = ((base > -0.3) & (base < 1.3)).all(dim=1)
    if int((~ok).sum()):
        print(f"[scan] culled {int((~ok).sum())} saturated gaussians")
        gs = sub(gs, ok)
    pts, col = load_mesh_points()
    pgs = points_as_gaussians(pts, col)

    keys = pick_views(3)
    F = int(args.seconds * FPS)
    cams = cam_path(keys, F)
    origin = torch.from_numpy(
        np.linalg.inv(keys[0])[:3, 3].astype(np.float32)).cuda()
    d_s = torch.linalg.norm(gs["means"] - origin, dim=1)
    d_p = torch.linalg.norm(pgs["means"] - origin, dim=1)
    rmax = float(torch.quantile(d_s, 0.995)) * 1.02

    Kin, W, H, _ = C.load_intrinsics()
    w = writer_for(args.out)
    cap = f"scene {C.SCENE_ID} - point cloud -> 3D Gaussian splat"
    if args.no_caption:
        cap = None
    for i in range(F):
        u = i / max(F - 1, 1)
        if u < 0.08:
            r = 0.0
        elif u > 0.92:
            r = rmax * 1.3
        else:
            r = rmax * smoothstep((u - 0.08) / 0.84)
        inside = d_s <= r
        band_s = inside & (d_s > r - 0.15)
        parts = []
        core = inside & ~band_s
        if core.any():
            parts.append(sub(gs, core))
        if band_s.any():
            parts.append(tint(sub(gs, band_s), CYAN, 0.55))
        p_out = d_p > r
        p_band = p_out & (d_p <= r + 0.12)
        p_far = p_out & ~p_band
        if p_far.any():
            parts.append(sub(pgs, p_far))
        if p_band.any():
            parts.append(tint(sub(pgs, p_band), CYAN, 0.75))
        rgb, _, _ = C.render_view(C.cat_gaussians(parts), cams[i],
                                  Kin, W, H, scale=SCALE)
        u8 = to_u8(rgb)
        w.append_data(u8 if cap is None else overlay_caption(u8, cap))
        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()
            print(f"[scan] {i + 1}/{F}", flush=True)
    w.close()
    print(f"[scan] wrote {args.out}")


def seg_grid(args):
    """2x2 montage of scan segments with staggered starts + one caption."""
    import imageio.v2 as imageio
    from PIL import Image
    rs = [imageio.get_reader(p) for p in args.inputs[:4]]
    ns = [r.count_frames() for r in rs]
    offs = [int(args.stagger * FPS * i) for i in range(len(rs))]
    total = max(n + o for n, o in zip(ns, offs)) + int(0.4 * FPS)
    h0, w0 = rs[0].get_data(0).shape[:2]
    tw, th = w0 // 2, h0 // 2
    ids = [os.path.basename(p).split(".")[0].split("_")[-1]
           for p in args.inputs[:4]]

    def tag(t, sid):
        from PIL import ImageDraw
        im = Image.fromarray(t).convert("RGBA")
        s = im.size[0] / 876.0
        d = ImageDraw.Draw(im, "RGBA")
        f = _font(int(15 * s))
        pad = int(6 * s)
        twd = d.textlength(sid, font=f)
        d.rectangle([pad, pad, pad * 3 + twd, pad + int(24 * s)],
                    fill=(0, 0, 0, 130))
        d.text((pad * 2, pad + int(3 * s)), sid, font=f,
               fill=(255, 255, 255, 220))
        return np.asarray(im.convert("RGB"))

    w = writer_for(args.out)
    for i in range(total):
        tiles = []
        for r, n, o, sid in zip(rs, ns, offs, ids):
            j = int(np.clip(i - o, 0, n - 1))
            t = np.asarray(Image.fromarray(r.get_data(j)).resize((tw, th)))
            tiles.append(tag(t, sid))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        w.append_data(overlay_caption(grid, args.caption))
        if (i + 1) % 60 == 0:
            print(f"[grid] {i + 1}/{total}", flush=True)
    w.close()
    print(f"[grid] wrote {args.out}")


# -------------------------------------------------------------- discover --

def seg_discover(args):
    import torch

    orig = C.load_gaussians()
    clean_ply = C.OUT / "inpaint" / "clean_background.ply"
    if not clean_ply.exists():
        raise SystemExit(f"[disc] {clean_ply} missing - run the inpaint "
                         "pipeline first (twin crossfade needs a clean bg)")
    clean = C.load_gaussians(clean_ply)
    placed, metas, excluded = load_assets(clean["sh_degree"])
    mesh_pts, insts = instance_geometry()
    if OBJ_SRC == "orig":
        # show every object with its ORIGINAL scene gaussians (assets are
        # still what the sim uses; the film keeps the real appearance)
        placed = list(orig_object_gaussians(orig, metas, mesh_pts,
                                            insts).values())
    twin = C.cat_gaussians([clean] + placed
                           + restore_excluded(orig, excluded, mesh_pts, insts))

    # highlight objects: label-deduped kept assets, best score first, <= 8
    best = {}
    for m in metas:
        lb = m["label"]
        if lb not in best or float(m["score"]) > float(best[lb]["score"]):
            best[lb] = m
    hi = sorted(best.values(), key=lambda m: -float(m["score"]))[:8]

    masks, cents = [], []
    for m in hi:
        g = insts.get(int(m["gt_object_id"]))
        if g is None:
            masks.append(None)
            cents.append(np.array(m["centroid"]))
            continue
        masks.append(splat_mask_near(orig, mesh_pts[g["vert_idx"]]))
        cents.append(np.array(g["centroid"]))
    K_hi = len(hi)

    # timeline
    T_HOLD, STAG, PULSE = 2.0, 0.75, 1.2
    t_i = [T_HOLD + STAG * i for i in range(K_hi)]
    t_fade0 = t_i[-1] + PULSE + 1.0
    FADE, T_TWIN = 1.5, 1.5
    total = t_fade0 + FADE + T_TWIN
    F = int(total * FPS)

    keys = pick_views(2, min_dist=0.6)
    cams = cam_path(keys + keys[:1][::-1], F) if len(keys) == 1 \
        else cam_path(keys, F)
    Kin, W, H, _ = C.load_intrinsics()
    Ksc = np.asarray(Kin, np.float64)

    w = writer_for(args.out)
    for i in range(F):
        t = i / FPS
        a = smoothstep((t - t_fade0) / FADE)
        w2c = cams[i]

        if a < 1.0:
            active = []
            for j in range(K_hi):
                dt = t - t_i[j]
                if masks[j] is not None and 0 < dt < PULSE:
                    active.append((j, float(np.sin(np.pi * dt / PULSE))))
            if active:
                any_m = torch.zeros_like(masks[active[0][0]])
                for j, _ in active:
                    any_m |= masks[j]
                parts = [sub(orig, ~any_m)]
                for j, wgt in active:
                    parts.append(tint(sub(orig, masks[j]), GREEN, 0.65 * wgt))
                frame_gs = C.cat_gaussians(parts)
            else:
                frame_gs = orig
            rgb, _, _ = C.render_view(frame_gs, w2c, Kin, W, H, scale=SCALE)
        if a > 0.0:
            rgb_t, _, _ = C.render_view(twin, w2c, Kin, W, H, scale=SCALE)
            rgb = rgb_t if a >= 1.0 else (1 - a) * rgb + a * rgb_t

        u8 = to_u8(rgb)
        labels = []
        for j in range(K_hi):
            if t < t_i[j]:
                continue
            la = min(1.0, (t - t_i[j]) / 0.4) * (1.0 - a)
            uv = project(cents[j], w2c, Ksc)
            if uv is not None:
                labels.append((uv[0], uv[1], hi[j]["label"], la))
        u8 = draw_labels(u8, labels)

        if t < T_HOLD:
            cap = "the twin understands what it sees"
        elif a < 0.5:
            cap = "SAM3 discovery -> per-object generation -> Gaussian-native removal"
        else:
            cap = "every object is now a simulation-ready asset"
        w.append_data(overlay_caption(u8, cap))
        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()
            print(f"[disc] {i + 1}/{F}", flush=True)
    w.close()
    print(f"[disc] wrote {args.out}")


# -------------------------------------------------------------- interact --

BIG_PREF = ["keyboard", "laptop", "box", "book", "tray", "plate"]
SMALL_PREF = ["bottle", "mug", "cup", "can", "vase"]
PUSH_PREF = ["mouse", "phone", "remote", "bowl", "bottle", "mug"]


def seg_interact_sim(args):
    import pybullet as p
    from robo.sim.s7_sim import build_background, com_pose, link_pose

    HZ = 240
    spf = HZ // FPS
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    aligneds = [json.loads(
        (C.OUT / "objects" / f"obj_{int(m['index']):02d}" / "aligned.json")
        .read_text()) for m in objects]
    kept = [(m, a) for m, a in zip(objects, aligneds)
            if not a.get("rejected") and demo_ok(m, a)]
    gts = {g["object_id"]: g for g in C.load_instances()}
    bg_path = build_background([m for m, _ in kept], [a for _, a in kept], gts)

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / HZ)
    col = p.createCollisionShape(p.GEOM_MESH, fileName=str(bg_path),
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    p.changeDynamics(p.createMultiBody(0, col), -1, lateralFriction=0.6)

    bodies, labels = {}, {}
    for m, a in kept:
        name = f"obj_{int(m['index']):02d}"
        odir = C.OUT / "objects" / name
        ph = json.loads((odir / "physics.json").read_text())
        s, R, t = C.decompose_similarity(np.array(a["T"]))
        q = C.rot_to_quat_wxyz(R)
        bid = p.loadURDF(str(odir / "object.urdf"), basePosition=t,
                         baseOrientation=[q[1], q[2], q[3], q[0]],
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(bid, -1, lateralFriction=ph["friction"],
                         restitution=ph["restitution"],
                         linearDamping=0.05, angularDamping=0.05)
        bodies[name], labels[name] = bid, m["label"]

    frames, targets = [], []
    focus = {"bid": None, "start": None}
    home = np.mean([link_pose(p, b)[0] for b in bodies.values()], axis=0) \
        if bodies else np.zeros(3)
    z_floor = min([link_pose(p, b)[0][2] for b in bodies.values()],
                  default=0.0)

    def set_focus(bid):
        focus["bid"] = bid
        focus["start"] = np.array(
            p.getBasePositionAndOrientation(bid)[0])

    def record():
        fr = {}
        for n, b in bodies.items():
            lp, lq = link_pose(p, b)
            fr[n] = [lp, [lq[3], lq[0], lq[1], lq[2]]]
        frames.append(fr)
        tgt = home
        if focus["bid"] is not None:
            pos = np.array(p.getBasePositionAndOrientation(focus["bid"])[0])
            # freeze gaze if the actor escaped the scene (fell through a
            # mesh hole / launched away) - keep looking at the last spot
            if (np.linalg.norm(pos - focus["start"]) < 3.0
                    and pos[2] > z_floor - 0.5):
                tgt = pos
            elif targets:
                tgt = np.array(targets[-1])
        targets.append([float(x) for x in tgt])

    def sim(seconds, zero_all=False):
        for _ in range(int(seconds * FPS)):
            for _ in range(spf):
                p.stepSimulation()
                if zero_all:
                    for b in bodies.values():
                        p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])
            record()

    def pick(prefs, used):
        for lb in prefs:
            for n in bodies:
                if n not in used and lb in labels[n]:
                    return n
        return None

    def drag(name, dpos, dyaw_deg, seconds):
        from scipy.spatial.transform import Rotation
        bid = bodies[name]
        lp0, lq0 = link_pose(p, bid)
        for i in range(int(seconds * FPS)):
            t = smoothstep((i + 1) / (seconds * FPS))
            lp = np.array(lp0) + t * np.array(dpos)
            rz = Rotation.from_euler("z", t * dyaw_deg, degrees=True)
            lq = (rz * Rotation.from_quat(lq0)).as_quat()
            p.resetBasePositionAndOrientation(
                bid, *com_pose(p, bid, lp.tolist(), lq.tolist()))
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
            for _ in range(spf):
                p.stepSimulation()
                p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
            record()

    used, beats = set(), []

    def beat(name, kind):
        beats.append({"f0": len(frames), "actor": name, "kind": kind})

    def beat_end():
        beats[-1]["f1"] = len(frames)

    sim(1.0, zero_all=True)
    big = pick(BIG_PREF, used)
    if big:
        used.add(big)
        set_focus(bodies[big])
        beat(big, "carry")
        drag(big, [0.45, 0.35, 0.55], 60, 2.5)   # big sweeping carry
        sim(2.0)                                  # release -> tumble down
        beat_end()
    small = pick(SMALL_PREF, used)
    if small:
        used.add(small)
        set_focus(bodies[small])
        beat(small, "throw")
        # springier for a readable bounce (demo-only, params are estimates)
        p.changeDynamics(bodies[small], -1, restitution=0.55)
        drag(small, [0.0, 0.10, 0.45], 0, 1.2)   # lift high...
        p.resetBaseVelocity(bodies[small], [0.9, 0.5, 0.4], [4, 0, 0])
        sim(2.2)                                  # ...and THROW: arc + bounce
        beat_end()
    push = pick(PUSH_PREF, used)
    if push:
        used.add(push)
        set_focus(bodies[push])
        beat(push, "push")
        bid = bodies[push]
        mass = p.getDynamicsInfo(bid, -1)[0]
        fx, fy = 2.4 * mass / 0.5, 1.2 * mass / 0.5   # impulse -> ~2.7 m/s
        for _ in range(int(0.5 * FPS)):
            for _ in range(spf):
                p.applyExternalForce(
                    bid, -1, [fx, fy, 0],
                    p.getBasePositionAndOrientation(bid)[0], p.WORLD_FRAME)
                p.stepSimulation()
            record()
        sim(1.7)                                  # shoved away, stays in room
        beat_end()

    keys = pick_views(3)
    cams = [m.tolist() for m in cam_path(keys, len(frames))]

    # per-beat action cameras: closest DSLR position with a clear line of
    # sight to that beat's action (mesh raycast) - close-ups, hard cuts
    scene = C.make_raycast_scene()
    import open3d as o3d

    RANGES = {"carry": (1.6, 3.2), "throw": (1.2, 2.6), "push": (1.0, 2.2)}

    def frame_clear(pos, tgt, d):
        """Center ray clear AND a wide frustum sample isn't jammed against
        nearby geometry (e.g. a shelf right above the camera) even when the
        direct line of sight to the target looks fine."""
        f = (tgt - pos) / d
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        r = r / (np.linalg.norm(r) + 1e-9)
        u = np.cross(r, f)
        rays = [np.concatenate([pos, f])]
        for off in (0.5 * r, -0.5 * r, 0.5 * u, -0.5 * u):
            dv = f + off
            rays.append(np.concatenate([pos, dv / np.linalg.norm(dv)]))
        rays = o3d.core.Tensor(np.stack(rays).astype(np.float32))
        hits = scene.cast_rays(rays)["t_hit"].numpy()
        return bool(np.all(hits > 0.55 * d))

    def sight_pos(tgt, kind=None):
        lo0, hi0 = RANGES.get(kind, (1.2, 2.6))
        best_pos, best_d = None, 1e9
        for lo, hi in [(lo0, hi0), (0.9, 5.0), (0.9, 12.0)]:
            for fn, w2c in sorted(C.load_colmap_w2c().items()):
                pos = np.linalg.inv(w2c)[:3, 3]
                d = float(np.linalg.norm(pos - tgt))
                if not (lo < d < hi) or d >= best_d:
                    continue
                if pos[2] < tgt[2] + 0.15:   # stay above the action plane
                    continue
                if frame_clear(pos, tgt, d):
                    best_pos, best_d = pos, d
            if best_pos is not None:
                return best_pos, best_d
        return np.linalg.inv(keys[0])[:3, 3], -1.0

    tarr = np.array(targets)
    for b in beats:
        bt = tarr[b["f0"]:b["f1"]].mean(axis=0)
        pos, d = sight_pos(bt, b.get("kind"))
        b["campos"] = [float(x) for x in pos]
        b["cam_d"] = round(d, 2)
    overall, d_all = sight_pos(tarr.mean(axis=0))
    C.save_json(args.out, {"fps": FPS, "frames": frames, "cams": cams,
                           "targets": targets, "beats": beats,
                           "campos": [float(x) for x in overall]})
    print(f"[isim] {len(frames)} frames, actors big={big} small={small} "
          f"push={push} beats={[(b['actor'], b['cam_d']) for b in beats]} "
          f"-> {args.out}")


def lookat_w2c(pos, target):
    """OpenCV-convention w2c looking from pos at target, world-up +z."""
    f = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross([0.0, 0.0, -1.0], f)
    n = np.linalg.norm(r)
    r = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else r / n
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, pos
    return np.linalg.inv(c2w)


def seg_interact_render(args):
    import torch
    log = json.loads(open(args.poses).read())
    clean = C.OUT / "inpaint" / "clean_background.ply"
    bg = C.load_gaussians(clean if clean.exists() else C.SPLAT_PLY)
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    canon, scales, kept, excluded = {}, {}, [], []
    for m in objects:
        name = f"obj_{int(m['index']):02d}"
        odir = C.OUT / "objects" / name
        aj = odir / "aligned.json"
        gp = odir / "trellis_gs.ply"
        if not (aj.exists() and gp.exists()):
            continue
        al = json.loads(aj.read_text())
        if al.get("rejected"):
            continue
        if not demo_ok(m, al):
            excluded.append(m)
            continue
        kept.append(m)
        if OBJ_SRC != "orig":
            canon[name] = C.pad_sh(C.load_gaussians(gp), bg["sh_degree"])
            scales[name] = C.decompose_similarity(np.array(al["T"]))[0]

    origobj = {}
    if (excluded or OBJ_SRC == "orig") and clean.exists():
        # original gaussians: paste back excluded objects' residue, and in
        # orig mode carry every kept object's real appearance through the sim
        orig = C.load_gaussians()
        mesh_pts, insts = instance_geometry()
        if OBJ_SRC == "orig":
            # animate every object the poses json recorded - it is the source
            # of truth for what the sim moved; the demo_ok kept set can drift
            # after the fact (e.g. audit-recomputed F1) and would strand a
            # moving actor as a static paste-back
            frame0 = log["frames"][0]
            moving = [m for m in kept + excluded
                      if f"obj_{int(m['index']):02d}" in frame0]
            origobj = orig_object_gaussians(orig, moving, mesh_pts, insts)
            excluded = [m for m in excluded
                        if f"obj_{int(m['index']):02d}" not in frame0]
        parts = restore_excluded(orig, excluded, mesh_pts, insts)
        if parts:
            bg = C.cat_gaussians([bg] + parts)
        del orig
        torch.cuda.empty_cache()
    if OBJ_SRC == "orig" and not clean.exists():
        raise SystemExit("[irend] DEMO_OBJ_SRC=orig needs inpaint/"
                         "clean_background.ply (else objects would double)")
    # frame-0 sim pose ~= aligned pose ~= where the original gaussians sit,
    # so per-frame placement is the rigid motion relative to frame 0
    P0inv = {n: np.linalg.inv(pose_mat(p, q))
             for n, (p, q) in log["frames"][0].items()} \
        if OBJ_SRC == "orig" else {}

    Kin, W, H, _ = C.load_intrinsics()
    tgts = log.get("targets")
    if tgts is not None:
        # per-beat close-up cameras with hard cuts; within a beat the
        # position dollies gently toward the action and the gaze (EMA)
        # tracks the actor. Settle frames use the overall position.
        ZOOM = 1.25
        Kin = np.asarray(Kin, np.float64).copy()
        Kin[0, 0] *= ZOOM
        Kin[1, 1] *= ZOOM
        F = len(log["frames"])
        tarr = np.array(tgts)
        beats = log.get("beats") or []
        spans = []
        prev_end = 0
        overall = np.array(log.get("campos",
                                   np.linalg.inv(pick_views(1)[0])[:3, 3]))
        for b in beats:
            if b["f0"] > prev_end:
                spans.append((prev_end, b["f0"], overall))
            spans.append((b["f0"], b["f1"], np.array(b["campos"])))
            prev_end = b["f1"]
        if prev_end < F:
            spans.append((prev_end, F, overall))
        cams = [None] * F
        for f0, f1, p0 in spans:
            seg_t = tarr[f0:f1].mean(axis=0)
            fwd = (seg_t - p0) / (np.linalg.norm(seg_t - p0) + 1e-9)
            ema = tarr[f0].copy()
            n = max(f1 - f0, 1)
            for i in range(f0, f1):
                ema = 0.78 * ema + 0.22 * tarr[i]
                u = smoothstep((i - f0) / n)
                cams[i] = lookat_w2c(p0 + 0.30 * u * fwd, ema)
        log["cams"] = [m.tolist() for m in cams]

    w = writer_for(args.out)
    cap = "PyBullet dynamics + Gaussian rendering - drag, drop, shove"
    for fi, fr in enumerate(log["frames"]):
        w2c = np.array(log["cams"][fi])
        parts = [bg]
        for name, (pos, quat) in fr.items():
            if name in origobj:
                parts.append(C.transform_gaussians(
                    origobj[name], pose_mat(pos, quat) @ P0inv[name]))
            elif name in canon:
                T = np.eye(4)
                T[:3, :3] = scales[name] * C.quat_to_rot_wxyz(quat)
                T[:3, 3] = pos
                parts.append(C.transform_gaussians(canon[name], T))
        rgb, _, _ = C.render_view(C.cat_gaussians(parts), w2c,
                                  Kin, W, H, scale=SCALE)
        w.append_data(overlay_caption(to_u8(rgb), cap))
        if (fi + 1) % 30 == 0:
            torch.cuda.empty_cache()
            print(f"[irend] {fi + 1}/{len(log['frames'])}", flush=True)
    w.close()
    print(f"[irend] wrote {args.out}")


# ---------------------------------------------------------------- replay --

def seg_replay(args):
    """Render a viewer-recorded take photoreal: clean background + every
    object as its ORIGINAL scene gaussians, moved rigidly by the recorded
    gizmo poses, seen through the recorded browser camera (viser cameras are
    OpenCV/COLMAP convention, wxyz+position = c2w)."""
    import torch
    take = json.loads(open(args.take).read())
    fps = int(take.get("fps", FPS))
    frames = take["frames"]
    if not frames:
        raise SystemExit("[replay] empty take")

    clean = C.OUT / "inpaint" / "clean_background.ply"
    if not clean.exists():
        raise SystemExit(f"[replay] {clean} missing - replay needs the "
                         "inpainted background (else moved objects double)")
    bg = C.load_gaussians(clean)

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    in_take = set(frames[0])
    metas = []
    for m in objects:
        name = f"obj_{int(m['index']):02d}"
        aj = C.OUT / "objects" / name / "aligned.json"
        if name not in in_take or not aj.exists():
            continue
        if json.loads(aj.read_text()).get("rejected"):
            continue
        metas.append(m)
    orig = C.load_gaussians()
    mesh_pts, insts = instance_geometry()
    origobj = orig_object_gaussians(orig, metas, mesh_pts, insts)
    del orig
    torch.cuda.empty_cache()
    # original gaussians sit at the registered (home) pose; per-frame
    # placement is the rigid motion relative to that home pose
    Hinv = {n: np.linalg.inv(pose_mat(p, q))
            for n, (p, q) in take["home"].items() if n in origobj}

    # cameras: forward/back-fill frames sampled with no client connected
    cams = list(take["cams"])
    first = next((c for c in cams if c), None)
    if first is None:
        raise SystemExit("[replay] take has no camera samples")
    last = first
    for i, c in enumerate(cams):
        last = c or last
        cams[i] = last
    _, W, H, _ = C.load_intrinsics()
    w2cs, fys = [], []
    for c in cams:
        c2w = np.eye(4)
        c2w[:3, :3] = C.quat_to_rot_wxyz(c["wxyz"])
        c2w[:3, 3] = c["position"]
        w2cs.append(np.linalg.inv(c2w))
        fys.append((H / 2.0) / np.tan(0.5 * float(c["fov"])))
    if args.smooth > 0:  # EMA against hand-jitter / discrete zoom steps
        sm, fy_s = w2cs[0], fys[0]
        for i in range(len(w2cs)):
            sm = slerp_w2c(sm, w2cs[i], 1.0 - args.smooth)
            fy_s = args.smooth * fy_s + (1 - args.smooth) * fys[i]
            w2cs[i], fys[i] = sm, fy_s

    import imageio.v2 as imageio
    w = imageio.get_writer(args.out, fps=fps, macro_block_size=None,
                           quality=8)
    for fi, fr in enumerate(frames):
        parts = [bg]
        for name, (pos, quat) in fr.items():
            if name in origobj:
                parts.append(C.transform_gaussians(
                    origobj[name], pose_mat(pos, quat) @ Hinv[name]))
        K = np.array([[fys[fi], 0, W / 2.0],
                      [0, fys[fi], H / 2.0], [0, 0, 1]])
        rgb, _, _ = C.render_view(C.cat_gaussians(parts), w2cs[fi],
                                  K, W, H, scale=SCALE)
        u8 = to_u8(rgb)
        if args.caption:
            u8 = overlay_caption(u8, args.caption)
        w.append_data(u8)
        if (fi + 1) % 30 == 0:
            torch.cuda.empty_cache()
            print(f"[replay] {fi + 1}/{len(frames)}", flush=True)
    w.close()
    print(f"[replay] wrote {args.out}")


# -------------------------------------------------------------- assemble --

TITLE = "SimAny"
SUBTITLE = "from a raw scan to an interactive world"
SUB2 = "posed RGB + reconstruction in - simulation-ready digital twin out"
STATS = "50 scenes  |  789 objects  |  11 min/scene on one A6000  |  zero annotations"


def make_card(size, lines):
    """lines: list of (text, font_size, dy), geometry authored at 876px."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", size, (8, 8, 10))
    d = ImageDraw.Draw(im)
    s = size[0] / 876.0
    y = size[1] / 2
    for txt, fs, dy in lines:
        f = _font(int(fs * s))
        tw = d.textlength(txt, font=f)
        d.text(((size[0] - tw) / 2, y + dy * s), txt, font=f,
               fill=(235, 235, 235))
    return np.asarray(im)


def seg_frames(path):
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    n = r.count_frames()
    return r, n


def assemble(args):
    from PIL import Image, ImageDraw

    r0, _ = seg_frames(args.segments[0])
    H, W = r0.get_data(0).shape[:2]
    size = (W, H)

    FADE = 8  # frames, fade through black at segment boundaries

    def faded(frame, i, n):
        g = 1.0
        if i < FADE:
            g = i / FADE
        if n - 1 - i < FADE:
            g = min(g, (n - 1 - i) / FADE)
        return (frame.astype(np.float32) * g).astype(np.uint8)

    w = writer_for(args.out)

    # title card
    card = make_card(size, [(TITLE, 64, -90), (SUBTITLE, 26, 20),
                            (SUB2, 17, 70)])
    nT = int(2.2 * FPS)
    for i in range(nT):
        w.append_data(faded(card, i, nT))

    stills = []
    for p in args.segments:
        r, n = seg_frames(p)
        for i in range(n):
            w.append_data(faded(r.get_data(i), i, n))
        stills.append(r.get_data(max(0, n - FADE - 5)))
        r.close()

    # outro: grid montage + stats, slow zoom, fade out
    base = list(stills)
    while len(stills) < 6:
        stills.append(base[len(stills) % len(base)])
    tiles = [np.asarray(Image.fromarray(s).resize((W // 3, H // 2)))
             for s in stills[:6]]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    grid = np.asarray(Image.fromarray(grid).resize(size))
    nO = int(5.0 * FPS)
    sw = W / 876.0
    fs = int(22 * sw)
    f_stats = _font(fs)
    d0 = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    while fs > 10 and d0.textlength(STATS, font=f_stats) > W - 40 * sw:
        fs -= 1
        f_stats = _font(fs)
    for i in range(nO):
        z = 1.0 + 0.08 * i / nO
        zw, zh = int(W / z), int(H / z)
        x0, y0 = (W - zw) // 2, (H - zh) // 2
        fr = np.asarray(Image.fromarray(
            grid[y0:y0 + zh, x0:x0 + zw]).resize(size))
        im = Image.fromarray(fr).convert("RGBA")
        d = ImageDraw.Draw(im)
        tw = d.textlength(STATS, font=f_stats)
        bh = int(44 * sw)
        bar = Image.new("RGBA", (W, bh), (0, 0, 0, 170))
        im.alpha_composite(bar, (0, H // 2 - bh // 2))
        d = ImageDraw.Draw(im)
        d.text(((W - tw) / 2, H // 2 - int(14 * sw)), STATS, font=f_stats,
               fill=(255, 255, 255, 255))
        fr = np.asarray(im.convert("RGB"))
        g = min(1.0, (nO - 1 - i) / (0.8 * FPS))
        w.append_data((fr.astype(np.float32) * g).astype(np.uint8))
    w.close()
    print(f"[asm] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--out", required=True)
    s.add_argument("--seconds", type=float, default=6.5)
    s.add_argument("--no-caption", action="store_true")
    s = sp.add_parser("grid")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--stagger", type=float, default=0.5)
    s.add_argument("--caption",
                   default="4 scenes - point cloud -> 3D Gaussian splat")
    s = sp.add_parser("discover")
    s.add_argument("--out", required=True)
    s = sp.add_parser("interact-sim")
    s.add_argument("--out", required=True)
    s = sp.add_parser("interact-render")
    s.add_argument("--poses", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("replay")
    s.add_argument("--take", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--smooth", type=float, default=0.6,
                   help="camera EMA weight, 0 = raw recorded camera")
    s.add_argument("--caption", default="")
    s = sp.add_parser("assemble")
    s.add_argument("--segments", nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"scan": seg_scan, "grid": seg_grid, "discover": seg_discover,
     "interact-sim": seg_interact_sim,
     "interact-render": seg_interact_render, "replay": seg_replay,
     "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
