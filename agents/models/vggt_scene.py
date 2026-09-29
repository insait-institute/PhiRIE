"""VGGT scene reconstruction: unposed frames -> cameras + depth + points.

Runs facebook/VGGT-1B (full model with camera+depth heads) on the frames of
one casual capture and writes a single npz:
    w2c    (N,4,4) float64  OpenCV world-to-camera (VGGT's native convention)
    K      (3,3)   float64  shared PINHOLE intrinsics at the ORIGINAL frame
                            resolution (median over VGGT's per-frame FoVs;
                            cx,cy at the image center, as VGGT assumes)
    names  (N,)    str      frame filenames, capture order
    depth  (N,Hp,Wp) float16  z-depth at VGGT's processing resolution
    K_depth (3,3)  float64  K rescaled to the depth resolution (extra key)
    points (M,3)  float32 + points_rgb (M,3) uint8: confidence-filtered
                            fused cloud, subsampled to <= 2M points
Scale is VGGT's arbitrary up-to-scale unit; agents.recon.metricize fixes it.

NOTE: we deliberately load facebook/VGGT-1B (in the HF cache), NOT
ReconViaGen's Stable-X/vggt-object-v0-1 - that checkpoint is an
object-centric finetune whose depth head RVG deletes; only the vendored
VGGT *code* is reused (same sys.path dance as agents/models/s4_reconviagen.py).

GPU, main .venv. Usage:
    python -m agents.models.vggt_scene --images-dir D --out recon.npz
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image

from agents.core import common as C

RVG_DIR = C.ROOT / "third_party" / "ReconViaGen"
sys.path.insert(0, str(RVG_DIR))
# vendored VGGT uses absolute self-imports (from vggt.models...)
sys.path.insert(0, str(RVG_DIR / "wheels" / "vggt"))

VGGT_CKPT = "facebook/VGGT-1B"
PROC_LONG = 518   # VGGT's training resolution; dims must be /14 (patch size)
# VGGT-Omega (CVPR'26): higher-quality successor, patch 16, 512-res variant.
# Weights are GATED on HF - request access at hf.co/facebook/VGGT-Omega, then
#   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt HF_TOKEN=... \
#   .venv/bin/python -c "from huggingface_hub import snapshot_download; \
#   snapshot_download('facebook/VGGT-Omega', local_dir='checkpoints/vggt-omega')"
OMEGA_DIR = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"
OMEGA_CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "vggt-omega"
OMEGA_PROC_LONG, OMEGA_PATCH = 512, 16
OMEGA_MAX_FRAMES_PER_FWD = 220  # 13.4GB@100 frames per their A100 table
# A6000 48GB fits ~100 frames at 518px in bf16 through the 1B aggregator
# (attention is quadratic in S*tokens); 96 leaves headroom for the heads.
MAX_FRAMES_PER_FWD = 96
CHUNK_OVERLAP = 12    # shared frames per chunk pair; Umeyama needs >=3,
                      # 12 keeps the similarity fit robust to a bad pose
CONF_KEEP_FRAC = 0.5  # keep the top half of depth pixels by confidence:
                      # VGGT conf is uncalibrated, a fixed quantile transfers
                      # across scenes better than an absolute threshold
MAX_POINTS = 2_000_000
FUSE_STRIDE = 2       # pixel stride before random subsample (4x cheaper)


def preprocess(paths, proc_long=PROC_LONG, patch=14):
    """Frames -> (S,3,Hp,Wp) float tensor in [0,1], dims /patch.

    Our own resize instead of the vendored load_and_preprocess_images: that
    copy was rewritten for RVG's (image, mask) tuples and center-crops tall
    images, which would silently break the intrinsics we export."""
    import torch
    w0, h0 = Image.open(paths[0]).size
    s = proc_long / max(w0, h0)
    wp = max(patch, round(w0 * s / patch) * patch)
    hp = max(patch, round(h0 * s / patch) * patch)
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (w0, h0):
            raise SystemExit(f"[vggt] frame size mismatch: {p}")
        im = im.resize((wp, hp), Image.Resampling.BICUBIC)
        imgs.append(torch.from_numpy(
            np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(imgs), (w0, h0), (wp, hp)


def umeyama_similarity(src, dst):
    """Least-squares s,R,t with dst ~= s*R@src + t (Umeyama 1991)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def run_chunk(model, images, device):
    """One VGGT forward -> (w2c (S,4,4), K_proc (S,3,3), depth (S,Hp,Wp),
    conf (S,Hp,Wp)) as float32/float64 numpy."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    hp, wp = images.shape[-2:]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images.to(device))
    extri, intri = pose_encoding_to_extri_intri(
        pred["pose_enc"].float(), image_size_hw=(hp, wp))
    extri = extri[0].cpu().numpy().astype(np.float64)      # (S,3,4) w2c
    intri = intri[0].cpu().numpy().astype(np.float64)      # (S,3,3)
    depth = pred["depth"][0, ..., 0].float().cpu().numpy() # (S,Hp,Wp)
    conf = pred["depth_conf"][0].float().cpu().numpy()
    w2c = np.tile(np.eye(4), (len(extri), 1, 1))
    w2c[:, :3, :] = extri
    return w2c, intri, depth.astype(np.float32), conf.astype(np.float32)


def run_chunk_omega(model, images, device):
    """VGGT-Omega forward, mapped to the exact run_chunk() contract.

    UNTESTED until the gated facebook/VGGT-Omega weights are downloaded -
    the API mirrors VGGT (pose_enc -> encoding_to_camera, depth/depth_conf
    heads); revisit against a GT-pose scene like the a29cccc784 pilot on
    first use."""
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

    hp, wp = images.shape[-2:]
    with torch.inference_mode(), \
            torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images.to(device))
    extri, intri = encoding_to_camera(pred["pose_enc"].float(),
                                      image_size_hw=(hp, wp))
    extri = extri[0].cpu().numpy().astype(np.float64)
    intri = intri[0].cpu().numpy().astype(np.float64)
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    w2c = np.tile(np.eye(4), (len(extri), 1, 1))
    w2c[:, :3, :] = extri[:, :3, :] if extri.shape[1] >= 3 else extri
    return w2c, intri, depth.astype(np.float32), conf.astype(np.float32)


def load_omega(device, *, source=None, checkpoint=None):
    sys.path.insert(0, str(source or OMEGA_DIR))
    import torch
    from vggt_omega.models import VGGTOmega
    ckpts = [Path(checkpoint)] if checkpoint is not None else sorted(list(OMEGA_CKPT_DIR.glob("*.pt"))
                   + list(OMEGA_CKPT_DIR.glob("*.pth"))
                   + list(OMEGA_CKPT_DIR.glob("*.safetensors")))
    if not ckpts:
        raise SystemExit(
            "[omega] no checkpoint in checkpoints/vggt-omega - the HF repo "
            "facebook/VGGT-Omega is gated; request access on huggingface.co, "
            "then download it there (see the note at the top of this file). "
            "Falling back: rerun with --backend vggt or colmap.")
    model = VGGTOmega().to(device).eval()
    if ckpts[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpts[0]))
    else:
        sd = torch.load(str(ckpts[0]), map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    return model


def fuse_points(w2c, K_depth, depth, conf, images_np):
    """Confidence-filtered multi-view unprojection -> world cloud."""
    pts, rgb = [], []
    st = FUSE_STRIDE
    for i in range(len(depth)):
        d = depth[i][::st, ::st].astype(np.float64)
        cf = conf[i][::st, ::st]
        img = images_np[i][::st, ::st]
        thr = np.quantile(cf, 1.0 - CONF_KEEP_FRAC)
        m = (cf >= thr) & np.isfinite(d) & (d > 1e-4)
        vv, uu = np.nonzero(m)
        z = d[vv, uu]
        u = (uu * st + 0.5 - K_depth[0, 2]) / K_depth[0, 0] * z
        v = (vv * st + 0.5 - K_depth[1, 2]) / K_depth[1, 1] * z
        c2w = np.linalg.inv(w2c[i])
        p = np.stack([u, v, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]
        pts.append(p.astype(np.float32))
        rgb.append(img[vv, uu])
    pts = np.concatenate(pts)
    rgb = np.concatenate(rgb)
    if len(pts) > MAX_POINTS:
        sel = np.random.RandomState(0).choice(len(pts), MAX_POINTS,
                                              replace=False)
        pts, rgb = pts[sel], rgb[sel]
    return pts, rgb


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", choices=["vggt", "omega"], default="vggt")
    ap.add_argument("--omega-source", type=Path)
    ap.add_argument("--omega-checkpoint", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seed < 0 or (args.backend != 'omega' and (args.omega_source or args.omega_checkpoint)):
        raise ValueError('invalid seed or Omega-only dependency override')
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite reconstruction: {args.out}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    paths = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < 2:
        raise SystemExit(f"[vggt] need >=2 frames in {args.images_dir}")
    names = [p.name for p in paths]

    if args.backend == "omega":
        images, (w0, h0), (wp, hp) = preprocess(
            paths, OMEGA_PROC_LONG, OMEGA_PATCH)
    else:
        images, (w0, h0), (wp, hp) = preprocess(paths)
    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    print(f"[{args.backend}] {len(paths)} frames {w0}x{h0} -> proc {wp}x{hp}")

    device = "cuda"
    if args.backend == "omega":
        model = load_omega(device, source=args.omega_source, checkpoint=args.omega_checkpoint)
        forward = run_chunk_omega
    else:
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained(VGGT_CKPT).to(device).eval()
        forward = run_chunk

    # chunk plan: [0, cap), then windows advancing by cap-overlap, each
    # aligned into the first chunk's frame via Umeyama on the shared cameras
    cap = OMEGA_MAX_FRAMES_PER_FWD if args.backend == "omega" \
        else MAX_FRAMES_PER_FWD
    ov = CHUNK_OVERLAP
    n = len(paths)
    starts = [0]
    while starts[-1] + cap < n:
        starts.append(starts[-1] + cap - ov)

    w2c_all = np.zeros((n, 4, 4))
    K_all = np.zeros((n, 3, 3))
    depth_all = np.zeros((n, hp, wp), np.float32)
    conf_all = np.zeros((n, hp, wp), np.float32)
    done = np.zeros(n, bool)
    for ci, s0 in enumerate(starts):
        idx = np.arange(s0, min(s0 + cap, n))
        w2c, intri, depth, conf = forward(model, images[idx], device)
        if ci > 0:
            shared = idx[done[idx]]
            li = shared - s0
            c_new = np.stack([np.linalg.inv(w2c[j])[:3, 3] for j in li])
            c_ref = np.stack([np.linalg.inv(w2c_all[j])[:3, 3]
                              for j in shared])
            s, R, t = umeyama_similarity(c_new, c_ref)
            print(f"[vggt] chunk {ci}: aligned on {len(li)} shared cams, "
                  f"scale {s:.4f}")
            for j in range(len(w2c)):
                c2w = np.linalg.inv(w2c[j])
                c2w[:3, :3] = R @ c2w[:3, :3]
                c2w[:3, 3] = s * R @ c2w[:3, 3] + t
                w2c[j] = np.linalg.inv(c2w)
            depth = depth * s
        w2c_all[idx], K_all[idx] = w2c, intri
        depth_all[idx], conf_all[idx] = depth, conf
        done[idx] = True
        torch.cuda.empty_cache()
    assert done.all()

    # shared K: median FoV over frames (VGGT predicts per-frame focals for
    # one physical camera; the median rejects the odd blurry-frame outlier)
    fx, fy = np.median(K_all[:, 0, 0]), np.median(K_all[:, 1, 1])
    K_depth = np.array([[fx, 0, wp / 2.0], [0, fy, hp / 2.0], [0, 0, 1.0]])
    # per-axis scale: preprocess() rounds wp/hp to multiples of 14
    # independently, so the resize is not exactly aspect-preserving
    K = K_depth.copy()
    K[0] *= w0 / wp
    K[1] *= h0 / hp

    points, points_rgb = fuse_points(w2c_all, K_depth, depth_all, conf_all,
                                     images_np)
    print(f"[vggt] fused {len(points)} points; "
          f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} @ {w0}x{h0}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, w2c=w2c_all, K=K, K_depth=K_depth,
        names=np.array(names), depth=depth_all.astype(np.float16),
        points=points.astype(np.float32), points_rgb=points_rgb,
        frame_wh=np.array([w0, h0]))
    print(f"[vggt] wrote {args.out}")


if __name__ == "__main__":
    main()
