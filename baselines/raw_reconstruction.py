"""Ablation baseline: raw fused reconstruction, NO object factorization.

plan/15_CONSTRUCTION_BASELINES.md lists "raw fused reconstruction / no
object factorization" as the first construction baseline. This is the
counterfactual for every "per-object factorization" claim the main SimAny
pipeline makes: take EXACTLY the upstream capture + reconstruction outputs
(COLMAP-posed frames -> trained 3DGS splat, plus a scan mesh -- either the
dataset's own or one fused from the splat, see
`agents/discover/derive_mesh_from_splat.py`) and use them as-is. Background
and every object stay baked into one static representation. No 2D/3D
instance discovery, no mask-to-gaussian removal, no per-object mesh/asset
generation, no Sim(3) registration, no CoACD collision decomposition, no
physics annotation, no articulation -- zero pipeline stages beyond capture
and splat training run.

This is our own ablation, not a third-party release: there is no "official
code" to pin, because "the reconstruction with nothing done to it" has no
author but the upstream capture/training tooling this repo already runs.
See `baselines/release_status.yaml` (status: implemented) and
`docs/BASELINE_REPRODUCTION.md`.

Scene-dir conventions (read-only; this module never writes into a scene
dir), matching `agents/core/common.py`'s layout:
  <root>/data/<scene_id>/scans/mesh_aligned_0.05.ply     ScanNet++ GT mesh
  <root>/data/<scene_id>/scans/segments.json,
        .../segments_anno.json                            ScanNet++ GT instances (optional context only)
  <root>/data/<scene_id>/dslr/nerfstudio/transforms_undistorted.json  intrinsics
  <root>/data/<scene_id>/dslr/colmap/images.txt           COLMAP w2c poses
  <root>/data/<scene_id>/dslr/resized_undistorted_images/ posed RGB frames
  <root>/splats/<scene_id>.ply                            trained 3DGS splat
        (this repo's own `data/recon_scenes/` layout; ScanNet++v2 gsplat
        instead defaults to /data/ScanNetppv2_gsplat/splats/<scene_id>.ply)
  <scene_dir>/derived_mesh.ply                            optional splat-only
        TSDF-fused mesh substitute, product of
        `agents.discover.derive_mesh_from_splat render && ... fuse` (NOT
        invoked by this script -- if absent and no dataset mesh exists,
        mesh-dependent fields are reported N/A with instructions, never
        fabricated).

Usage (CPU-only default path; works without a GPU):
  .venv/bin/python -m baselines.raw_reconstruction \
    --scene-dir data/recon_scenes/data/<scene> \
    --out-dir outputs/<scene>_baselines/raw_reconstruction

Optional GPU render self-check (needs CUDA + gsplat; separate from, and NOT
comparable to, the main pipeline's official held-out DSLR-test-split render
protocol -- see `_render_selfcheck` docstring):
  .venv/bin/python -m baselines.raw_reconstruction \
    --scene-dir data/recon_scenes/data/<scene> --render-eval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Oracle-table metric names the main pipeline reports (see docs/BASELINES.md
# and plan/14_ORACLE_BEHAVIOR_BENCHMARK.md-style rows). By construction, a
# no-factorization baseline cannot produce any of these -- they are always
# reported N/A with an explicit reason, never as a fabricated zero (task
# acceptance criterion: "Mark unsupported cells N/A, never zero.").
NA_METRICS = {
    "instance_discovery_f1":
        "no per-object instance discovery is performed by raw_reconstruction "
        "by construction (single fused body, background+objects combined)",
    "removal_iou":
        "no object removal is performed; nothing is factored out of the "
        "fused splat/mesh, so there is no removal set to score",
    "registration_error_m":
        "no per-object Sim(3) registration is performed; there are no "
        "discovered/generated object poses to compare against GT",
    "stability_ratio":
        "no per-object rigid bodies exist to drop/settle; the scene is one "
        "static fused body with no dynamics to evaluate",
    "penetration_count":
        "no per-object collision geometry exists to test object-object or "
        "object-floor penetration",
    "policy_success_rate":
        "no individually manipulable objects exist for a policy to interact "
        "with under this baseline",
}


# ----------------------------------------------------------------- helpers --

def _sha256_head(path: Path, n_bytes: int = 8 * 1024 * 1024) -> str | None:
    """Hash of the first n_bytes only. These plys run hundreds of MB; a head
    hash is cheap provenance-tagging, not a strong integrity check."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _na(reason: str) -> dict:
    return {"value": None, "status": "not_applicable", "reason": reason}


def _err(exc: Exception) -> dict:
    return {"value": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------- input resolution --

def resolve_inputs(scene_dir, mesh_ply=None, splat_ply=None, splats_root=None):
    """Locate the mesh + splat this ablation will treat as "the whole scene",
    honouring both the ScanNet++ layout and this repo's own
    data/recon_scenes/data/<scene> layout (see module docstring). Returns
    (scene_id, mesh_path_or_None, mesh_reason_or_None,
     splat_path_or_None, splat_reason_or_None, candidates_dict) --
    candidates_dict records every path checked, for the report's audit
    trail (never silently guess which file "must be" the right one)."""
    scene_dir = Path(scene_dir)
    root = scene_dir.parent.parent  # <root>/data/<scene_id>
    scene_id = scene_dir.name

    mesh_candidates = []
    if mesh_ply:
        mesh_candidates.append(("explicit --mesh-ply", Path(mesh_ply)))
    mesh_candidates.append(("scans/mesh_aligned_0.05.ply (ScanNet++ GT scan mesh)",
                             scene_dir / "scans" / "mesh_aligned_0.05.ply"))
    mesh_candidates.append(("derived_mesh.ply (splat-only TSDF fusion product; "
                             "see agents.discover.derive_mesh_from_splat)",
                             scene_dir / "derived_mesh.ply"))
    mesh_path, mesh_reason = None, None
    for _, p in mesh_candidates:
        if p.exists():
            mesh_path = p
            break
    if mesh_path is None:
        mesh_reason = ("no mesh found; checked " +
                        "; ".join(str(p) for _, p in mesh_candidates) +
                        ". For a splat-only capture with no dataset mesh, run "
                        "`agents.discover.derive_mesh_from_splat render` then "
                        "`... fuse` first to produce derived_mesh.ply.")

    splat_candidates = []
    if splat_ply:
        splat_candidates.append(("explicit --splat-ply", Path(splat_ply)))
    splat_candidates.append(("<root>/splats/<scene_id>.ply (recon_scenes convention)",
                              root / "splats" / f"{scene_id}.ply"))
    if splats_root:
        splat_candidates.append(("--splats-root/<scene_id>.ply",
                                  Path(splats_root) / f"{scene_id}.ply"))
    splat_candidates.append(("/data/ScanNetppv2_gsplat/splats/<scene_id>.ply "
                              "(ScanNet++v2 gsplat default)",
                              Path("/data/ScanNetppv2_gsplat/splats") / f"{scene_id}.ply"))
    splat_path, splat_reason = None, None
    for _, p in splat_candidates:
        if p.exists():
            splat_path = p
            break
    if splat_path is None:
        splat_reason = ("no trained splat found; checked " +
                         "; ".join(str(p) for _, p in splat_candidates))

    candidates = {
        "mesh_checked": [str(p) for _, p in mesh_candidates],
        "splat_checked": [str(p) for _, p in splat_candidates],
    }
    return scene_id, mesh_path, mesh_reason, splat_path, splat_reason, candidates


# -------------------------------------------------------- descriptive stats --

def mesh_stats(mesh_ply: Path):
    """CPU-only, open3d. Returns (stats dict, open3d TriangleMesh)."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    n_v, n_t = len(mesh.vertices), len(mesh.triangles)
    aabb = mesh.get_axis_aligned_bounding_box()
    stats = {
        "path": str(mesh_ply),
        "sha256_head": _sha256_head(mesh_ply),
        "n_vertices": int(n_v),
        "n_triangles": int(n_t),
        "watertight": bool(mesh.is_watertight()) if n_t else False,
        "aabb_min": aabb.get_min_bound().tolist(),
        "aabb_max": aabb.get_max_bound().tolist(),
        "aabb_extent_m": (aabb.get_max_bound() - aabb.get_min_bound()).tolist(),
    }
    return stats, mesh


def splat_stats(splat_ply: Path):
    """CPU-only, plyfile (no torch/gsplat -- keeps the default path GPU-free).
    Inria-format ply: opacity is stored pre-sigmoid."""
    from plyfile import PlyData

    ply = PlyData.read(str(splat_ply))
    v = ply["vertex"]
    names = {p.name for p in v.properties}
    xyz = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    sh_degree = int(round(np.sqrt(1 + n_rest / 3.0) - 1)) if n_rest else 0
    return {
        "path": str(splat_ply),
        "sha256_head": _sha256_head(splat_ply),
        "n_gaussians": int(xyz.shape[0]),
        "sh_degree": sh_degree,
        "opacity_mean": float(opac.mean()),
        "opacity_gt_half_frac": float((opac > 0.5).mean()),
        "aabb_min": xyz.min(axis=0).tolist(),
        "aabb_max": xyz.max(axis=0).tolist(),
    }


def collision_stats(mesh):
    """Single static collision body descriptive stats -- explicitly NOT a
    per-object decomposition (this baseline has no objects to decompose)."""
    verts = np.asarray(mesh.vertices)
    n_t = len(mesh.triangles)
    watertight = bool(mesh.is_watertight()) if n_t else False
    volume_m3 = None
    if watertight:
        try:
            volume_m3 = float(mesh.get_volume())
        except Exception:
            volume_m3 = None
    hull_volume_m3 = None
    if len(verts) >= 4:
        try:
            hull, _ = mesh.compute_convex_hull()
            if hull.is_watertight():
                hull_volume_m3 = float(hull.get_volume())
        except Exception:
            hull_volume_m3 = None
    return {
        "n_static_bodies": 1,
        "n_movable_bodies": 0,
        "decomposition_method": "none -- single fused body, no CoACD/V-HACD",
        "mesh_watertight": watertight,
        "mesh_volume_m3": volume_m3,
        "convex_hull_volume_m3": hull_volume_m3,
    }


def gt_instance_context(scene_dir: Path):
    """Best-effort, informational only: how many GT-annotated objects exist
    in this ScanNet++ scene that a raw fused reconstruction cannot
    individually separate/remove/reposition. Returns None (not an error) if
    GT annotations are not part of this scene dir, e.g. a real phone capture
    with no external ground-truth scanner."""
    seg_json = scene_dir / "scans" / "segments.json"
    anno_json = scene_dir / "scans" / "segments_anno.json"
    if not (seg_json.exists() and anno_json.exists()):
        return None
    try:
        anno = json.loads(anno_json.read_text())
        groups = anno.get("segGroups", [])
        return {
            "source": "ScanNet++ GT instance annotations (scans/segments_anno.json)",
            "n_gt_instances": len(groups),
            "labels": [g.get("label") for g in groups],
            "note": ("informational only -- count of individually-annotated "
                      "objects this baseline, by construction, cannot "
                      "separate/remove/reposition; not a score this method "
                      "achieves"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _render_selfcheck(scene_dir: Path, splat_path: Path, max_frames: int):
    """OPT-IN, GPU-only train-view self-consistency PSNR: renders the fused
    splat at a sample of the SAME posed frames used to train it and compares
    against the captured images. This checks "is the fused representation
    renderable at all" -- it is NOT the main pipeline's official held-out
    DSLR-test-split render protocol (agents/eval/factory_eval_render.py),
    which needs the official train/test split and is out of scope for this
    ablation. Reusing that name for this number would violate the acceptance
    criterion that "metric thresholds and references match across methods",
    so this is deliberately named/reported as a different, narrower metric.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("render self-check needs a CUDA device for the "
                            "gsplat rasterizer; none visible")
    if splat_path is None:
        raise RuntimeError("no trained splat available to render")

    transforms_json = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    images_txt = scene_dir / "dslr" / "colmap" / "images.txt"
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    if not (transforms_json.exists() and images_txt.exists() and img_dir.exists()):
        raise RuntimeError(f"missing posed-DSLR inputs under {scene_dir}/dslr "
                            "(need dslr/nerfstudio/transforms_undistorted.json, "
                            "dslr/colmap/images.txt, "
                            "dslr/resized_undistorted_images/)")

    sys.path.insert(0, str(ROOT))
    from agents.core import common as C  # noqa: E402  (lazy: only needed here)
    from PIL import Image

    K, W, H, _ = C.load_intrinsics(transforms_json)
    w2c_all = sorted(C.load_colmap_w2c(images_txt).items())
    if not w2c_all:
        raise RuntimeError(f"no poses parsed from {images_txt}")
    stride = max(1, len(w2c_all) // max_frames)
    sample = w2c_all[::stride][:max_frames]

    gs = C.load_gaussians(splat_path, device="cuda")
    psnrs = []
    for fname, w2c in sample:
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        gt = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
        rgb, _, _ = C.render_view(gs, w2c, K, W, H, render_mode="RGB")
        if rgb.shape[:2] != gt.shape[:2]:
            gt_img = Image.fromarray((gt * 255).astype(np.uint8))
            gt = np.asarray(gt_img.resize((rgb.shape[1], rgb.shape[0])),
                             dtype=np.float64) / 255.0
        psnrs.append(C.psnr(rgb, gt))
    if not psnrs:
        raise RuntimeError("no frames scored (no matching images found under "
                            f"{img_dir})")
    return {"value": float(np.mean(psnrs)), "status": "ok",
            "n_frames": len(psnrs),
            "protocol": ("train-view self-consistency (frames used to train "
                         "the splat; NOT the held-out DSLR test split used "
                         "by the main pipeline's row-G render numbers)")}


# --------------------------------------------------------------- top level --

def build_report(scene_dir, mesh_ply=None, splat_ply=None, splats_root=None,
                  render_eval=False, render_eval_max_frames=8, out_dir=None):
    """Run the raw_reconstruction ablation against one real scene dir and
    return the report dict (also written to <out_dir>/raw_reconstruction_report.json
    if out_dir is given). Every step is best-effort and logged: nothing that
    fails is silently converted into a zero or omitted."""
    t_start = time.time()
    timings = {}
    failures = []
    scene_dir = Path(scene_dir)

    scene_id, mesh_path, mesh_reason, splat_path, splat_reason, candidates = \
        resolve_inputs(scene_dir, mesh_ply, splat_ply, splats_root)

    report = {
        "schema_version": "1.0",
        "method": "raw_reconstruction",
        "scene_id": scene_id,
        "scene_dir": str(scene_dir.resolve()) if scene_dir.exists() else str(scene_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "candidates_checked": candidates,
        "construction": {
            "description": (
                "Single fused static representation (background + every "
                "object baked together) taken directly from the upstream "
                "capture/reconstruction pipeline (COLMAP poses + trained "
                "3DGS splat [+ dataset or splat-derived mesh]) with NO "
                "per-object stage applied: no instance discovery, no "
                "mask-to-gaussian removal, no per-object asset generation, "
                "no Sim(3) registration, no CoACD collision decomposition, "
                "no physics annotation, no articulation."
            ),
            "stages_run": [],
            "n_objects_factored": 0,
            "capabilities": {
                "background_object_separation": False,
                "per_object_removal": False,
                "per_object_reposition": False,
                "per_object_collision": False,
                "per_object_physics": False,
                "articulation": False,
            },
        },
        "human_intervention": {
            "seconds": 0,
            "notes": ("fully automatic: this script only loads and measures "
                      "the capture/training pipeline's own existing outputs; "
                      "contrast with PolaRiS-manual-construction, where a "
                      "human places assets by hand (see release_status.yaml)"),
        },
    }

    inputs = {}
    mesh_o3d = None
    t0 = time.time()
    if mesh_path is not None:
        try:
            stats, mesh_o3d = mesh_stats(mesh_path)
            inputs["mesh"] = stats
        except Exception as e:
            inputs["mesh"] = {"path": str(mesh_path), "status": "error",
                               "error": str(e)}
            failures.append({"stage": "mesh_load", "error": str(e)})
    else:
        inputs["mesh"] = {"status": "not_applicable", "reason": mesh_reason}
    timings["mesh_load_s"] = round(time.time() - t0, 4)

    t0 = time.time()
    if splat_path is not None:
        try:
            inputs["splat"] = splat_stats(splat_path)
        except Exception as e:
            inputs["splat"] = {"path": str(splat_path), "status": "error",
                                "error": str(e)}
            failures.append({"stage": "splat_load", "error": str(e)})
    else:
        inputs["splat"] = {"status": "not_applicable", "reason": splat_reason}
    timings["splat_load_s"] = round(time.time() - t0, 4)
    report["inputs"] = inputs

    t0 = time.time()
    if mesh_o3d is not None:
        try:
            report["collision"] = collision_stats(mesh_o3d)
        except Exception as e:
            report["collision"] = {"status": "error", "error": str(e)}
            failures.append({"stage": "collision_stats", "error": str(e)})
    else:
        report["collision"] = _na("no mesh available for even a single "
                                   "static collision body")
    timings["collision_stats_s"] = round(time.time() - t0, 4)

    t0 = time.time()
    ctx = gt_instance_context(scene_dir)
    if ctx is not None:
        report["context_gt_instances"] = ctx
    timings["gt_context_s"] = round(time.time() - t0, 4)

    metrics = {name: _na(reason) for name, reason in NA_METRICS.items()}
    t0 = time.time()
    if render_eval:
        try:
            metrics["render_selfconsistency_psnr_db"] = _render_selfcheck(
                scene_dir, splat_path, render_eval_max_frames)
        except Exception as e:
            metrics["render_selfconsistency_psnr_db"] = _err(e)
            failures.append({"stage": "render_selfcheck", "error": str(e)})
    else:
        metrics["render_selfconsistency_psnr_db"] = _na(
            "pass --render-eval to compute (needs a CUDA device + gsplat); "
            "this is a train-view fidelity sanity check, NOT the official "
            "held-out DSLR-test-split protocol used by the main pipeline's "
            "row-G render numbers")
    timings["render_eval_s"] = round(time.time() - t0, 4)
    report["metrics"] = metrics

    timings["total_s"] = round(time.time() - t_start, 4)
    report["timings_s"] = timings
    report["failures"] = failures
    report["build_success"] = bool(
        (mesh_path is not None or splat_path is not None)
        and not any(f["stage"] in ("mesh_load", "splat_load") for f in failures)
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "raw_reconstruction_report.json"
        out_path.write_text(json.dumps(report, indent=2))
        report["_report_path"] = str(out_path)
    return report


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(
        description="raw_reconstruction: no-object-factorization ablation "
                     "baseline (plan/15_CONSTRUCTION_BASELINES.md)")
    ap.add_argument("--scene-dir", required=True,
                     help="e.g. data/recon_scenes/data/<scene> or "
                          "/data/ScanNetpp/data/<scene>")
    ap.add_argument("--mesh-ply", default=None)
    ap.add_argument("--splat-ply", default=None)
    ap.add_argument("--splats-root", default=None)
    ap.add_argument("--out-dir", default=None,
                     help="default: outputs/<scene_id>_baselines/"
                          "raw_reconstruction under the repo root")
    ap.add_argument("--render-eval", action="store_true",
                     help="opt-in GPU train-view PSNR self-check (see "
                          "_render_selfcheck docstring for scope)")
    ap.add_argument("--render-eval-max-frames", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        scene_id = Path(args.scene_dir).name
        out_dir = ROOT / "outputs" / f"{scene_id}_baselines" / "raw_reconstruction"

    report = build_report(
        args.scene_dir, args.mesh_ply, args.splat_ply, args.splats_root,
        args.render_eval, args.render_eval_max_frames, out_dir=out_dir)

    summary = {k: v for k, v in report.items()
               if k not in ("candidates_checked",)}
    print(json.dumps(summary, indent=2))
    print(f"[raw-recon] build_success={report['build_success']} "
          f"failures={len(report['failures'])} -> {report.get('_report_path')}")


if __name__ == "__main__":
    main()
