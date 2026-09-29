"""Stage 4: TRELLIS image-to-3D per object (paper's V_mesh slot).

Input: obj_XX/rgba.png. Output (canonical z-up ~[-0.5,0.5]^3 frame for BOTH):
  trellis_mesh.ply     vertex-colored FlexiCubes mesh (full res)
  mesh_sim.ply/.obj    <=40k-triangle decimation (collision/sim input)
  trellis_gs.ply       gaussians, saved with transform=None (same frame!)
"""
import json
import os
import sys
import types
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
sys.modules.setdefault("rembg", types.ModuleType("rembg"))

import numpy as np
from PIL import Image

from agents.core import common as C

sys.path.insert(0, str(C.TRELLIS_DIR))

SEED = 42
SIM_TRIS = 40_000
MODEL_SOURCE = os.environ.get(
    "SIMANY_TRELLIS_MODEL", "microsoft/TRELLIS-image-large"
)


def _configure_xformers_compat():
    # Apply the compatibility patch only when a backend is actually invoked.
    # A missing xformers installation is diagnosed by the real TRELLIS backend;
    # it does not prevent importing this bridge for CPU contract checks.
    try:
        import xformers.ops.fmha as fmha
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    except ModuleNotFoundError as exc:
        if exc.name == "xformers":
            return
        raise
    if not hasattr(fmha, "BlockDiagonalMask"):
        fmha.BlockDiagonalMask = BlockDiagonalMask


def main():
    _configure_xformers_compat()
    import torch
    import trimesh
    from trellis.pipelines import TrellisImageTo3DPipeline

    # Paper launchers pass immutable, repository-local TRELLIS and DINOv2
    # sources. TRELLIS initializes DINOv2 through torch.hub internally, so
    # redirect that one known request to the pinned local source tree. The
    # model-ID fallbacks remain available for legacy interactive use only.
    dinov2_source = os.environ.get("SIMANY_DINOV2_REPO")
    original_hub_load = torch.hub.load

    def pinned_hub_load(repo_or_dir, model, *args, **kwargs):
        if dinov2_source and repo_or_dir == "facebookresearch/dinov2":
            repo_or_dir = dinov2_source
            kwargs["source"] = "local"
        return original_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = pinned_hub_load
    try:
        pipe = TrellisImageTo3DPipeline.from_pretrained(MODEL_SOURCE)
    finally:
        torch.hub.load = original_hub_load
    pipe.cuda()

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    for meta in objects:
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        gs_ply = odir / "trellis_gs.ply"
        strict_records = os.environ.get("SIMANY_GENERATION_RECORDS")
        if strict_records and (
                (Path(strict_records) / f"object_{meta['index']:02d}.json").exists()
                or any((odir / name).exists() for name in
                       ("trellis_gs.ply", "trellis_mesh.ply", "mesh_sim.ply", "mesh_sim.obj"))):
            raise FileExistsError(f"strict generation refuses existing proposal: {odir}")
        # cache valid only if newer than its input (indices can remap across runs)
        if gs_ply.exists() and gs_ply.stat().st_mtime > (odir / "rgba.png").stat().st_mtime:
            print(f"[s4] {odir.name}: cached, skip")
            continue
        started = time.monotonic()
        out = None
        try:
            img = Image.open(odir / "rgba.png").convert("RGBA")
            out = pipe.run(img, seed=SEED, formats=["mesh", "gaussian"])

            mesh = out["mesh"][0]
            v = mesh.vertices.cpu().numpy().astype(np.float64)
            f = mesh.faces.cpu().numpy()
            rgb = mesh.vertex_attrs[:, :3].clamp(0, 1).cpu().numpy()
            tm = trimesh.Trimesh(v, f, vertex_colors=(rgb * 255).astype(np.uint8),
                                 process=False)
            tm.export(odir / "trellis_mesh.ply")

            import open3d as o3d
            om = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
            om.vertex_colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            if len(f) > SIM_TRIS:
                om = om.simplify_quadric_decimation(SIM_TRIS)
            om.remove_degenerate_triangles()
            o3d.io.write_triangle_mesh(str(odir / "mesh_sim.ply"), om)
            trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles),
                            process=False).export(odir / "mesh_sim.obj")

            out["gaussian"][0].save_ply(str(odir / "trellis_gs.ply"), transform=None)
            print(f"[s4] {odir.name} {meta['label']}: mesh {len(v)}v/{len(f)}f, "
                  f"sim {len(om.triangles)}f, gs {out['gaussian'][0].get_xyz.shape[0]}")
        except Exception as exc:
            if not strict_records:
                raise
            record = {"object_index": meta["index"], "seed": SEED,
                      "wall_s": time.monotonic() - started,
                      "model_source": MODEL_SOURCE, "status": "generation_failed",
                      "error_type": type(exc).__name__,
                      "reason": f"{type(exc).__name__}: {exc}"}
            record_path = Path(strict_records) / f"object_{meta['index']:02d}.json"
            with record_path.open("x") as stream:
                json.dump(record, stream, indent=2)
            print(f"[s4] {odir.name}: generation_failed: {record['reason']}")
            del out
            torch.cuda.empty_cache()
            continue
        if strict_records:
            record = {"object_index": meta["index"], "seed": SEED,
                      "wall_s": time.monotonic() - started,
                      "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "model_source": MODEL_SOURCE, "status": "generated"}
            record_path = Path(strict_records) / f"object_{meta['index']:02d}.json"
            with record_path.open("x") as stream:
                json.dump(record, stream, indent=2)
        del out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
