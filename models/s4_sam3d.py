"""V_mesh slot alternative: SAM 3D Objects (Meta, single-image + mask -> 3D).

Same per-object contract as s4_trellis but into an obj_XX/sam3d/ subdir (the
candidate layout factory_hybrid consumes, mirroring obj_XX/rvg/):
  sam3d_mesh.ply   vertex-colored FlexiCubes mesh (slat_decoder_mesh, full res)
  sam3d_gs.ply     gaussians, raw canonical coords -- SAME frame as the mesh
  sam3d_meta.json  predicted cam pose + runtime/VRAM (informational only)

ENV: this module does NOT run under the pipeline .venv. It needs the
dedicated SAM 3D Objects env (torch 2.5.1+cu121, pytorch3d, kaolin,
flash-attn, built per third_party/sam-3d-objects/doc/setup.md):
  ${SIMANY_ROOT}/.envs/sam3d-objects/bin/python
Launch from the repo root via run/env.sh's `run_sam3d models.s4_sam3d`
(SIMANY_SCENE/SIMANY_OUT selected exactly like s4_trellis). GPU required
(~17.5 GB peak on the factory crops -> fits the A6000 with headroom).

Frame: SAM 3D Objects decodes into a TRELLIS-style z-up ~[-0.5,0.5]^3
canonical cube -- verified empirically on smoke crops (keyboard thin axis,
bottle long axis and squat-mug height all land on canonical z; see
outputs/sam3d_smoke/). Both artifacts are therefore saved UNROTATED, exactly
like s4_trellis, and s5_align's yaw-only upright prior applies unchanged.
The predicted canonical->camera pose (rotation/translation/scale) is only
recorded in sam3d_meta.json; registration to the scene stays the job of the
factory_align/s5_align machinery so all V_mesh candidates are scored under
the identical transform family.
Usage: s4_sam3d.py [--objects 0,2,3] (default: all objects.json entries)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# notebook/inference.py hard-reads CONDA_PREFIX (sets CUDA_HOME from it);
# running the env's python directly (no `conda activate`) leaves it unset.
os.environ.setdefault("CONDA_PREFIX", str(Path(sys.executable).parents[1]))

import numpy as np
from PIL import Image

from agents.core import common as C

SAM3D_DIR = C.ROOT / "third_party" / "sam-3d-objects"
sys.path.insert(0, str(SAM3D_DIR / "notebook"))

# pipeline.yaml resolves the ckpt paths relative to itself (workspace_dir)
CONFIG = os.environ.get("SIMANY_SAM3D_CONFIG",
                        str(SAM3D_DIR / "checkpoints" / "hf" / "pipeline.yaml"))
SEED = 42  # same generation seed as s4_trellis


def load_component_pipeline(source_dir, config, moge_checkpoint=None):
    """Reuse the factory's official Objects API with explicit offline paths."""
    source_dir = Path(source_dir)
    sys.path[:0] = [str(source_dir / "notebook"), str(source_dir)]
    from inference import Inference
    if moge_checkpoint is None:
        return Inference(str(config), compile=False)
    # Preserve the official model and configuration; resolve only this named
    # auxiliary dependency to the already existing, hashed checkpoint file.
    import inspect
    from moge.model.v1 import MoGeModel
    descriptor = inspect.getattr_static(MoGeModel, "from_pretrained")
    original = MoGeModel.from_pretrained
    def local_moge(pretrained_model_name_or_path, *args, **kwargs):
        name = pretrained_model_name_or_path
        if name != "Ruicheng/moge-vitl":
            raise ValueError("undeclared MoGe resource: " + str(name))
        return original(str(moge_checkpoint), *args, **kwargs)
    MoGeModel.from_pretrained = staticmethod(local_moge)
    try:
        return Inference(str(config), compile=False)
    finally:
        MoGeModel.from_pretrained = descriptor


def component_proposal(pipe, rgba_path, out_dir, seed):
    """One proposal in the unchanged factory canonical frame; no registration."""
    import torch
    import trimesh
    out_dir = Path(out_dir)
    with Image.open(rgba_path) as image:
        if image.mode != "RGBA":
            raise ValueError("an authentic premasked RGBA is required")
        rgba = np.asarray(image)
    if not (rgba[..., 3] > 0).any() or not (rgba[..., 3] == 0).any():
        raise ValueError("foreground and background alpha required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generated = pipe(rgba[..., :3], rgba[..., 3] > 0, seed=seed)
    inference_seconds = time.monotonic() - started
    native = generated["mesh"][0]
    vertices = native.vertices.detach().cpu().numpy()
    faces = native.faces.detach().cpu().numpy()
    attrs = native.vertex_attrs.detach().cpu().numpy()
    if (not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
            or not np.isfinite(attrs).all() or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("invalid SAM 3D Objects mesh")
    np.savez_compressed(out_dir / "native_candidate.npz", vertices=vertices, faces=faces, vertex_attrs=attrs)
    mesh = trimesh.Trimesh(vertices, faces, process=False,
                          vertex_colors=(np.clip(attrs[:, :3], 0, 1) * 255).astype(np.uint8))
    mesh.export(out_dir / "sam3d_mesh.ply")
    generated["gs"].save_ply(str(out_dir / "sam3d_gs.ply"))
    pose = {key: np.asarray(generated[key].detach().cpu()).reshape(-1).tolist()
            for key in ("rotation", "translation", "scale")}
    return {"seed": seed, "frame": "canonical z-up (raw)",
            "predicted_pose_informational_only": pose,
            "inference_seconds": inference_seconds,
            "generation_export_seconds": time.monotonic() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "vertices": len(vertices), "faces": len(faces),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def main():
    import torch
    import trimesh
    from inference import Inference  # noqa: sam3d notebook API

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=None,
                    help="comma-separated indices (default: all)")
    args = ap.parse_args()
    picks = None if args.objects is None \
        else {int(x) for x in args.objects.split(",")}

    inference = None  # lazy: skip the ~13GB ckpt load when everything cached
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    for meta in objects:
        idx = meta["index"]
        if picks is not None and idx not in picks:
            continue
        odir = C.OUT / "objects" / f"obj_{idx:02d}"
        sdir = odir / "sam3d"
        gs_ply = sdir / "sam3d_gs.ply"
        mesh_ply = sdir / "sam3d_mesh.ply"
        rgba_png = odir / "rgba.png"
        # cache valid only if newer than its input (same rule as s4_trellis)
        if gs_ply.exists() and mesh_ply.exists() \
                and gs_ply.stat().st_mtime > rgba_png.stat().st_mtime:
            print(f"[s4d] {odir.name}: cached, skip")
            continue

        if inference is None:
            inference = Inference(CONFIG, compile=False)
        rgba = np.array(Image.open(rgba_png).convert("RGBA"))
        image, mask = rgba[..., :3], rgba[..., 3] > 0

        sdir.mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = inference(image, mask, seed=SEED)
        dt = time.time() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 2 ** 30

        mesh = out["mesh"][0]
        v = mesh.vertices.detach().cpu().numpy().astype(np.float64)
        f = mesh.faces.cpu().numpy()
        rgb = mesh.vertex_attrs[:, :3].clamp(0, 1).detach().cpu().numpy()
        trimesh.Trimesh(v, f, vertex_colors=(rgb * 255).astype(np.uint8),
                        process=False).export(mesh_ply)

        # sam3d Gaussian.save_ply(path) has no transform kwarg and writes the
        # raw canonical coords -- already the same frame as the mesh above
        out["gs"].save_ply(str(gs_ply))

        C.save_json(sdir / "sam3d_meta.json", {
            "seed": SEED, "config": CONFIG, "frame": "canonical z-up (raw)",
            "runtime_s": dt, "peak_vram_gb": vram_gb,
            "cam_rotation_wxyz": np.asarray(
                out["rotation"].detach().cpu()).reshape(-1),
            "cam_translation": np.asarray(
                out["translation"].detach().cpu()).reshape(-1),
            "cam_scale": np.asarray(out["scale"].detach().cpu()).reshape(-1)})
        print(f"[s4d] {odir.name} {meta['label']}: mesh {len(v)}v/{len(f)}f, "
              f"gs {out['gs'].get_xyz.shape[0]}, {dt:.0f}s, "
              f"peak {vram_gb:.1f}GB")
        del out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
