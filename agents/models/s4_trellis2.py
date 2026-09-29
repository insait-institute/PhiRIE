"""Offline TRELLIS.2 mesh/PBR producer. Gaussian output is not supported.

GPU dependencies are imported only by ``load_pipeline``. The official model
and samplers are used unchanged; premasked RGBA avoids an unused rembg model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUTPUTS = ("trellis2_mesh.ply", "mesh_sim.ply", "mesh_sim.obj", "trellis2_pbr.npz",
           "trellis2_conditioning.png")
PIPELINES = ("512", "1024", "1024_cascade", "1536_cascade")
PREPROCESSING = "official_rgba_alpha_bbox_crop_black_composite_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    """Publish exclusively; retain staging files if interrupted."""
    path = Path(path)
    stage = path.with_name(path.name + ".partial")
    with stage.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(stage, path)
    stage.unlink()


def validate_sources(source_dir, model_dir, dinov3_model, ss_decoder, source_commit):
    source = Path(source_dir).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True)
    if not source_commit or head != source_commit or dirty:
        raise ValueError("TRELLIS.2 source must be clean and match the pinned commit")
    if not (source / "trellis2/pipelines/trellis2_image_to_3d.py").is_file():
        raise ValueError("pinned source is not TRELLIS.2")
    model = Path(model_dir).resolve(strict=True)
    dino = Path(dinov3_model).resolve(strict=True)
    if not (dino / "config.json").is_file() or not list(dino.glob("*.safetensors")):
        raise FileNotFoundError("a complete local DINOv3 snapshot is required")
    config = json.loads((model / "pipeline.json").read_text())
    if config.get("name") != "Trellis2ImageTo3DPipeline":
        raise ValueError("unexpected pipeline class")
    args = config["args"]
    if args["image_cond_model"]["name"] != "DinoV3FeatureExtractor":
        raise ValueError("expected the official DINOv3 conditioning model")
    paths = {}
    for name, relative in args["models"].items():
        prefix = Path(ss_decoder) if name == "sparse_structure_decoder" else model / relative
        prefix = prefix.resolve()
        for suffix in (".json", ".safetensors"):
            if not Path(str(prefix) + suffix).is_file():
                raise FileNotFoundError(f"missing local model component: {prefix}{suffix}")
        paths[name] = str(prefix)
    return args, paths, {"source_dir": str(source), "source_commit": head,
                         "model_source": str(model), "pipeline_config_sha256": _sha(model / "pipeline.json"),
                         "dinov3_model": str(dino), "checkpoint_prefixes": paths,
                         "rembg": "disabled_explicit_premasked_rgba_only"}


def load_pipeline(args, paths, provenance):
    """Construct official components exclusively from existing local paths."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = provenance["source_dir"]
    existing = sys.modules.get("trellis2")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError("a different TRELLIS.2 package is already imported")
    sys.path.insert(0, source)
    import torch
    from trellis2 import models
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import samplers
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    loaded = {name: models.from_pretrained(prefix) for name, prefix in paths.items()}
    kwargs = {"models": loaded, "image_cond_model": DinoV3FeatureExtractor(provenance["dinov3_model"]),
              "rembg_model": None, "low_vram": args.get("low_vram", True),
              "default_pipeline_type": args.get("default_pipeline_type", "1024_cascade")}
    for stage in ("sparse_structure", "shape_slat", "tex_slat"):
        spec = args[stage + "_sampler"]
        kwargs[stage + "_sampler"] = getattr(samplers, spec["name"])(**spec["args"])
        kwargs[stage + "_sampler_params"] = spec["params"]
    for stage in ("shape_slat", "tex_slat"):
        kwargs[stage + "_normalization"] = args[stage + "_normalization"]
    pipe = Trellis2ImageTo3DPipeline(**kwargs)
    pipe.cuda()
    return pipe, torch


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no native bfloat16; conversion to float32 preserves every
        # represented bfloat16 value and does not clamp or repair invalid data.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def validate_mesh_pbr(mesh):
    import numpy as np
    arrays = {name: _array(getattr(mesh, name)) for name in ("vertices", "faces", "attrs", "coords", "origin")}
    for name, value in arrays.items():
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"empty or nonfinite TRELLIS.2 {name}")
    v, f, attrs, coords = (arrays[x] for x in ("vertices", "faces", "attrs", "coords"))
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("invalid triangle mesh shape")
    if not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v):
        raise ValueError("invalid triangle indices")
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    if not np.any(area > 0):
        raise ValueError("mesh has no positive-area triangles")
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("invalid PBR voxel coordinates")
    if attrs.ndim != 2 or attrs.shape != (len(coords), 6):
        raise ValueError("invalid PBR attribute shape")
    expected = {"base_color": (0, 3), "metallic": (3, 4), "roughness": (4, 5), "alpha": (5, 6)}
    layout = {key: [value.start, value.stop] for key, value in mesh.layout.items()}
    if layout != {key: list(value) for key, value in expected.items()}:
        raise ValueError("unexpected PBR attribute layout")
    voxel_size = float(mesh.voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0 or arrays["origin"].shape != (3,):
        raise ValueError("invalid PBR spatial transform")
    arrays.update(voxel_size=np.asarray(voxel_size), voxel_shape=np.asarray(mesh.voxel_shape),
                  layout_json=np.asarray(json.dumps(layout, sort_keys=True)))
    return arrays


def _export(mesh, stage, sim_tris):
    import numpy as np
    import trimesh
    arrays = validate_mesh_pbr(mesh)
    tm = trimesh.Trimesh(arrays["vertices"], arrays["faces"], process=False)
    tm.export(stage / "trellis2_mesh.ply")
    sim = tm
    if len(tm.faces) > sim_tris:
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),
                                      o3d.utility.Vector3iVector(tm.faces))
        om = om.simplify_quadric_decimation(sim_tris)
        om.remove_degenerate_triangles()
        sim = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
    if not len(sim.faces) or not np.isfinite(sim.vertices).all() or sim.area <= 0:
        raise ValueError("invalid simulation mesh after decimation")
    sim.export(stage / "mesh_sim.ply")
    sim.export(stage / "mesh_sim.obj")
    np.savez_compressed(stage / "trellis2_pbr.npz", **arrays)
    return {"vertices": len(tm.vertices), "faces": len(tm.faces), "sim_faces": len(sim.faces),
            "pbr_voxels": len(arrays["coords"])}


def generate_objects(out_root, records_dir, *, source_dir, model_dir, dinov3_model,
                     ss_decoder, source_commit, seed=42, pipeline_type="512", sim_tris=40000):
    """Generate one proposal per declared prepared object, retaining failures.

    Process/model initialization failures propagate. Object failures are terminal
    records and do not discard subsequent objects. Existing outputs/claims fail
    closed; a rerun needs a new experiment directory.
    """
    import numpy as np
    from PIL import Image
    if pipeline_type not in PIPELINES or sim_tris < 1:
        raise ValueError("invalid pipeline type or simulation triangle budget")
    out_root, records_dir = Path(out_root), Path(records_dir)
    objects = json.loads((out_root / "objects/objects.json").read_text())
    indices = [meta["index"] for meta in objects]
    if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("object indices must be unique nonnegative integers")
    records_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        odir = out_root / "objects" / f"obj_{index:02d}"
        reserved = [records_dir / f"object_{index:02d}.json", records_dir / f"object_{index:02d}.claim"]
        reserved += [odir / name for name in (*OUTPUTS, ".trellis2_stage")]
        if any(path.exists() for path in reserved):
            raise FileExistsError(f"TRELLIS.2 refuses existing proposal or claim: {odir}")
    args, paths, provenance = validate_sources(source_dir, model_dir, dinov3_model, ss_decoder, source_commit)
    # Loading once per scene is deliberate; all objects use this exact instance.
    pipe, torch = load_pipeline(args, paths, provenance)
    records = []
    for meta in objects:
        index = meta["index"]
        odir = out_root / "objects" / f"obj_{index:02d}"
        claim = records_dir / f"object_{index:02d}.claim"
        with claim.open("x") as stream:
            json.dump({"object_index": index, "pid": os.getpid()}, stream)
        started = time.monotonic()
        record = {"object_index": index, "seed": seed, "pipeline_type": pipeline_type,
                  "source_commit": source_commit,
                  "generator": "trellis2", "provenance": provenance, "model_source": str(model_dir),
                  "preprocessing": PREPROCESSING, "capabilities": {"mesh": True, "pbr_voxels": True,
                  "gaussian": False, "baked_glb": False}, "sim_triangle_budget": sim_tris}
        generated = None
        try:
            torch.cuda.reset_peak_memory_stats()
            stage = odir / ".trellis2_stage"
            stage.mkdir()
            image_path = odir / "rgba.png"
            record["input_sha256"] = _sha(image_path)
            with Image.open(image_path) as source:
                if source.mode != "RGBA":
                    raise ValueError("TRELLIS.2 requires the authentic premasked RGBA input")
                image = source.copy()
            alpha = np.asarray(image)[:, :, 3]
            if np.all(alpha == 255) or not np.any(alpha > 204):
                raise ValueError("RGBA must contain background alpha and supported foreground")
            conditioned = pipe.preprocess_image(image)
            if conditioned.width < 1 or conditioned.height < 1 or conditioned.mode != "RGB":
                raise ValueError("official preprocessing returned invalid RGB conditioning")
            conditioned.save(stage / "trellis2_conditioning.png")
            record["conditioning_sha256"] = _sha(stage / "trellis2_conditioning.png")
            record["conditioning_size"] = list(conditioned.size)
            generated = pipe.run(conditioned, seed=seed, pipeline_type=pipeline_type, preprocess_image=False)
            if len(generated) != 1:
                raise ValueError("TRELLIS.2 must return exactly one proposal")
            record["geometry"] = _export(generated[0], stage, sim_tris)
            record["artifacts"] = {}
            for name in OUTPUTS:
                src, dest = stage / name, odir / name
                if not src.is_file() or src.stat().st_size == 0:
                    raise ValueError(f"empty TRELLIS.2 output: {name}")
                os.link(src, dest)
                record["artifacts"][name] = {"sha256": _sha(dest), "bytes": dest.stat().st_size}
            record["status"] = "generated"
        except Exception as exc:
            record.update(status="generation_failed", error_type=type(exc).__name__,
                          reason=f"{type(exc).__name__}: {exc}")
        finally:
            del generated
            record["wall_s"] = time.monotonic() - started
            record["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            _atomic_json(records_dir / f"object_{index:02d}.json", record)
            torch.cuda.empty_cache()
        records.append(record)
        print(f"[s4_trellis2] obj_{index:02d}: {record['status']}", flush=True)
    return records


def main():
    generate_objects(os.environ["SIMANY_OUT"], os.environ["SIMANY_GENERATION_RECORDS"],
                     source_dir=os.environ["SIMANY_TRELLIS2_DIR"], model_dir=os.environ["SIMANY_TRELLIS2_MODEL"],
                     dinov3_model=os.environ["SIMANY_DINOV3_MODEL"], ss_decoder=os.environ["SIMANY_SS_DECODER"],
                     source_commit=os.environ["SIMANY_TRELLIS2_SOURCE_COMMIT"],
                     seed=int(os.environ.get("SIMANY_TRELLIS2_SEED", "42")),
                     pipeline_type=os.environ.get("SIMANY_TRELLIS2_PIPELINE_TYPE", "512"))


if __name__ == "__main__":
    main()
