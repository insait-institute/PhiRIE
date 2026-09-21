"""Factory stage 1: enumerate GT instances and cut best-view RGBA crops.

Dataset-scale mode: instead of SAM3 + single-frame depth (zero-shot demo),
use ScanNet++ GT instance annotations and the full DSLR trajectory:
  - whitelist + size-gated instances from segments_anno.json
  - per instance, score every frame by visibility (mesh raycast occlusion),
    projected pixel area, and crop sharpness; take the best
  - occlusion-aware mask by comparing instance-submesh vs full-mesh raycast
Writes the SAME layout the zero-shot stages use (objects/objects.json +
obj_XX/{rgba.png, gt_points.ply, meta.json}) so s4/s6 run unchanged
under SIMANY_OUT.
"""
import argparse
import functools
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from agents.core import common as C

WHITELIST = {
    "bottle", "plastic bottle", "glass bottle", "water bottle", "cup", "mug",
    "paper cup", "box", "cardboard box", "carboard box", "storage box",
    "tissue box", "book", "notebook", "keyboard", "mouse", "computer mouse",
    "telephone", "headphones", "headphone", "shoe", "shoes", "bag", "backpack",
    "toy", "bowl", "plate", "can", "jar", "remote", "remote control", "pen",
    "pencil", "marker", "whiteboard marker", "stapler", "scissors",
    "plant pot", "flower pot", "vase", "kettle", "laptop", "tablet",
    "cellphone", "phone", "basket", "tray", "folder", "charger",
    "laptop charger", "wireless charger", "pen holder", "speaker", "clock",
}
MIN_DIM, MAX_DIM = 0.03, 0.8
N_VIS_SAMPLES = 120
MIN_VIS_FRAC = 0.5
# Pixel gates (MIN_BBOX_PX, MIN_MASK_PX) were tuned on 1752x1168 ScanNet++
# DSLR frames. Emulated scenes at other resolutions must override them via
# env (bbox scales linearly with frame height, mask area quadratically) -
# run/run_behavior_recon.sh does this for its 320x180 BEHAVIOR extracts.
MIN_BBOX_PX = int(C.env("MIN_BBOX_PX", "48"))
# Raycast/known-point agreement tolerance for the visibility check below.
# 2cm assumes near-continuous scan-mesh fidelity; a TSDF-fused mesh at
# VOXEL-scale voxels (derive_mesh_from_splat.py) has vertex positions
# quantized to roughly that scale, so points genuinely on the surface can
# miss a 2cm window even when correctly visible - widen it to match.
VIS_TOL_M = float(C.env("VIS_TOL_M", "0.02"))
TOP_K_SHARP = 8
MIN_MASK_PX = int(C.env("MIN_MASK_PX", "400"))
GT_SURFACE_SAMPLE_SEED = 42


@functools.lru_cache(maxsize=16)
def load_image(name):
    image = np.asarray(Image.open(C.IMAGES_DIR / name).convert("RGB"))
    _READ_FRAMES.add(name)
    return image


_READ_FRAMES = set()


def _write_read_frames(path, frames):
    """Atomically publish the exact successful ``load_image`` frame union."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite read-frame record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError(f"read-frame record parent is a symlink: {destination.parent}")
    temporary = destination.with_name(
        f".{destination.name}.staging-{uuid.uuid4().hex}"
    )
    data = json.dumps(
        {"frames": sorted(frames)}, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _plain_frame_names(values, *, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty JSON list")
    if any(
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        for value in values
    ):
        raise ValueError(f"{label} must contain plain non-empty frame names")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicate frame names")
    return set(values)


def load_frame_allowlist(path):
    """Read a strict JSON frame allowlist for leakage-free preparation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("frames")
    return _plain_frame_names(payload, label="frame allowlist")


def load_output_index_map(path):
    """Read ``{source object_id: preserved output index}`` from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("output index map must be a non-empty JSON object")
    result = {}
    for raw_object_id, raw_index in payload.items():
        try:
            object_id = int(raw_object_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid source object ID: {raw_object_id!r}") from exc
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise ValueError(f"invalid preserved output index for {object_id}")
        if object_id in result:
            raise ValueError(f"duplicate source object ID: {object_id}")
        result[object_id] = raw_index
    if len(result.values()) != len(set(result.values())):
        raise ValueError("preserved output indices must be unique")
    return result


def prepare_objects(
    *,
    frame_allowlist=None,
    selected_object_ids=None,
    output_index_by_object_id=None,
    read_frames_out=None,
):
    """Run the production preparation logic with optional strict selection.

    ``frame_allowlist`` limits *every* visibility/sharpness candidate before
    scoring. ``selected_object_ids`` selects source instances by their stable
    ``object_id``. ``output_index_by_object_id`` allows a remediation bundle
    to preserve sparse indices from an immutable legacy build.
    """
    load_image.cache_clear()
    _READ_FRAMES.clear()

    import open3d as o3d
    import trimesh
    from agents.assets.s5_align import scene_mesh_arrays

    K, W, H, _ = C.load_intrinsics()
    w2c_all = sorted(C.load_colmap_w2c().items())
    if frame_allowlist is not None:
        frame_allowlist = set(frame_allowlist)
        if not frame_allowlist:
            raise ValueError("frame_allowlist must not be empty")
        w2c_all = [item for item in w2c_all if item[0] in frame_allowlist]
        if not w2c_all:
            raise RuntimeError("no registered camera is present in frame_allowlist")
    verts, faces = scene_mesh_arrays()
    scene_full = C.make_raycast_scene()

    if C.env("AUTO") == "1":
        # automated mode: instances come from auto_segment (already
        # vocabulary-limited by its SAM3 prompts)
        gts = C.load_auto_instances()
    else:
        vocab = C.VOCAB if C.env("FULL") == "1" else WHITELIST
        gts = [g for g in C.load_gt_instances()
               if g["label"].strip().lower() in vocab
               and g["label"].strip().lower() not in C.STRUCTURAL_EXCLUDE]
    selected = set(selected_object_ids) if selected_object_ids is not None else None
    if selected is not None:
        if not selected or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in selected
        ):
            raise ValueError("selected_object_ids must contain non-negative integers")
        available = {int(g["object_id"]) for g in gts}
        missing = sorted(selected - available)
        if missing:
            raise RuntimeError(
                f"selected object IDs are absent after source-instance gates: {missing}"
            )
        gts = [g for g in gts if int(g["object_id"]) in selected]
    index_map = dict(output_index_by_object_id or {})
    if index_map:
        if selected is None or set(index_map) != selected:
            raise ValueError(
                "output_index_by_object_id keys must exactly match selected_object_ids"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in index_map.values()
        ) or len(index_map.values()) != len(set(index_map.values())):
            raise ValueError("output_index_by_object_id values must be unique non-negative integers")
    kept, objects = 0, []
    produced_source_ids = set()
    rng = np.random.RandomState(0)

    for g in gts:
        ext = g["aabb"][1] - g["aabb"][0]
        max_dim = C.VOCAB.get(g["label"].strip().lower(), MAX_DIM)
        if not (MIN_DIM <= ext.max() <= max_dim):
            continue
        # furniture rarely fits fully in frame: relax the in-view fraction
        inb_thr = 0.6 if ext.max() <= 1.2 else 0.3
        vidx = g["vert_idx"]
        sample = verts[rng.choice(vidx, min(N_VIS_SAMPLES, len(vidx)),
                                  replace=False)]

        # --- rank frames by geometric visibility -------------------------
        cand = []
        for name, w2c in w2c_all:
            pc = sample @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            if (z < 0.25).mean() > 0.05:
                continue
            u = pc[:, 0] / z * K[0, 0] + K[0, 2]
            v = pc[:, 1] / z * K[1, 1] + K[1, 2]
            inb = (z > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.mean() < inb_thr:
                continue
            cam = np.linalg.inv(w2c)[:3, 3]
            dirs = sample[inb] - cam
            dist = np.linalg.norm(dirs, axis=1)
            rays = o3d.core.Tensor(np.concatenate(
                [np.broadcast_to(cam, dirs.shape), dirs / dist[:, None]],
                axis=1).astype(np.float32))
            t_hit = scene_full.cast_rays(rays)["t_hit"].numpy()
            vis = float((np.abs(t_hit - dist) < VIS_TOL_M).mean()) * inb.mean()
            bw, bh = u[inb].ptp(), v[inb].ptp()
            if vis < MIN_VIS_FRAC or min(bw, bh) < MIN_BBOX_PX:
                continue
            cand.append((vis * vis * np.sqrt(bw * bh), vis, name, w2c,
                         (u[inb], v[inb])))
        if not cand:
            print(f"[fp] {g['object_id']} {g['label']}: no usable view, skip")
            continue
        cand.sort(key=lambda c: -c[0])

        # --- refine top-k by crop sharpness -------------------------------
        best = None
        for score_geo, vis, name, w2c, (u, v) in cand[:TOP_K_SHARP]:
            img = load_image(name)
            u0, u1 = int(max(u.min(), 0)), int(min(u.max(), W - 1))
            v0, v1 = int(max(v.min(), 0)), int(min(v.max(), H - 1))
            gray = cv2.cvtColor(img[v0:v1 + 1, u0:u1 + 1], cv2.COLOR_RGB2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            s = score_geo * np.clip(sharp / 80.0, 0.5, 1.5)
            if best is None or s > best[0]:
                best = (s, vis, sharp, name, w2c)
        _, vis, sharp, name, w2c = best

        # --- occlusion-aware mask via double raycast ----------------------
        inset = np.zeros(len(verts), bool)
        inset[vidx] = True
        fmask = inset[faces].all(axis=1)
        if fmask.sum() < 4:
            continue
        sub = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces[fmask])))
        scene_inst = o3d.t.geometry.RaycastingScene()
        scene_inst.add_triangles(sub)

        pc = verts[vidx] @ w2c[:3, :3].T + w2c[:3, 3]
        u = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
        v = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
        pad = int(0.15 * max(u.ptp(), v.ptp()) + 8)
        u0, u1 = int(max(u.min() - pad, 0)), int(min(u.max() + pad, W))
        v0, v1 = int(max(v.min() - pad, 0)), int(min(v.max() + pad, H))

        c2w = np.linalg.inv(w2c)
        uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
        d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                          np.ones_like(uu)], axis=-1)
        d_world = (d_cam @ c2w[:3, :3].T).reshape(-1, 3)
        o_world = np.broadcast_to(c2w[:3, 3], d_world.shape)
        rays = o3d.core.Tensor(np.concatenate([o_world, d_world], axis=1)
                               .astype(np.float32))
        t_full = scene_full.cast_rays(rays)["t_hit"].numpy()
        t_inst = scene_inst.cast_rays(rays)["t_hit"].numpy()
        mask = (np.isfinite(t_inst) & (t_inst < t_full + 0.005)) \
            .reshape(uu.shape).astype(np.uint8)
        # transparent/thin objects scan incompletely -> fragmented projected
        # masks; close gaps, fill holes, keep the largest component
        k = max(5, int(0.06 * max(mask.shape)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n_cc, cc = cv2.connectedComponents(mask)
        if n_cc > 2:
            sizes = np.bincount(cc.ravel())[1:]
            mask = (cc == 1 + int(np.argmax(sizes))).astype(np.uint8)
        flood = mask.copy()
        cv2.floodFill(flood, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2),
                                      np.uint8), (0, 0), 1)
        mask = mask | (1 - flood)  # holes = not reachable from the border
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        if mask.sum() < MIN_MASK_PX:
            print(f"[fp] {g['object_id']} {g['label']}: mask {mask.sum()}px, skip")
            continue

        object_id = int(g["object_id"])
        output_index = index_map.get(object_id, kept)
        odir = C.OUT / "objects" / f"obj_{output_index:02d}"
        odir.mkdir(parents=True, exist_ok=True)
        img = load_image(name)
        rgba = np.dstack([img[v0:v1, u0:u1], mask * 255])
        Image.fromarray(rgba).save(odir / "rgba.png")

        sub_tm = trimesh.Trimesh(verts, faces[fmask], process=False)
        gt_pts, _ = trimesh.sample.sample_surface(
            sub_tm, 20_000, seed=GT_SURFACE_SAMPLE_SEED)
        pc_o3d = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(gt_pts)))
        o3d.io.write_point_cloud(str(odir / "gt_points.ply"), pc_o3d)

        meta = {"index": output_index, "label": g["label"], "score": vis,
                "gt_object_id": object_id, "extent": ext,
                "centroid": g["centroid"], "aabb": g["aabb"],
                "frame": name, "vis_frac": vis, "sharpness": float(sharp),
                "n_pts": int(len(vidx)), "bbox_px": [u0, v0, u1, v1],
                "gt_surface_sample_seed": GT_SURFACE_SAMPLE_SEED}
        C.save_json(odir / "meta.json", meta)
        objects.append(meta)
        print(f"[fp] obj_{output_index:02d} {g['label']} (gt {g['object_id']}): "
              f"frame {name} vis={vis:.2f} sharp={sharp:.0f} "
              f"mask={int(mask.sum())}px ext={np.round(ext, 3)}")
        kept += 1
        produced_source_ids.add(object_id)

    if selected is not None and produced_source_ids != selected:
        missing = sorted(selected - produced_source_ids)
        raise RuntimeError(
            "train-only preparation could not produce selected object IDs: "
            f"{missing}"
        )
    objects.sort(key=lambda item: item["index"])
    C.save_json(C.OUT / "objects" / "objects.json", objects)
    if read_frames_out is not None:
        _write_read_frames(read_frames_out, _READ_FRAMES)
    print(f"[fp] prepared {kept}/{len(gts)} whitelist instances")
    return objects


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frame-allowlist",
        help="JSON list (or {frames:[...]}) restricting every candidate frame",
    )
    ap.add_argument(
        "--object-id", action="append", type=int, dest="object_ids",
        help="source instance object_id to prepare; repeat for a selective bundle",
    )
    ap.add_argument(
        "--output-index-map",
        help="JSON object mapping selected source object IDs to preserved output indices",
    )
    ap.add_argument(
        "--read-frames-out",
        help="fresh JSON path receiving the exact successful load_image() frame union",
    )
    args = ap.parse_args(argv)
    frame_allowlist = (
        load_frame_allowlist(args.frame_allowlist) if args.frame_allowlist else None
    )
    index_map = (
        load_output_index_map(args.output_index_map) if args.output_index_map else None
    )
    selected = set(args.object_ids) if args.object_ids is not None else None
    if index_map is not None:
        if selected is None:
            selected = set(index_map)
        elif selected != set(index_map):
            ap.error("--object-id values differ from --output-index-map keys")
    prepare_objects(
        frame_allowlist=frame_allowlist,
        selected_object_ids=selected,
        output_index_by_object_id=index_map,
        read_frames_out=args.read_frames_out,
    )


if __name__ == "__main__":
    main()
