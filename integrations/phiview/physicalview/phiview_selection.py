"""Click discovery, visible Gaussian lifting and session-local object persistence."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np


def lift_mask(means, labels, mask, depth, w2c, K):
    """Keep unassigned Gaussians near the visible surface inside the point mask."""
    import torch

    device = means.device
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape or mask.ndim != 2:
        raise ValueError("Mask and displayed depth dimensions differ")
    camera = torch.as_tensor(w2c, dtype=means.dtype, device=device)
    intrinsics = torch.as_tensor(K, dtype=means.dtype, device=device)
    xyz = means @ camera[:3, :3].T + camera[:3, 3]
    uvz = xyz @ intrinsics.T
    uv = torch.floor(uvz[:, :2] / xyz[:, 2:].clamp_min(0.0001)).long()
    h, w = mask.shape
    valid = (labels == 0) & torch.isfinite(xyz).all(1) & (xyz[:, 2] > 0.01)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = torch.where(valid)[0]
    x, y = uv[ids].T
    sampled_depth = torch.as_tensor(depth, device=device)[y, x]
    inside = torch.as_tensor(mask, device=device)[y, x]
    # A single view provides a partial surface, not hidden geometry completion.
    valid = inside & torch.isfinite(sampled_depth) & (sampled_depth > 0.01)
    valid &= (xyz[ids, 2] - sampled_depth).abs() <= 0.08
    return ids[valid]


def record_from_meta(directory, meta):
    from physicalview.scene_state import ObjectRecord

    return ObjectRecord(
        name=meta["name"],
        index=meta["index"],
        label=meta["label"],
        meta=meta,
        aligned=None,
        accepted=False,
        rejected_reason="Interactive selection",
        T_world=None,
        world_dims=np.diff(meta["aabb"], axis=0)[0].tolist(),
        proposals={},
        chosen_source=None,
        hybrid=None,
        physics=None,
        collision_parts=0,
        drop_test=None,
        artifacts={},
        rgba_png=None,
        dir=directory,
    )


def object_directory(state, out, name):
    build = str(state.result_set.out_dir.resolve())
    namespace = hashlib.sha256(build.encode()).hexdigest()[:16]
    return out / 'interactive_objects' / namespace / name


def restore_objects(state, out):
    current = object_directory(state, out, 'obj_00').parent
    directories = sorted((out / 'interactive_objects').glob('obj_*')) + sorted(current.glob('obj_*'))
    for directory in directories:
        matching = directory.parent == current
        meta_path = directory / "selection.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["scene_splat"] != str(state.result_set.splat_ply):
                continue
            if meta.get('scene_build', str(state.result_set.out_dir)) != str(state.result_set.out_dir):
                continue
            # Legacy IDs can collide after SAM3 rediscovery. Never bind by name alone.
            if meta['name'] in state.objects:
                continue
            state.objects[meta["name"]] = record_from_meta(directory, meta)
            matching = True
        proxy_path = directory / "proxy.json"
        if matching and proxy_path.exists() and directory.name in state.objects:
            proxy = json.loads(proxy_path.read_text())
            state.objects[directory.name].meta["interactive_proxy"] = proxy
            state.objects[directory.name].physics = proxy["physics"]
    # Keep superseded click files as provenance; match receipts, not IDs alone.
    for rec in list(state.objects.values()):
        for old in rec.meta.get('supersedes', []):
            previous = state.objects.get(old['name'])
            if previous and previous.meta.get('selection_work') == old['selection_work']:
                state.objects.pop(old['name'])


class ClickSelection:
    def __init__(self, demo):
        self.demo = demo
        self.thread = None
        self.pending = None
        self.status = {"state": "idle"}
        self.generation = 0
        self.lock = threading.Lock()

    def cancel(self):
        # A running GPU call may finish, but cannot republish a cancelled result.
        with self.lock:
            self.generation += 1
            self.pending = None
            self.status = {"state": "idle", "message": "Selection cleared."}

    @property
    def busy(self):
        return self.pending is not None or (
            self.thread is not None and self.thread.is_alive()
        )

    def start(self, fid, x, y, box=None, preset=None):
        d = self.demo
        if self.busy or (d.pipeline_thread and d.pipeline_thread.is_alive()):
            raise ValueError("Wait for the current object or scene build to finish")
        if fid not in d.frame_images:
            raise ValueError("Displayed frame expired; click the refreshed image")
        _, depth, w2c, K = d.frames[fid]
        sample = depth[y:y+1, x:x+1] if box is None else depth[box[1]:box[3], box[0]:box[2]]
        if not (np.isfinite(sample) & (sample > .01)).any():
            raise ValueError("No reconstructed surface at this point")
        d.physics.running = False
        d.selected = None
        work = d.out / "clicks" / str(time.time_ns())
        work.mkdir(parents=True)
        from PIL import Image

        Image.fromarray(d.frame_images[fid]).save(work / "image.png")
        np.save(work / "depth.npy", depth)
        camera = {
            "frame": fid,
            "w2c": w2c.tolist(),
            "K": K.tolist(),
            "point_xy": [x, y],
            "box_xyxy": box,
        }
        (work / "camera.json").write_text(json.dumps(camera, indent=2))
        self.status = {"state": "running", "message": "Finding the boxed object…" if box else "Finding the clicked object…"}
        generation = self.generation

        def worker():
            try:
                argv = [
                    str(d.config.interpreter("sam3")),
                    str(Path(__file__).with_name("phiview_point_segment.py")),
                    "--image",
                    str(work / "image.png"),
                    "--out",
                    str(work),
                ]
                argv += ['--box', *map(str, box)] if box is not None else ['--x', str(x), '--y', str(y)]
                env = {
                    **os.environ,
                    "HF_HOME": "/group/worldcept/hf_cache",
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                }
                with (work / "inference.log").open("w") as log:
                    subprocess.run(
                        argv,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=180,
                        check=True,
                    )
                with self.lock:
                    if generation == self.generation:
                        self.pending = (work, depth, w2c, K, preset)
            except Exception as exc:
                with self.lock:
                    if generation == self.generation:
                        self.status = {
                            "state": "failed",
                            "message": "Object selection failed; try another point.",
                            "error": str(exc),
                            "log": str(work / "inference.log"),
                        }

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def finish(self):
        if self.pending is None:
            return
        d = self.demo
        work, depth, w2c, K, preset = self.pending
        self.pending = None
        try:
            mask = np.load(work / "mask.npy")
            idx = lift_mask(d.scene.raw["means"], d.scene.labels, mask, depth, w2c, K)
            supersedes = []
            if preset:
                from physicalview.phiview_demo import complete_fragments
                idx, supersedes = complete_fragments(d, mask, depth, w2c, K, idx)
            if len(idx) < 24:
                raise ValueError(
                    "Too little unassigned 3D surface; try another view or point"
                )
            points = d.scene.raw["means"][idx].detach().cpu().numpy()
            lo, hi = np.quantile(points, [0.001, 0.999], axis=0)
            index = max((rec.index for rec in d.state.objects.values()), default=-1) + 1
            name = f"obj_{index:02d}"
            directory = object_directory(d.state, d.out, name)
            directory.mkdir(parents=True)
            np.save(directory / "gaussian_indices.npy", idx.cpu().numpy())
            meta = {
                "name": name,
                "index": index,
                "label": preset['label'] if preset else f"Clicked object {index}",
                "centroid": ((lo + hi) / 2).tolist(),
                "aabb": [lo.tolist(), hi.tolist()],
                "scene_splat": str(d.state.result_set.splat_ply),
                "scene_build": str(d.state.result_set.out_dir),
                "selection_work": str(work),
                "mask_source": "SAM3 prompted mask lifted to visible Gaussians (partial surface)",
                "gaussians": len(idx),
                "interactive": True,
            }
            if preset:
                meta.update(demo_preset=preset['id'], supersedes=supersedes)
            (directory / "selection.json").write_text(json.dumps(meta, indent=2))
            rec = record_from_meta(directory, meta)
            for old in supersedes:
                d.state.objects.pop(old['name'])
            d.state.objects[name] = rec
            d.scene.add_object(name, idx, meta["mask_source"], supersedes=[old['name'] for old in supersedes])
            d.selected = name
            d.mode = "original"
            d.frames.clear()
            d.frame_images.clear()
            d.dirty = True
            self.status = {
                "state": "selected",
                "message": "Object selected. Click Make simulatable.",
                "object": name,
                "gaussians": len(idx),
            }
        except Exception as exc:
            self.status = {"state": "failed", "message": str(exc)}
