"""Evaluator-only target masks and standard object-region appearance metrics.

Reuse frozen native renders. Masks come from the unchanged reference simulator,
not from method predictions. Crop SSIM/LPIPS are named as crops, not masked LPIPS.
No pose fitting, resizing, exposure fitting, or construction feedback is allowed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from robo.roundtrip.scorer_sensitivity import sha


METHODS = ("B0_FIXED_NATIVE", "B1_FIXED_PRIORITY", "B2_EVIDENCE", "B3_AGENT_NATIVE")


def crop_from_mask(mask, padding_fraction=.1, min_side=32):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("nonempty boolean reference mask required")
    y, x = np.nonzero(mask); h, w = mask.shape
    side_x = max(min_side, int(math.ceil((x.max() - x.min() + 1) * (1 + 2 * padding_fraction))))
    side_y = max(min_side, int(math.ceil((y.max() - y.min() + 1) * (1 + 2 * padding_fraction))))
    side_x, side_y = min(w, side_x), min(h, side_y)
    left = max(0, min(w - side_x, int(math.floor((x.min() + x.max() + 1 - side_x) / 2))))
    top = max(0, min(h - side_y, int(math.floor((y.min() + y.max() + 1 - side_y) / 2))))
    return left, top, left + side_x, top + side_y


def segmentation_mask(segmentation, geom_ids, geom_type):
    """RoboSuite MjSim.render format is [object TYPE, object ID], not Renderer order."""
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[2] != 2 or not np.issubdtype(seg.dtype, np.integer):
        raise ValueError("RoboSuite segmentation must be HxWx2 integer type/id")
    return (seg[..., 0] == int(geom_type)) & np.isin(seg[..., 1], list(geom_ids))


def render_reference_mask(adapter, camera, width, height):
    """Same camera and visual groups as the pinned native RGB capture."""
    import mujoco
    from scipy.spatial.transform import Rotation
    sim = adapter.native.sim; body = int(adapter.native.obj_body_id["obj"])
    descendants = {body}
    for b in range(body + 1, sim.model.nbody):
        if int(sim.model.body_parentid[b]) in descendants:
            descendants.add(b)
    geom_ids = [g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) in descendants
                and sim.model.geom_group[g] == 1 and sim.model.geom_rgba[g, 3] > 0]
    if not geom_ids:
        raise ValueError("native reference has no visible target geoms")
    name = camera.get("native_name", "robot0_agentview_center")
    cid = sim.model.camera_name2id(name)
    pos, quat = sim.model.cam_pos[cid].copy(), sim.model.cam_quat[cid].copy()
    try:
        if "T_world_from_camera" in camera:
            T = np.asarray(camera["T_world_from_camera"], float)
            parent = sim.model.cam_bodyid[cid]
            W = np.eye(4); W[:3, :3] = sim.data.body_xmat[parent].reshape(3, 3); W[:3, 3] = sim.data.body_xpos[parent]
            local = np.linalg.inv(W) @ T @ np.diag([1., -1., -1., 1.])
            sim.model.cam_pos[cid] = local[:3, 3]
            q = Rotation.from_matrix(local[:3, :3]).as_quat()
            sim.model.cam_quat[cid] = q[[3, 0, 1, 2]]; sim.forward()
        seg = np.asarray(sim.render(width=width, height=height, camera_name=name, segmentation=True))[::-1].copy()
        if seg.shape[:2] != (height, width):
            raise ValueError("segmentation grid differs from held-out RGB")
        return segmentation_mask(seg, geom_ids, mujoco.mjtObj.mjOBJ_GEOM)
    finally:
        sim.model.cam_pos[cid] = pos; sim.model.cam_quat[cid] = quat; sim.forward()


def mask_instance(binding):
    from robo.roundtrip.fidelity_native import checked_heldout, checked_train_render_prelude, static_render_state
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.paired import prepare_paired_adapter
    frames, spec = checked_heldout(binding)
    root = Path(binding["bundle_dir"])
    config = json.loads(Path(binding["native_config"]).read_text())
    source = {"state": json.loads((root / "canonical_state.json").read_text()),
              "xml": (root / "scene.xml").read_text(), "provenance": {"reset_seed": config["reset_seeds"][0]}}
    adapter = RoboCasaAdapter(config)
    try:
        prepare_paired_adapter(adapter, source)
        before = np.asarray(adapter.get_state()["integration_state"]).copy()
        prelude, _ = checked_train_render_prelude(binding)
        for row in prelude:
            adapter.render_capture(row["camera"], width=spec["width"], height=spec["height"])
        for frame in frames:
            rgb = adapter.render_capture(frame["camera"], width=spec["width"], height=spec["height"])["rgb"]
            target = np.asarray(Image.open(frame["reference_rgb"]).convert("RGB"))
            if not np.array_equal(rgb, target):
                yield frame, None, "NATIVE_RENDER_IDENTITY_FAILED"
                continue
            mask = render_reference_mask(adapter, frame["camera"], spec["width"], spec["height"])
            static_render_state(adapter.native.sim.model._model, before, adapter.get_state()["integration_state"])
            yield frame, mask, "READY" if mask.any() else "TARGET_NOT_VISIBLE"
    finally:
        adapter.close()


def export_masks(bindings_path, out, shard_index=0, shard_count=1):
    from robo.roundtrip.matrix import read_rows, save_new
    if not 0 <= shard_index < shard_count:
        raise ValueError("require 0 <= shard-index < shard-count")
    all_bindings = sorted(read_rows(bindings_path), key=lambda b: b["canonical_instance_id"])
    if not all_bindings or len({b["canonical_instance_id"] for b in all_bindings}) != len(all_bindings):
        raise ValueError("nonempty unique canonical binding roster required")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); records = []
    for index, binding in enumerate(all_bindings):
        if index % shard_count != shard_index:
            continue
        iid = binding["canonical_instance_id"]
        try:
            for frame, mask, status in mask_instance(binding):
                row = dict(canonical_instance_id=iid, frame_id=frame["frame_id"], status=status,
                           reference_rgb=frame["reference_rgb"], reference_sha256=sha(frame["reference_rgb"]))
                if status == "READY":
                    path = out / "masks" / iid / (frame["frame_id"] + ".png"); path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                    row.update(mask=str(path), mask_sha256=sha(path), mask_pixels=int(mask.sum()),
                               crop_xyxy=list(crop_from_mask(mask)), width=mask.shape[1], height=mask.shape[0])
                records.append(row)
        except Exception as exc:
            records.append(dict(canonical_instance_id=iid, frame_id=None, status="MASK_INSTANCE_UNAVAILABLE",
                                reason=f"{type(exc).__name__}: {exc}"))
    result = dict(schema_version=1, kind="reference_target_masks", bindings=str(Path(bindings_path).resolve()),
                  bindings_sha256=sha(bindings_path), planned_instances=len(all_bindings),
                  shard_index=shard_index, shard_count=shard_count, records=records,
                  mask_source="native evaluator segmentation; RoboSuite type/id, visual group 1",
                  crop_rule="reference-mask bbox + 10% per side; minimum 32 pixels; clip to image",
                  constructor_access=False, source_sha256=sha(__file__))
    save_new(out / "mask_manifest.json", result); return result


def encode_psnr(value):
    if math.isinf(value) and value > 0:
        return {"masked_psnr": None, "masked_psnr_status": "POSITIVE_INFINITY"}
    if not math.isfinite(value):
        raise ValueError("invalid masked PSNR")
    return {"masked_psnr": float(value), "masked_psnr_status": "FINITE"}


def metrics(mask_manifests, render_roots, out, device="cpu", methods=METHODS, skip_lpips=False):
    from robo.eval.fidelity_metrics import load_rgb, load_mask, psnr, ssim, LPIPSEvaluator
    from robo.roundtrip.matrix import save_new
    from robo.eval.metric_utils import write_csv
    masks = {}; renders = {}; source_hashes = {}; master = None; seen_shards = set()
    for path in map(Path, mask_manifests):
        m = json.loads(path.read_text()); source_hashes[str(path.resolve())] = sha(path)
        identity = (m["bindings_sha256"], m["planned_instances"], m["shard_count"])
        if master is not None and master != identity:
            raise ValueError("mask shards use different canonical cohorts")
        master = identity
        if m["shard_index"] in seen_shards:
            raise ValueError("duplicate mask shard")
        seen_shards.add(m["shard_index"])
        if sha(m["bindings"]) != m["bindings_sha256"]:
            raise ValueError("mask cohort binding changed")
        for row in m["records"]:
            key = (row["canonical_instance_id"], row["frame_id"])
            if key in masks:
                raise ValueError("duplicate target mask record")
            masks[key] = row
    if master is None or seen_shards != set(range(master[2])):
        raise ValueError("all declared mask shards are required, including failed instances")
    for root in map(Path, render_roots):
        paths = [root] if root.is_file() else sorted(root.rglob("appearance_render.json"))
        for path in paths:
            r = json.loads(path.read_text())
            if not r.get("native_import_render_identity") or not all(x["byte_exact"] for x in r["native_import_render_identity"]):
                continue
            source_hashes[str(path.resolve())] = sha(path)
            for row in r["records"]:
                key = (r["canonical_instance_id"], row["frame_id"], row["method"])
                if key in renders and renders[key] != row:
                    if renders[key]["pred_sha256"] != row["pred_sha256"] or renders[key]["reference_sha256"] != row["reference_sha256"]:
                        raise ValueError("multiple distinct predictions for the same frozen frame; choose exact render roots")
                renders[key] = row
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    lpips = None if skip_lpips else LPIPSEvaluator(device)
    if lpips is not None and lpips.model is None:
        raise RuntimeError("pinned LPIPS backend unavailable; install/cache it, or explicitly use --skip-lpips for a partial diagnostic")
    records = []
    for (iid, fid), mask_row in sorted(masks.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        for method in methods:
            row = dict(canonical_instance_id=iid, frame_id=fid, method=method, status=mask_row["status"])
            prediction = renders.get((iid, fid, method))
            if mask_row["status"] == "READY" and prediction is None:
                row["status"] = "RENDER_UNAVAILABLE"
            elif mask_row["status"] == "READY":
                try:
                    for field, checksum in (("pred_rgb", "pred_sha256"), ("reference_rgb", "reference_sha256")):
                        if sha(prediction[field]) != prediction[checksum]:
                            raise ValueError("frozen image changed")
                    if prediction["reference_sha256"] != mask_row["reference_sha256"] or sha(mask_row["mask"]) != mask_row["mask_sha256"]:
                        raise ValueError("mask and render have different reference identities")
                    pred, target = load_rgb(prediction["pred_rgb"]), load_rgb(prediction["reference_rgb"])
                    if pred.shape != target.shape or pred.shape[:2] != (mask_row["height"], mask_row["width"]):
                        raise ValueError("exact image/mask dimensions required; no resizing")
                    mask = load_mask(mask_row["mask"], pred.shape[:2], allow_resize=False, require_nonempty=True)
                    x0, y0, x1, y1 = mask_row["crop_xyxy"]
                    if list(crop_from_mask(mask)) != [x0, y0, x1, y1]:
                        raise ValueError("predeclared reference crop changed")
                    a, b = pred[y0:y1, x0:x1], target[y0:y1, x0:x1]
                    value = None if lpips is None else lpips(a, b)
                    if lpips is not None and (value is None or not np.isfinite(value)):
                        raise ValueError("LPIPS did not return a finite crop distance")
                    row.update(status="MEASURED", **encode_psnr(psnr(pred, target, mask)),
                               crop_ssim=float(ssim(a, b)), crop_lpips=value,
                               masked_mse=float(np.mean((pred[mask].astype(float) - target[mask]) ** 2)),
                               mask_pixels=int(mask.sum()), crop_xyxy=[x0, y0, x1, y1],
                               pred_sha256=prediction["pred_sha256"], mask_sha256=mask_row["mask_sha256"])
                except Exception as exc:
                    row.update(status="METRIC_UNAVAILABLE", reason=f"{type(exc).__name__}: {exc}")
            records.append(row)
    common = {key for key in masks if key[1] is not None and all(any(
        r["canonical_instance_id"] == key[0] and r["frame_id"] == key[1] and r["method"] == method and r["status"] == "MEASURED"
        for r in records) for method in methods)}
    summary = []
    for method in methods:
        available = [r for r in records if r["method"] == method and r["status"] == "MEASURED"]
        paired = [r for r in available if (r["canonical_instance_id"], r["frame_id"]) in common]
        by_instance = {}
        for row in paired:
            by_instance.setdefault(row["canonical_instance_id"], []).append(row)
        def macro(field):
            return float(np.mean([np.mean([r[field] for r in rows]) for rows in by_instance.values()])) if by_instance else None
        any_inf = any(r["masked_psnr_status"] == "POSITIVE_INFINITY" for r in paired)
        summary.append(dict(method=method, planned_instances=master[1],
            available_instances=len({r["canonical_instance_id"] for r in available}),
            common_instances=len(by_instance), common_views=len(paired),
            masked_psnr=None if any_inf else macro("masked_psnr"),
            masked_psnr_status="POSITIVE_INFINITY" if any_inf else "FINITE" if paired else "NOT_MEASURED",
            crop_ssim=macro("crop_ssim"), crop_lpips=None if skip_lpips else macro("crop_lpips"),
            metric_scope="GT-mask PSNR; fixed reference-bbox crop SSIM/LPIPS; instance macro on common support"))
    save_new(out / "object_region_records.jsonl", records, jsonl=True)
    save_new(out / "object_region_summary.json", summary); write_csv(out / "object_region_summary.csv", summary)
    save_new(out / "provenance.json", dict(source_hashes=source_hashes, source_sha256=sha(__file__),
        methods=list(methods), lpips=None if lpips is None else lpips.provenance,
        diagnostic_partial=skip_lpips, heldout_only=True, original_full_frame_results_unchanged=True))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="phase", required=True)
    q = sub.add_parser("masks"); q.add_argument("--bindings", required=True); q.add_argument("--out", required=True)
    q.add_argument("--shard-index", type=int, default=0); q.add_argument("--shard-count", type=int, default=1)
    q = sub.add_parser("metrics"); q.add_argument("--masks", action="append", required=True)
    q.add_argument("--renders", action="append", required=True); q.add_argument("--out", required=True)
    q.add_argument("--device", default="cpu"); q.add_argument("--skip-lpips", action="store_true")
    a = p.parse_args(argv)
    result = export_masks(a.bindings, a.out, a.shard_index, a.shard_count) if a.phase == "masks" else metrics(a.masks, a.renders, a.out, a.device, skip_lpips=a.skip_lpips)
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
