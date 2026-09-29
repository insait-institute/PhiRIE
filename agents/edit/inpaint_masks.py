"""Inpainting stage 2 (sam3 env, GPU): image-evidence masks for removal.

The projected GT mask misses parts the scan never captured (transparent
bottles, thin structures) - inpainting with an incomplete mask leaves object
ghosts. Per related view: SAM3 with the class prompt, take instances with
IoU > MIN_IOU against the projected mask, and store union(projected, SAM3)
dilated - the final 2D removal mask for the Qwen inpainting stage.

Usage: inpaint_masks.py --images-dir D --out-dir SIMANY_OUT
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageFilter

from agents.core import common as _common
PROMPT_MAP = {
    "plastic bottle": "bottle", "glass bottle": "bottle",
    "water bottle": "bottle", "cup": "mug", "mouse": "computer mouse",
    "carboard box": "cardboard box", "storage box": "box",
}
MIN_IOU = 0.15
DILATE = 9


def _merge_mask(projected, candidates):
    projected = np.asarray(projected)
    candidates = np.asarray(candidates)
    if (projected.ndim != 2 or projected.dtype != bool or candidates.ndim != 3
            or candidates.shape[1:] != projected.shape or not np.isfinite(candidates).all()
            or not np.isin(candidates, [0, 1]).all()):
        raise ValueError('SAM3/projected masks have invalid shape or values')
    final = projected.copy()
    selected = 0
    for mask in candidates.astype(bool):
        inter = np.logical_and(mask, projected).sum()
        if inter / max(np.logical_or(mask, projected).sum(), 1) > MIN_IOU:
            final |= mask
            selected += 1
    final = np.asarray(Image.fromarray((final * 255).astype(np.uint8))
        .filter(ImageFilter.MaxFilter(DILATE))) > 127
    return final, selected


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda", checkpoint_path=_common.resolve_sam3_ckpt(),
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=0.4)

    objects = json.loads((out / "objects" / "objects.json").read_text())
    jobs = []  # (frame, obj_meta, view_index)
    for m in objects:
        odir = out / "inpaint" / f"obj_{m['index']:02d}"
        if not (odir / "proj_masks.npz").exists():
            continue
        pmz = np.load(odir / "proj_masks.npz")
        for k, frame in enumerate(pmz["frames"]):
            jobs.append((str(frame), m, k))

    by_frame = {}
    for frame, m, k in jobs:
        by_frame.setdefault(frame, []).append((m, k))

    for frame, items in by_frame.items():
        image = Image.open(Path(args.images_dir) / frame).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(image)
            cache = {}
            for m, k in items:
                odir = out / "inpaint" / f"obj_{m['index']:02d}"
                pmz = np.load(odir / "proj_masks.npz")
                proj = pmz["masks"][k]
                prompt = PROMPT_MAP.get(m["label"].strip().lower(),
                                        m["label"].strip().lower())
                if prompt not in cache:
                    proc.reset_all_prompts(state)
                    state = proc.set_text_prompt(prompt=prompt, state=state)
                    cache[prompt] = state["masks"].squeeze(1).cpu().numpy()
                final = _merge_mask(proj, cache[prompt])[0]
                fpath = odir / f"mask_{k}.png"
                Image.fromarray((final * 255).astype(np.uint8)).save(fpath)
                print(f"[im] obj_{m['index']:02d} view{k} {frame}: "
                      f"proj {int(proj.sum())}px -> final {int(final.sum())}px")
    print("[im] done")


if __name__ == "__main__":
    main()
