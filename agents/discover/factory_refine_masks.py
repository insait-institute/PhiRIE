"""Factory stage 1.5: refine projected GT masks with SAM3 image evidence.

Transparent/reflective objects (bottles!) scan incompletely, so the
mesh-projected mask misses most of the object. On each object's best frame,
run SAM3 with a class prompt and swap in the SAM3 instance that best
overlaps the projected mask -- image evidence recovers the parts the scan
never captured. Falls back to the projected mask when nothing matches.

MUST run under the sam3 env (standalone, like s1_segment.py).
Usage: factory_refine_masks.py --images-dir D --out-dir SIMANY_OUT
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from agents.core import common as _common
CKPT = _common.resolve_sam3_ckpt()

PROMPT_MAP = {
    "plastic bottle": "bottle", "glass bottle": "bottle",
    "water bottle": "bottle", "cup": "mug", "mouse": "computer mouse",
    "carboard box": "cardboard box", "storage box": "box",
}
MIN_IOU = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()
    out = Path(args.out_dir)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda", checkpoint_path=CKPT,
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=args.threshold)

    objects = json.loads((out / "objects" / "objects.json").read_text())
    by_frame = {}
    for m in objects:
        by_frame.setdefault(m["frame"], []).append(m)

    n_swap = 0
    for frame, metas in by_frame.items():
        image = Image.open(Path(args.images_dir) / frame).convert("RGB")
        W, H = image.size
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(image)
            cache = {}
            for m in metas:
                prompt = PROMPT_MAP.get(m["label"].strip().lower(),
                                        m["label"].strip().lower())
                if prompt not in cache:
                    proc.reset_all_prompts(state)
                    state = proc.set_text_prompt(prompt=prompt, state=state)
                    cache[prompt] = state["masks"].squeeze(1).cpu().numpy()
                sam_masks = cache[prompt]

                odir = out / "objects" / f"obj_{m['index']:02d}"
                rgba = np.asarray(Image.open(odir / "rgba.png"))
                u0, v0, u1, v1 = m["bbox_px"]
                proj = np.zeros((H, W), bool)
                proj[v0:v1, u0:u1] = rgba[..., 3] > 127

                best_iou, best = 0.0, None
                for sm in sam_masks:
                    inter = np.logical_and(sm, proj).sum()
                    iou = inter / max(np.logical_or(sm, proj).sum(), 1)
                    if iou > best_iou:
                        best_iou, best = iou, sm
                if best is None or best_iou < MIN_IOU:
                    print(f"[frm] obj_{m['index']:02d} {m['label']}: keep "
                          f"projected mask (best IoU {best_iou:.2f})")
                    continue

                vv, uu = np.nonzero(best)
                pad = int(0.15 * max(np.ptp(vv), np.ptp(uu)) + 8)
                nv0, nv1 = max(vv.min() - pad, 0), min(vv.max() + pad, H)
                nu0, nu1 = max(uu.min() - pad, 0), min(uu.max() + pad, W)
                img_np = np.asarray(image)
                new = np.dstack([img_np[nv0:nv1, nu0:nu1],
                                 (best[nv0:nv1, nu0:nu1] * 255).astype(np.uint8)])
                Image.fromarray(new).save(odir / "rgba.png")
                m["bbox_px"] = [int(nu0), int(nv0), int(nu1), int(nv1)]
                m["mask_source"] = f"sam3 iou={best_iou:.2f}"
                meta = json.loads((odir / "meta.json").read_text())
                meta["bbox_px"] = m["bbox_px"]
                meta["mask_source"] = m["mask_source"]
                (odir / "meta.json").write_text(json.dumps(meta, indent=1))
                n_swap += 1
                print(f"[frm] obj_{m['index']:02d} {m['label']}: SAM3 mask "
                      f"(IoU {best_iou:.2f}, {int(best.sum())}px vs "
                      f"{int(proj.sum())}px projected)")

    (out / "objects" / "objects.json").write_text(json.dumps(objects, indent=1))
    print(f"[frm] refined {n_swap}/{len(objects)} masks with SAM3")


if __name__ == "__main__":
    main()
