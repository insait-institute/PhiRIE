"""Stage 1: SAM3 text-prompted instance segmentation on the representative frame.

MUST run under the sam3 env: ${SIMANY_SAM3_PY}
(standalone on purpose: no common.py import, torch versions differ).

Usage: s1_segment.py --image X.JPG --out-dir OUT/masks \
                     --prompts bottle mug "computer mouse" ...
Writes masks.npz {masks (N,H,W) bool, labels, scores} + overlay.png.
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


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device="cuda", checkpoint_path=CKPT,
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=args.threshold)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size
    dets = []  # (label, score, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(image)
        for prompt in args.prompts:
            proc.reset_all_prompts(state)
            state = proc.set_text_prompt(prompt=prompt, state=state)
            masks = state["masks"].squeeze(1).cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
            for m, s in zip(masks, scores):
                dets.append((prompt, float(s), m.astype(bool)))
            print(f"[s1] '{prompt}': {len(masks)} instances "
                  f"(scores {np.round(scores, 2).tolist()})")

    # area + border gates
    kept = []
    for label, score, m in dets:
        area = int(m.sum())
        if not (400 <= area <= 0.08 * W * H):
            continue
        edges = sum([m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any()])
        if edges >= 3:
            continue
        kept.append((label, score, m))
    # cross/within-prompt dedup: greedy by score, reject IoU > 0.5
    kept.sort(key=lambda d: -d[1])
    final = []
    for d in kept:
        if all(iou(d[2], f[2]) < 0.5 for f in final):
            final.append(d)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [d[0] for d in final]
    scores = [d[1] for d in final]
    masks = np.stack([d[2] for d in final]) if final else np.zeros((0, H, W), bool)
    np.savez_compressed(out / "masks.npz", masks=masks,
                        labels=np.array(labels), scores=np.array(scores))

    rgb = np.asarray(image).astype(np.float32)
    rng = np.random.RandomState(3)
    overlay = rgb.copy()
    for i, (label, score, m) in enumerate(final):
        color = rng.randint(60, 255, 3).astype(np.float32)
        overlay[m] = 0.45 * overlay[m] + 0.55 * color
    Image.fromarray(overlay.astype(np.uint8)).save(out / "overlay.png")
    (out / "detections.json").write_text(json.dumps(
        [{"i": i, "label": l, "score": round(s, 4), "area": int(m.sum())}
         for i, (l, s, m) in enumerate(final)], indent=1))
    print(f"[s1] kept {len(final)} instances -> {out}/masks.npz")


if __name__ == "__main__":
    main()
