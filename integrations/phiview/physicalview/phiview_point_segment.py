"""SAM3 point prompting on an actual viewer frame, in the inference environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def choose_mask(masks, scores, x, y, box=None):
    import cv2
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("Invalid segmentation output")
    h, w = masks.shape[1:]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click outside image")
    if box is not None:
        b = np.asarray(box)
        if b.shape != (4,) or not np.isfinite(b).all() or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
            raise ValueError('Box outside image')
    # SAM can return disconnected distractors inside a loose rectangle. Keep the
    # component at the prompt center, or the largest visible component for a
    # hollow/occluded object whose center lies in empty space.
    focused = []
    for mask in masks:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        component = int(labels[y, x])
        if component == 0 and box is not None and len(stats) > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        focused.append(labels == component if component else np.zeros_like(mask))
    masks = np.asarray(focused)
    def matches(mask):
        if box is None:
            return mask[y, x]
        x0, y0, x1, y1 = box
        return mask[y0:y1, x0:x1].sum() >= .8*mask.sum()

    valid = [
        i
        for i, mask in enumerate(masks)
        if matches(mask)
        and 24 <= mask.sum() <= h * w * 0.5
        and np.isfinite(scores[i])
        and scores[i] >= 0.5
    ]
    if not valid:
        raise ValueError(
            "No confident object at this point. Try clicking nearer its center."
        )
    i = max(valid, key=lambda i: scores[i])
    return masks[i], float(scores[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--x", type=int)
    ap.add_argument("--y", type=int)
    ap.add_argument("--box", nargs=4, type=int)
    ap.add_argument("--checkpoint", default=os.environ.get("PHIVIEW_SAM3_CHECKPOINT"))
    args = ap.parse_args()
    if args.box is not None:
        args.x = (args.box[0]+args.box[2])//2
        args.y = (args.box[1]+args.box[3])//2
    elif args.x is None or args.y is None:
        ap.error('Supply --x and --y, or --box')
    from PIL import Image
    image = Image.open(args.image).convert('RGB')
    w, h = image.size
    if args.box and not (0 <= args.box[0] < args.box[2] <= w and 0 <= args.box[1] < args.box[3] <= h):
        ap.error('Box outside image')
    if not (0 <= args.x < w and 0 <= args.y < h):
        ap.error('Click outside image')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint = args.checkpoint
    if not checkpoint:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("facebook/sam3", "sam3.pt", local_files_only=True)
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=checkpoint,
        load_from_HF=False,
        enable_inst_interactivity=True,
        compile=False,
    )
    processor = Sam3Processor(model, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        prompt = {'box': np.asarray(args.box, dtype=np.float32)} if args.box else {
            'point_coords': np.array([[args.x, args.y]], dtype=np.float32),
            'point_labels': np.array([1], dtype=np.int32)}
        masks, scores, _ = model.predict_inst(state, multimask_output=True, **prompt)
    mask, score = choose_mask(masks, scores, args.x, args.y, args.box)
    np.save(args.out / "mask.npy", mask)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / "mask.png")
    report = {
        "method": "SAM3 box prompt" if args.box else "SAM3 positive point prompt",
        "box_xyxy": args.box,
        "score": score,
        "pixels": int(mask.sum()),
        "point_xy": [args.x, args.y],
        "checkpoint": str(checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "fresh_inference": True,
    }
    (args.out / "segmentation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
