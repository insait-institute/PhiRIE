"""MaskClustering Step-1 substitute: per-frame instance-id masks via SAM3.

Cropformer (their published 2D segmenter) needs an unbuildable CUDA op on
this cluster; using SAM3 for BOTH MaskClustering and our auto_segment makes
the baseline comparison cleaner anyway (isolates the 3D aggregation method).
Writes third_party/MaskClustering/data/scannetpp/data/<seq>/output/mask/
frame_%06d.png (uint16 instance-id maps, 0 = background).

sam3 env. Usage: baseline_mc_masks.py --seq c50d2d1d42 [--stride N]
NB frame density is normally set by baseline_maskclustering.py prepare
--stride (which builds the rgb dir); a matched-density run vs auto_segment
(FRAME_STRIDE=12) should prepare a fresh seq with --stride 12, since
MaskClustering expects a mask png for every prepared frame.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from agents.core import common as _common
CKPT = _common.resolve_sam3_ckpt()
PROMPTS = ["bottle", "mug", "computer mouse", "keyboard", "headphones",
           "telephone", "box", "book", "laptop", "backpack", "shoe",
           "bowl", "plate", "kettle", "plant pot", "speaker"]
SCORE_MIN = 0.45
MC_ROOT = (Path(__file__).resolve().parents[2]
           / "third_party" / "MaskClustering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth prepared frame (default 1 = all)")
    args = ap.parse_args()
    seq_dir = MC_ROOT / "data" / "scannetpp" / "data" / args.seq
    rgb_dir = seq_dir / "iphone" / "rgb"
    out_dir = seq_dir / "output" / "mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", checkpoint_path=CKPT,
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=SCORE_MIN)

    frames = sorted(rgb_dir.glob("frame_*.jpg"))[::args.stride]
    for fi, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        W, H = img.size
        dets = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for prompt in PROMPTS:
                proc.reset_all_prompts(state)
                state = proc.set_text_prompt(prompt=prompt, state=state)
                masks = state["masks"].squeeze(1).cpu().numpy()
                scores = state["scores"].float().cpu().numpy()
                for m, s in zip(masks, scores):
                    a = int(m.sum())
                    if 500 <= a <= 0.15 * W * H:
                        dets.append((float(s), m))
        idmap = np.zeros((H, W), dtype=np.uint16)
        # low-score first so higher-score instances overwrite on overlap
        for k, (s, m) in enumerate(sorted(dets, key=lambda d: d[0])):
            idmap[m] = k + 1
        Image.fromarray(idmap).save(out_dir / fp.name.replace(".jpg", ".png"))
        if fi % 20 == 0:
            print(f"[mcm] {fi + 1}/{len(frames)}: {len(dets)} instances",
                  flush=True)
    print(f"[mcm] wrote {len(frames)} mask maps -> {out_dir}")


if __name__ == "__main__":
    main()
