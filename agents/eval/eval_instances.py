"""Instance-discovery evaluation: predicted instances (auto_instances.npz)
vs GT whitelist instances, matched by mesh-vertex IoU.

Metrics: precision/recall/F1 at IoU 0.25 and 0.5 (greedy matching, one GT
per prediction), plus counts. Works for auto_segment and for converted
MaskClustering output (same npz contract).

Usage: eval_instances.py --npz <auto_instances.npz>
"""
import argparse

import numpy as np

from agents.core import common as C
from agents.discover.factory_prepare import WHITELIST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    gts = [g for g in C.load_gt_instances()
           if g["label"].strip().lower() in WHITELIST]
    d = np.load(args.npz)
    preds = []
    for k, label in enumerate(d["labels"]):
        preds.append({"label": str(label), "score": float(d["scores"][k]),
                      "verts": set(d[f"vert_idx_{k}"].tolist())})
    gt_sets = [set(g["vert_idx"].tolist()) for g in gts]

    # Matching is vertex-IoU only, label-free: same class-agnostic protocol
    # for ours and MaskClustering by construction.
    for tau in (0.25, 0.5):
        used = set()
        tp = 0
        for p in sorted(preds, key=lambda x: -x["score"]):
            best_iou, best_j = 0.0, None
            for j, gs in enumerate(gt_sets):
                if j in used:
                    continue
                iou = len(p["verts"] & gs) / max(len(p["verts"] | gs), 1)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= tau:
                used.add(best_j)
                tp += 1
        prec = tp / max(len(preds), 1)
        rec = tp / max(len(gts), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"IoU@{tau}: TP={tp}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"(pred {len(preds)}, gt {len(gts)})")


if __name__ == "__main__":
    main()
