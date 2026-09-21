"""Removal verification, stage 2 (sam3 env, GPU): residual re-detection.

For every removed object: run SAM3 with its class prompt on the CLEAN
render (right half of the before/after jpgs) at each verification view.
Success = the object is no longer detected where it used to be (max
detection score inside its removal-mask region drops below threshold).
Also re-detects on the ORIGINAL render (left half) as the reference score.

Usage: verify_removal_check.py --out-dir SIMANY_OUT
Writes inpaint/verify/verify_report.json (merges stage-1 metrics).
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


def region_score(proc, state, prompt, region_mask):
    import torch as t
    with t.autocast("cuda", dtype=t.bfloat16):
        proc.reset_all_prompts(state)
        state = proc.set_text_prompt(prompt=prompt, state=state)
    masks = state["masks"].squeeze(1).cpu().numpy()
    scores = state["scores"].float().cpu().numpy()
    best = 0.0
    for m, s in zip(masks, scores):
        inter = np.logical_and(m, region_mask).sum()
        if inter > 0.3 * max(m.sum(), 1):
            best = max(best, float(s))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    vdir = out / "inpaint" / "verify"
    report = json.loads((vdir / "verify_render.json").read_text())

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", checkpoint_path=CKPT,
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=0.3)

    summary = []
    for name, rec in report.items():
        prompt = PROMPT_MAP.get(rec["label"].strip().lower(),
                                rec["label"].strip().lower())
        idir = out / "inpaint" / name
        views = json.loads((idir / "views.json").read_text()) \
            if (idir / "views.json").exists() else []
        s_before, s_after, skipped = [], [], []
        for e in rec["views"]:
            side = np.asarray(Image.open(vdir / e["render"]).convert("RGB"))
            h, w2 = side.shape[0], side.shape[1] // 2
            left, right = side[:, :w2], side[:, w2:]
            if e.get("tag") == "held":
                # held views must use the localized region projected by
                # stage 1 (verify/held_regions/obj_XX_<frame>.npz); if it is
                # missing, exclude the view rather than score the whole frame
                rp = vdir / e["region"] if e.get("region") else None
                if rp is None or not rp.exists():
                    skipped.append(e["frame"])
                    continue
                region = np.load(rp)["region"]
                if region.shape != (h, w2):
                    skipped.append(e["frame"])
                    continue
            else:
                region = np.ones((h, w2), bool)
                for k, v in enumerate(views):
                    if v["frame"] == e["frame"] and (idir / f"mask_{k}.png").exists():
                        mfull = np.asarray(Image.open(idir / f"mask_{k}.png")) > 127
                        region = np.asarray(Image.fromarray(mfull).resize(
                            (w2, h), Image.NEAREST)) > 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                st_l = proc.set_image(Image.fromarray(left))
                sb = region_score(proc, st_l, prompt, region)
                st_r = proc.set_image(Image.fromarray(right))
                sa = region_score(proc, st_r, prompt, region)
            e["det_before"], e["det_after"] = sb, sa
            s_before.append(sb)
            s_after.append(sa)
        rec["det_before_mean"] = float(np.mean(s_before)) if s_before else None
        rec["det_after_mean"] = float(np.mean(s_after)) if s_after else None
        rec["removed_ok"] = bool(s_after and max(s_after) < 0.5)
        rec["skipped_views"] = skipped
        covs = [e.get("alpha_cov") for e in rec["views"]
                if e.get("alpha_cov") is not None]
        deps = [e.get("depth_vs_plane_med_mm") for e in rec["views"]
                if e.get("depth_vs_plane_med_mm") is not None]
        summary.append({
            "obj": name, "label": rec["label"],
            "det_before": rec["det_before_mean"],
            "det_after": rec["det_after_mean"],
            "removed_ok": rec["removed_ok"],
            "skipped_views": skipped,
            "alpha_cov": float(np.mean(covs)) if covs else None,
            "depth_plane_mm": float(np.median(deps)) if deps else None})
        print(f"[vc] {name} {rec['label']}: det {rec['det_before_mean']:.2f}"
              f" -> {rec['det_after_mean']:.2f}  removed_ok="
              f"{rec['removed_ok']}")

    ok = sum(1 for s in summary if s["removed_ok"])
    (vdir / "verify_report.json").write_text(json.dumps(
        {"summary": summary, "removal_success_rate": ok / max(len(summary), 1),
         "detail": report}, indent=1))
    print(f"[vc] removal success: {ok}/{len(summary)}")


if __name__ == "__main__":
    main()
