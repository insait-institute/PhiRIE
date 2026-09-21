"""Factory stage 3: sim-readiness QA + yield report.

For every non-rejected asset: drop it on a flat plane in PyBullet from 5mm,
settle with velocity zeroing, then run 2s free dynamics -- "stable" if it
neither sinks nor walks (<3cm drift). Emits report.json + crops contact
sheet. (Rendering gallery is factory_gallery.py under the mini-viewer env.)
"""
import json

import numpy as np
from PIL import Image, ImageDraw

from agents.core import common as C
from robo.sim.s7_sim import link_pose

DROP_HZ = 240


def drop_test(p, urdf):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / DROP_HZ)
    plane_col = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, plane_col)
    bid = p.loadURDF(str(urdf), basePosition=[0, 0, 0.005],
                     flags=p.URDF_USE_INERTIA_FROM_FILE)
    for _ in range(2 * DROP_HZ):
        p.stepSimulation()
        p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
    # drift is defined on the canonical link frame, not pybullet's
    # inertial/COM frame (s7_sim.link_pose does the conversion)
    pos0, _ = link_pose(p, bid)
    for _ in range(2 * DROP_HZ):
        p.stepSimulation()
    pos1, _ = link_pose(p, bid)
    drift = float(np.linalg.norm(np.array(pos1) - np.array(pos0)))
    sunk = pos1[2] < -0.05
    return {"drift_m": drift, "sunk": bool(sunk),
            "stable": bool(drift < 0.03 and not sunk)}


def main():
    import pybullet as p

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    p.connect(p.DIRECT)
    rows, n_stable = [], 0
    for meta in objects:
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        al = json.loads((odir / "aligned.json").read_text())
        row = {"index": meta["index"], "label": meta["label"],
               "gt_object_id": meta["gt_object_id"], "tier": al["tier"],
               "f1_20": al["eval"]["f1@20mm"]["f1"],
               "f1_40": al["eval"]["f1@40mm"]["f1"],
               "chamfer_mm": al["chamfer_med_m"] * 1000,
               "rejected": al.get("rejected")}
        if not al.get("rejected") and (odir / "object.urdf").exists():
            row["drop_test"] = drop_test(p, odir / "object.urdf")
            n_stable += row["drop_test"]["stable"]
        rows.append(row)
    p.disconnect()

    tiers = [r["tier"] for r in rows]
    summary = {
        "n_instances": len(rows),
        "tier_A": tiers.count("A"), "tier_B": tiers.count("B"),
        "tier_C_rejected": tiers.count("C"),
        "sim_stable": n_stable,
        "yield_AB": (tiers.count("A") + tiers.count("B")) / max(len(rows), 1),
        "mean_f1_20_AB": float(np.mean(
            [r["f1_20"] for r in rows if r["tier"] in "AB"] or [0])),
        "objects": rows,
    }
    C.save_json(C.OUT / "report.json", summary)
    print(f"[fr] {len(rows)} instances: A={summary['tier_A']} "
          f"B={summary['tier_B']} C={summary['tier_C_rejected']} "
          f"stable-on-plane={n_stable}  yield(A+B)={summary['yield_AB']:.0%}")

    # crops contact sheet
    CELL, CAP, PAD, COLS = 190, 34, 6, 5
    n = len(rows)
    R = (n + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * (CELL + PAD) + PAD,
                              R * (CELL + CAP + PAD) + PAD), (250, 250, 247))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        im = Image.open(C.OUT / "objects" / f"obj_{r['index']:02d}" /
                        "rgba.png").convert("RGB")
        im.thumbnail((CELL, CELL))
        x = PAD + (i % COLS) * (CELL + PAD)
        y = PAD + (i // COLS) * (CELL + CAP + PAD)
        sheet.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
        col = {"A": (15, 126, 112), "B": (166, 107, 18), "C": (180, 60, 50)}[r["tier"]]
        d.text((x + 2, y + CELL + 2),
               f"obj_{r['index']:02d} {r['label']}", fill=(34, 37, 42))
        d.text((x + 2, y + CELL + 16),
               f"tier {r['tier']}  F1@20={r['f1_20']:.2f}", fill=col)
    sheet.save(C.OUT / "crops_sheet.png")
    print(f"[fr] wrote {C.OUT / 'report.json'} + crops_sheet.png")


if __name__ == "__main__":
    main()
