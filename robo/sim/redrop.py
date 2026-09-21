"""Re-run the factory drop test with the COM->link frame correction.

factory_report.py's drop_test() used to measure drift in pybullet's
inertial/COM frame; this replays the identical test (same settle procedure,
same 3cm/2s criterion) for every outputs/*_factory and outputs/*_auto scene
with poses converted to the URDF link frame (s7_sim.link_pose) and writes
outputs/<scene>/drop_v2.json. report.json is left untouched. The raw COM
drift from the same trajectory is kept alongside so classification flips are
attributable to the frame fix alone.
"""
import json
from pathlib import Path

import numpy as np
import pybullet as p

from robo.sim.s7_sim import link_pose

ROOT = Path(__file__).resolve().parents[2]
DROP_HZ = 240


def drop_test_v2(urdf):
    # identical to factory_report.drop_test() except the two measured poses
    # are converted COM -> link frame
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
    raw0, _ = p.getBasePositionAndOrientation(bid)
    pos0, _ = link_pose(p, bid)
    for _ in range(2 * DROP_HZ):
        p.stepSimulation()
    raw1, _ = p.getBasePositionAndOrientation(bid)
    pos1, _ = link_pose(p, bid)
    drift = float(np.linalg.norm(np.array(pos1) - np.array(pos0)))
    sunk = pos1[2] < -0.05
    drift_raw = float(np.linalg.norm(np.array(raw1) - np.array(raw0)))
    sunk_raw = raw1[2] < -0.05
    return {"drift_m": drift, "sunk": bool(sunk),
            "stable": bool(drift < 0.03 and not sunk),
            "drift_raw_m": drift_raw,
            "stable_raw": bool(drift_raw < 0.03 and not sunk_raw)}


def main():
    p.connect(p.DIRECT)
    scenes = sorted(list((ROOT / "outputs").glob("*_factory"))
                    + list((ROOT / "outputs").glob("*_auto")))
    for scene in scenes:
        objects = json.loads((scene / "objects" / "objects.json").read_text())
        rows, n_stable = [], 0
        for meta in objects:
            odir = scene / "objects" / f"obj_{meta['index']:02d}"
            al = json.loads((odir / "aligned.json").read_text())
            if al.get("rejected") or not (odir / "object.urdf").exists():
                continue
            r = drop_test_v2(odir / "object.urdf")
            rows.append({"name": odir.name, "label": meta["label"], **r})
            n_stable += r["stable"]
        out = {"n_tested": len(rows), "n_stable": n_stable,
               "stable_rate": n_stable / max(len(rows), 1), "objects": rows}
        (scene / "drop_v2.json").write_text(json.dumps(out, indent=1))
        print(f"[redrop] {scene.name}: {n_stable}/{len(rows)} stable",
              flush=True)
    p.disconnect()


if __name__ == "__main__":
    main()
