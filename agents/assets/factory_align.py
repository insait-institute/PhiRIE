"""Factory stage 2: register TRELLIS assets to their GT instance submesh.

Complete-to-complete registration (far better conditioned than the zero-shot
partial-depth-cloud case) using the same validated alignment objective. Since
image-to-3D assets need not use world z as their canonical up axis, all six
signed source-up hypotheses are evaluated without changing the yaw/scale/ICP
search or the strict quality gates. Quality
tiers on the free GT-based F1:
  A: F1@20mm >= 0.40   B: F1@40mm >= 0.20   C (rejected): below B or size gate
Writes aligned.json per object (schema-compatible with s6/s7/s8).
"""
import json

import numpy as np

from agents.core import common as C
from agents.assets.s5_align import (align_object_with_signed_source_up, apply_T,
                                    f1_eval)

# F1 evidence outranks the size heuristic: scan-incomplete GT (transparent
# bottles) inflates GT extents and used to false-reject F1~0.8 assets.
SIZE_RATIO_RANGE = (0.4, 2.5)
TIER_A_F1_20 = 0.40
TIER_B_F1_40 = 0.20
ALIGN_TGT_SAMPLES = 6000  # subsample the 20k GT points for alignment speed
                          # (F1 evaluation still uses the full set)
ALIGN_MESH_SAMPLE_SEED = 42


def main():
    import open3d as o3d
    import trimesh

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    results = []
    for meta in objects:
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        mesh = trimesh.load(odir / "trellis_mesh.ply", process=False)
        mesh_pts, _ = trimesh.sample.sample_surface(
            mesh, 20_000, seed=ALIGN_MESH_SAMPLE_SEED)
        mesh_pts = np.asarray(mesh_pts, dtype=np.float64)
        tgt = np.asarray(
            o3d.io.read_point_cloud(str(odir / "gt_points.ply")).points)

        tgt_align = tgt[:: max(len(tgt) // ALIGN_TGT_SAMPLES, 1)]
        T, chamfer, chamfer_init, tilt, source_up = \
            align_object_with_signed_source_up(mesh_pts, tgt_align)
        s, _, _ = C.decompose_similarity(T)
        ev = f1_eval(apply_T(T, mesh_pts), tgt)
        world_dims = s * (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        ratio = float(np.max(world_dims)) / max(float(np.max(meta["extent"])), 1e-6)

        rec = {"index": meta["index"], "label": meta["label"], "scale": s,
               "T": T, "chamfer_med_m": chamfer, "chamfer_init_m": chamfer_init,
               "icp_tilt_deg": tilt, "eval": ev, "world_dims": world_dims,
               "size_ratio_vs_obs": ratio,
               "source_up_hypothesis": source_up,
               "alignment_mesh_sample_seed": ALIGN_MESH_SAMPLE_SEED}
        f20 = ev["f1@20mm"]["f1"]
        f40 = ev["f1@40mm"]["f1"]
        if not (SIZE_RATIO_RANGE[0] <= ratio <= SIZE_RATIO_RANGE[1]):
            rec["tier"] = "C"
            rec["rejected"] = f"size ratio {ratio:.2f} outside {SIZE_RATIO_RANGE}"
        elif f20 >= TIER_A_F1_20:
            rec["tier"] = "A"
        elif f40 >= TIER_B_F1_40:
            rec["tier"] = "B"
        else:
            rec["tier"] = "C"
            rec["rejected"] = f"F1@20mm {f20:.2f} / @40mm {f40:.2f} below tier B"
        print(f"[fa] obj_{meta['index']:02d} {meta['label']}: tier {rec['tier']} "
              f"source_up={source_up} s={s:.3f} "
              f"chamfer={chamfer * 1000:.1f}mm "
              f"F1@20={f20:.3f} @40={f40:.3f} ratio={ratio:.2f}")
        C.save_json(odir / "aligned.json", rec)
        results.append(rec)

    C.save_json(C.OUT / "objects" / "aligned_all.json", results)
    tiers = [r["tier"] for r in results]
    print(f"[fa] tiers: A={tiers.count('A')} B={tiers.count('B')} "
          f"C={tiers.count('C')} of {len(tiers)}")


if __name__ == "__main__":
    main()
