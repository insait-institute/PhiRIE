"""SHARP-variant hybrid V_mesh stage: per object, keep whichever of the
existing TRELLIS asset (stage 4/s4_trellis.py, already run) or a freshly
generated ReconViaGen asset (VGGT-conditioned TRELLIS fed the SHARP-splat's
own near-field synthetic views from sharp_render_object_views.py) registers
better -- winner selection is verbatim factory_hybrid.py's sym_score
arbitration, just with tgt = this object's own single-view lifted
points.ply (there is no gt_points.ply here -- no GT exists for this input).

Expected outcome, NOT a bug: RVG may lose most/all comparisons, since its
"extra views" are near-field renders of one already-hallucinated splat, not
independent real multi-view observations -- there is no new information for
RVG's VGGT conditioning to exploit, only re-renders of TRELLIS's single
best-view competitor's actual data source. The existing winner-selection
machinery is left to arbitrate exactly as designed.

CAVEAT (see final report): make_align_record()'s internal f1_eval scores the
registered mesh against `tgt` (=points.ply) as if it were ground truth. Since
tgt here is just this object's own single-view partial observation (not an
independent GT mesh), those numbers mean "how well does the registered mesh
explain the one observed view," NOT true 3D reconstruction accuracy. We
relabel them fit@20mm/fit@40mm in this script's own printed/summarized
output; the underlying JSON keys (reused verbatim from s5_align.py /
factory_hybrid.py) remain f1@20mm/f1@40mm.

.venv, GPU (VGGT + TRELLIS conditioning networks). No CLI args.
"""
import json
import time
import traceback

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from agents.core import common as C
# Importing factory_hybrid pulls in s4_reconviagen at module level (sys.path
# mutation for ReconViaGen's vendored `trellis` fork + the xformers/rembg
# patches) as a documented, harmless side effect -- see module docstring.
# Do NOT also `import s4_trellis` in this process: its own sys.path.insert
# of the plain-TRELLIS repo would shadow the RVG fork's `trellis` package
# (factory_hybrid.py:52-53).
from agents.assets.factory_hybrid import (snapshot_trellis, make_align_record, sample_mesh,
                            registered_residual, materialize_rvg, materialize_trellis, copy_atomic)
from agents.assets.s5_align import apply_T, sym_score

N_VIEWS = 12


def load_rvg_pipe():
    import torch
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    pipe = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    # their low_vram path forgets to move the slat conditioning modules;
    # A6000 fits the whole no-refinement pipeline (same workaround as
    # s4_reconviagen.py / factory_hybrid.py)
    pipe.low_vram = False
    if hasattr(pipe, "cuda"):
        pipe.cuda()
    for mod in pipe.models.values():
        mod.to("cuda")
    for attr in vars(pipe).values():
        if isinstance(attr, torch.nn.Module):
            attr.to("cuda")
    return pipe


def gen_rvg(pipe, odir, meta):
    """Generate rvg/rvg_mesh.ply + rvg_gs.ply from the pre-rendered
    obj_XX/rvg/view_NN.png crops (cached if both already exist)."""
    import torch
    import trimesh
    from PIL import Image

    rdir = odir / "rvg"
    if (rdir / "rvg_mesh.ply").exists() and (rdir / "rvg_gs.ply").exists():
        print(f"[sh] obj_{meta['index']:02d}: rvg asset cached, skip gen")
        return 0.0

    view_paths = sorted(rdir.glob("view_*.png"))
    if len(view_paths) < 2:
        raise RuntimeError(f"only {len(view_paths)} usable views (<2)")

    t0 = time.time()
    imgs = [pipe.preprocess_image(Image.open(p)) for p in view_paths]
    print(f"[sh] obj_{meta['index']:02d} {meta['label']}: {len(imgs)} views")

    out, _, _ = pipe.run(image=imgs, seed=42, formats=["mesh", "gaussian"],
                         preprocess_image=False)
    mesh = out["mesh"][0]
    v = mesh.vertices.cpu().numpy().astype(np.float64)
    f = mesh.faces.cpu().numpy()
    cols = None
    if getattr(mesh, "vertex_attrs", None) is not None \
            and mesh.vertex_attrs.shape[-1] >= 3:
        cols = (mesh.vertex_attrs[:, :3].clamp(0, 1).cpu().numpy()
                * 255).astype(np.uint8)
    trimesh.Trimesh(v, f, vertex_colors=cols, process=False).export(
        rdir / "rvg_mesh.ply")
    # RVG's Gaussian.save_ply(path) takes no transform kwarg -- passing
    # transform=None explicitly is WRONG for this fork (documented footgun,
    # matches factory_hybrid.py's run_rvg and the operator's own memory notes)
    out["gaussian"][0].save_ply(str(rdir / "rvg_gs.ply"))
    n_gs = out["gaussian"][0].get_xyz.shape[0]
    del out
    torch.cuda.empty_cache()
    dt = time.time() - t0
    print(f"[sh] obj_{meta['index']:02d}: rvg mesh {len(v)}v/{len(f)}f, "
          f"gs {n_gs}, {dt:.0f}s")
    return dt


def main():
    t_start = time.time()
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    pipe = None
    rows = []

    for meta in objects:
        idx = meta["index"]
        odir = C.OUT / "objects" / f"obj_{idx:02d}"
        if not (odir / "aligned.json").exists() \
                or not (odir / "trellis_mesh.ply").exists():
            print(f"[sh] obj_{idx:02d} {meta['label']}: no TRELLIS asset, skip")
            continue

        tdir = snapshot_trellis(odir)
        trec = json.loads((tdir / "aligned.json").read_text())
        if trec.get("rejected"):
            print(f"[sh] obj_{idx:02d} {meta['label']}: TRELLIS rejected "
                  f"({trec['rejected']}), no viable candidate, skip")
            continue

        tgt = np.asarray(o3d.io.read_point_cloud(str(odir / "points.ply")).points)
        tgt_tree = cKDTree(tgt)
        res_t = registered_residual(tdir / "trellis_mesh.ply", trec["T"],
                                    tgt, tgt_tree)

        row = {"index": idx, "label": meta["label"],
               "criterion": "sym_score (symmetric clipped chamfer, clip=0.03m) "
                            "vs this object's own points.ply -- a single-view "
                            "partial cloud, NOT ground truth (see fit@*mm "
                            "caveat in module docstring / final report)",
               "sym_chamfer_trellis_m": res_t,
               "trellis": {"chamfer_med_m": trec["chamfer_med_m"],
                           "scale": trec["scale"]},
               "sym_chamfer_rvg_m": None, "rvg": None, "rvg_error": None,
               "rvg_skip_reason": None}

        skip_marker = odir / "rvg" / ".skip_rvg"
        rrec = None
        try:
            if skip_marker.exists():
                row["rvg_skip_reason"] = "no_views"
                raise RuntimeError(skip_marker.read_text().strip())
            if pipe is None:
                pipe = load_rvg_pipe()
            row["gen_seconds"] = gen_rvg(pipe, odir, meta)
            mesh_r, pts_r = sample_mesh(odir / "rvg" / "rvg_mesh.ply")
            rrec = make_align_record(meta, mesh_r, pts_r, tgt)
            C.save_json(odir / "rvg" / "aligned.json", rrec)
            res_r = float(sym_score(apply_T(np.asarray(rrec["T"]), pts_r),
                                    tgt, tgt_tree))
            row["sym_chamfer_rvg_m"] = res_r
            row["rvg"] = {"chamfer_med_m": rrec["chamfer_med_m"],
                          "scale": rrec["scale"], "rejected": rrec.get("rejected"),
                          "fit_20mm": rrec["eval"]["f1@20mm"]["f1"],
                          "fit_40mm": rrec["eval"]["f1@40mm"]["f1"]}
        except Exception as e:
            traceback.print_exc()
            row["rvg_error"] = f"{type(e).__name__}: {e}"

        if row["rvg"] is not None and not rrec.get("rejected") \
                and row["sym_chamfer_rvg_m"] < row["sym_chamfer_trellis_m"]:
            row["winner"] = "rvg"
        else:
            row["winner"] = "trellis"
        row["canonical_source"] = row["winner"]

        if row["winner"] == "rvg":
            materialize_rvg(odir, meta, rrec)
        else:
            materialize_trellis(odir, meta)
        C.save_json(odir / "hybrid.json", row)
        rv = row["rvg"] or {}
        print(f"[sh] obj_{idx:02d} {meta['label']}: WINNER {row['winner']}  "
              f"sym {res_t * 1000:.1f}mm vs "
              f"{(row['sym_chamfer_rvg_m'] or float('nan')) * 1000:.1f}mm  "
              f"fit@20 n/a vs {rv.get('fit_20mm', float('nan')):.3f}")
        rows.append(row)

    # refresh objects/aligned_all.json so downstream report tooling sees winners
    all_path = C.OUT / "objects" / "aligned_all.json"
    backup = C.OUT / "objects" / "aligned_all_prehybrid.json"
    if all_path.exists() and not backup.exists():
        copy_atomic(all_path, backup)
    recs = []
    for meta in objects:
        aj = C.OUT / "objects" / f"obj_{meta['index']:02d}" / "aligned.json"
        if aj.exists():
            recs.append(json.loads(aj.read_text()))
    C.save_json(all_path, recs)

    mean = lambda vals: float(np.mean(vals)) if vals else None
    n_rvg = sum(r["winner"] == "rvg" for r in rows)
    n_no_views = sum(r["rvg_skip_reason"] == "no_views" for r in rows)
    n_other_err = sum(r["rvg_error"] is not None and r["rvg_skip_reason"] is None
                      for r in rows)
    summary = {
        "n_objects": len(rows), "n_rvg_wins": n_rvg,
        "n_rvg_skipped_no_views": n_no_views,
        "n_rvg_other_errors": n_other_err,
        "mean_sym_chamfer_trellis_mm": mean(
            [r["sym_chamfer_trellis_m"] * 1000 for r in rows]),
        "mean_sym_chamfer_rvg_mm": mean(
            [r["sym_chamfer_rvg_m"] * 1000 for r in rows
             if r["sym_chamfer_rvg_m"] is not None]),
        "wall_seconds": time.time() - t_start, "rows": rows}
    C.save_json(C.OUT / "objects" / "hybrid_all.json", summary)
    print(f"[sh] DONE: {len(rows)} objects, RVG wins {n_rvg}, "
          f"no-views {n_no_views}, other-errors {n_other_err}, "
          f"{summary['wall_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
