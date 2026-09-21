"""Select the zero-shot scene set: scenes with good manipulation objects.

Scores every ScanNet++ v2 scene that has BOTH a prebuilt SceneSplat gsplat
(so the photoreal composite path works) AND GT instance annotations (used
only for *selection statistics* — the zero-shot pipeline itself never reads
them), then takes the top N. The paper's 50-scene nvs_sem_val benchmark
split is excluded so the zero-shot set is disjoint from the tuned benchmark.

Per scene, from segments_anno.json OBBs:
  manip      objects whose label is in the pipeline WHITELIST and whose max
             OBB extent passes the pipeline gates (0.03-0.8 m)
  graspable  the subset that a Robotiq 2F-85 could actually close on:
             min extent <= 0.085 m (fits the 85 mm stroke on some axis),
             0.04 <= max extent <= 0.35 m (not a sliver, liftable)
  diversity  number of distinct manip categories (a desk with 6 different
             objects beats a shelf with 202 books; per-category counts are
             capped so monocultures don't dominate)
  workspace  presence of a table/desk/counter (a surface to manipulate on)
  psnr       SceneSplat render fidelity from meta.csv (the composite
             observation quality ceiling); scenes below --min-psnr are out

score = 2.0*graspable_capped + 0.5*(manip_capped - graspable_capped)
      + 1.5*diversity + 2.0*workspace + 0.5*clip(psnr - min_psnr, 0, 8)

Eligibility: graspable >= 3 and psnr >= --min-psnr.

Usage: python -m agents.eval.select_manip_scenes [--n 100] [--out data/]
"""
import argparse
import csv
import json
from pathlib import Path

from agents.discover.factory_prepare import WHITELIST

WORKSPACE_LABELS = {
    "table", "office table", "desk", "office desk", "dining table",
    "coffee table", "side table", "kitchen counter", "counter",
    "countertop", "kitchen island", "workbench", "desk table",
}
PER_CATEGORY_CAP = 6
GRIPPER_STROKE = 0.085


def scene_stats(anno_path: Path):
    groups = json.loads(anno_path.read_text())["segGroups"]
    per_cat = {}
    grasp_per_cat = {}
    workspace = False
    for g in groups:
        label = g.get("label", "").strip().lower()
        if label in WORKSPACE_LABELS:
            workspace = True
        if label not in WHITELIST:
            continue
        obb = g.get("obb") or {}
        ext = sorted(obb.get("axesLengths", [0, 0, 0]))
        if not (0.03 <= ext[-1] <= 0.8):          # pipeline size gates
            continue
        per_cat[label] = per_cat.get(label, 0) + 1
        if ext[0] <= GRIPPER_STROKE and 0.04 <= ext[-1] <= 0.35:
            grasp_per_cat[label] = grasp_per_cat.get(label, 0) + 1
    manip_capped = sum(min(c, PER_CATEGORY_CAP) for c in per_cat.values())
    grasp_capped = sum(min(c, PER_CATEGORY_CAP) for c in grasp_per_cat.values())
    return {
        "manip": sum(per_cat.values()),
        "manip_capped": manip_capped,
        "graspable": sum(grasp_per_cat.values()),
        "graspable_capped": grasp_capped,
        "diversity": len(per_cat),
        "workspace": workspace,
        "categories": dict(sorted(per_cat.items(), key=lambda kv: -kv[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannetpp-root", default="/data/ScanNetpp")
    ap.add_argument("--splats-meta", default="/data/ScanNetppv2_gsplat/meta.csv")
    ap.add_argument("--exclude-split", default="/data/ScanNetpp/splits/nvs_sem_val.txt")
    ap.add_argument("--min-psnr", type=float, default=26.0)
    ap.add_argument("--min-graspable", type=int, default=3)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    psnr = {}
    with open(args.splats_meta) as f:
        for row in csv.DictReader(f):
            psnr[row["scene"]] = float(row["psnr"])
    exclude = set()
    p = Path(args.exclude_split)
    if p.exists():
        exclude = {s.strip() for s in p.read_text().split() if s.strip()}

    rows, skipped = [], {"no_splat": 0, "no_anno": 0, "benchmark_split": 0,
                         "low_psnr": 0, "few_graspable": 0}
    for d in sorted(Path(args.scannetpp_root, "data").iterdir()):
        scene = d.name
        anno = d / "scans" / "segments_anno.json"
        if scene not in psnr:
            skipped["no_splat"] += 1
            continue
        if not anno.exists():
            skipped["no_anno"] += 1
            continue
        if scene in exclude:
            skipped["benchmark_split"] += 1
            continue
        if psnr[scene] < args.min_psnr:
            skipped["low_psnr"] += 1
            continue
        s = scene_stats(anno)
        if s["graspable"] < args.min_graspable:
            skipped["few_graspable"] += 1
            continue
        s["scene"] = scene
        s["psnr"] = round(psnr[scene], 2)
        s["score"] = round(
            2.0 * s["graspable_capped"]
            + 0.5 * (s["manip_capped"] - s["graspable_capped"])
            + 1.5 * s["diversity"]
            + 2.0 * s["workspace"]
            + 0.5 * min(max(psnr[scene] - args.min_psnr, 0.0), 8.0), 2)
        rows.append(s)

    rows.sort(key=lambda r: -r["score"])
    top = rows[:args.n]
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / "zeroshot_100_scenes.txt").write_text(
        "\n".join(r["scene"] for r in top) + "\n")
    (out / "zeroshot_100_scenes.json").write_text(json.dumps({
        "criteria": vars(args), "skipped": skipped,
        "eligible": len(rows), "selected": len(top), "scenes": top,
    }, indent=1))

    print(f"eligible {len(rows)}  skipped {skipped}")
    print(f"selected {len(top)} -> {out}/zeroshot_100_scenes.txt")
    if top:
        import statistics as st
        for k in ("graspable", "manip", "diversity", "psnr"):
            v = [r[k] for r in top]
            print(f"  {k:10s} min {min(v):5.1f}  median {st.median(v):5.1f}  max {max(v):5.1f}")
        print(f"  workspace  {sum(r['workspace'] for r in top)}/{len(top)}")
        print("top 10:")
        for r in top[:10]:
            cats = ", ".join(list(r["categories"])[:5])
            print(f"  {r['scene']}  score {r['score']:6.1f}  grasp {r['graspable']:3d} "
                  f"div {r['diversity']:2d}  psnr {r['psnr']:.1f}  [{cats}]")


if __name__ == "__main__":
    main()
