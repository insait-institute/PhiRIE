"""Aggregate the 50-scene validation fleet into the paper-style tables.

Reads every outputs/<scene>_factory/{report.json, render_metrics.json,
timings.txt} and emits outputs/val_summary.json + a markdown table on
stdout: render quality (PSNR/SSIM/LPIPS, ours-composite vs SceneSplat
background), physical plausibility (tier yield, drop-test stability),
efficiency (per-stage GPU time), resources.

render_metrics_v2.json (official-split eval) is preferred over
render_metrics.json when present; scenes missing required files are
counted in scenes_skipped rather than silently dropped.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "outputs"
H200_SPEEDUP = 2.5  # assumed A6000->H200 factor; overridden by bench if present


def main():
    rows, skipped, rm_src = [], [], {}
    f1_20_pool, f1_40_pool = [], []
    for d in sorted(ROOT.glob("*_factory")):
        if d.name.startswith("h200_bench"):
            continue
        scene = d.name.replace("_factory", "")
        rep = d / "report.json"
        rm = d / "render_metrics_v2.json"  # official-split eval, preferred
        if not rm.exists():
            rm = d / "render_metrics.json"
        if not (rep.exists() and rm.exists()):
            skipped.append(scene)
            continue
        rm_src[scene] = rm.name
        r = json.loads(rep.read_text())
        m = json.loads(rm.read_text())
        tim = {}
        if (d / "timings.txt").exists():
            for ln in (d / "timings.txt").read_text().splitlines():
                parts = ln.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    # SUM every occurrence per stage: total GPU-seconds
                    # actually spent, including partial passes re-logged
                    # after RESUME=1 (last-wins let a cheap cached rerun
                    # overwrite the real first-pass time)
                    k = parts[0].split(" (")[0]
                    tim[k] = tim.get(k, 0) + int(parts[1])
        stable = [o["drop_test"]["stable"] for o in r["objects"]
                  if "drop_test" in o]
        ab = [o for o in r["objects"] if o["tier"] in "AB"]
        f1_20_pool += [o["f1_20"] for o in ab]
        f1_40_pool += [o["f1_40"] for o in ab]
        rows.append({
            "scene": scene,
            "n": r["n_instances"], "A": r["tier_A"], "B": r["tier_B"],
            "C": r["tier_C_rejected"],
            "stable": sum(stable), "tested": len(stable),
            "f1_20": r["mean_f1_20_AB"],
            "bg": m["mean"]["bg"], "twin": m["mean"]["twin"],
            "time_s": sum(tim.values()), "timings": tim,
        })

    n_scenes = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    wmean = lambda f: float(np.mean([f(r) for r in rows]))
    stage_tot = {}
    for r in rows:
        for k, v in r["timings"].items():
            stage_tot[k] = stage_tot.get(k, 0) + v
    total_s = sum(stage_tot.values())

    summary = {
        "n_scenes": n_scenes,
        "scenes_skipped": skipped,
        "render_metrics_source": rm_src,
        "objects": {"total": tot("n"), "tier_A": tot("A"), "tier_B": tot("B"),
                    "rejected": tot("C"),
                    "yield_AB": (tot("A") + tot("B")) / max(tot("n"), 1),
                    # macro: unweighted mean of per-scene means (legacy)
                    "mean_f1at20mm_AB": wmean(lambda r: r["f1_20"]),
                    # pooled: mean over all A/B objects (scene weighted by
                    # its object count, same convention as yield_AB)
                    "f1_20_AB_pooled": float(np.mean(f1_20_pool or [0])),
                    "f1_40_AB_pooled": float(np.mean(f1_40_pool or [0])),
                    "drop_stable": tot("stable"),
                    "drop_tested": tot("tested")},
        "render": {
            "scenesplat_bg": {k: wmean(lambda r, k=k: r["bg"][k])
                              for k in ("psnr", "ssim", "lpips")},
            "ours_twin": {k: wmean(lambda r, k=k: r["twin"][k])
                          for k in ("psnr", "ssim", "lpips")}},
        "efficiency": {
            "total_gpu_seconds_a6000": total_s,
            "a6000_hours": total_s / 3600,
            "h200_hours_est": total_s / 3600 / H200_SPEEDUP,
            "per_scene_minutes_mean": total_s / max(n_scenes, 1) / 60,
            "per_object_seconds_mean": total_s / max(tot("n"), 1),
            "stage_seconds_total": stage_tot},
        "per_scene": rows,
    }
    (ROOT / "val_summary.json").write_text(json.dumps(summary, indent=1))

    if skipped:
        print(f"!!! WARNING: {len(skipped)} scene(s) skipped — missing "
              f"report.json/render_metrics(.v2).json: {', '.join(skipped)}")
    o, re_, e = summary["objects"], summary["render"], summary["efficiency"]
    print(f"scenes: {n_scenes}  objects: {o['total']} "
          f"(A {o['tier_A']} / B {o['tier_B']} / rej {o['rejected']}, "
          f"yield {o['yield_AB']:.0%}, F1@20mm macro {o['mean_f1at20mm_AB']:.3f} "
          f"/ pooled {o['f1_20_AB_pooled']:.3f})")
    print(f"drop-test stable: {o['drop_stable']}/{o['drop_tested']}")
    print(f"render  bg   PSNR {re_['scenesplat_bg']['psnr']:.2f} "
          f"SSIM {re_['scenesplat_bg']['ssim']:.3f} "
          f"LPIPS {re_['scenesplat_bg']['lpips']:.3f}")
    print(f"render  twin PSNR {re_['ours_twin']['psnr']:.2f} "
          f"SSIM {re_['ours_twin']['ssim']:.3f} "
          f"LPIPS {re_['ours_twin']['lpips']:.3f}")
    print(f"compute: {e['a6000_hours']:.1f} A6000h "
          f"(≈{e['h200_hours_est']:.1f} H200h at {H200_SPEEDUP}x), "
          f"{e['per_scene_minutes_mean']:.1f} min/scene, "
          f"{e['per_object_seconds_mean']:.0f} s/object")
    for k, v in sorted(stage_tot.items(), key=lambda x: -x[1]):
        print(f"  {k:55s} {v / 3600:5.2f} h")


if __name__ == "__main__":
    main()
