"""Baseline: SimFoundry-reproduction, made callable against ANY scene layout.

`run/run_simfoundry.sh` already implements this baseline faithfully (one
representative frame, monocular metric depth, single-view TRELLIS, no GT,
no hybrid slot -- the paper's ablation row D, arXiv:2606.28276). By design
it has "no code of its own" (see agents/baselines/README.md): it is the
degenerate configuration of the SAME pipeline stages the main method runs,
which is what keeps the construction-paper's ablation ladder a controlled
same-codebase comparison rather than a cross-codebase one. This module does
NOT change that design -- it only makes the same degenerate configuration
callable programmatically against ANY ScanNet++-layout scene dir, not just
via the shell script.

IMPORTANT SCOPE NOTE (corrected 2026-08-30, same day this file was added):
this is admissible as a baseline in the `mujoco_paired` predictive track
ONLY, not `oracle_causal`. Two of the shared stages this glue calls --
`agents.discover.s0_select_frame` and `agents.assets.s5_align` -- hard-
depend on `agents.core.common.load_gt_instances()`, which only resolves
ScanNet++'s `segments.json`/`segments_anno.json`. On BEHAVIOR/oracle scenes
that either crashes outright or, worse, would leak hidden oracle GT into
the baseline's own frame selection if a same-shaped file ever existed
there -- exactly the leak the oracle track's GT-inaccessibility guarantee
(`oracle/gt_export.py`) exists to prevent. `resolve_scene_env()` below still
generalizes the scene-dir *path* handling to the `recon_scenes` layout
(harmless and correct on its own), but do not actually invoke `run_pipeline`
against a BEHAVIOR/DROID scene until s0/s5 get a genuine GT-free mode --
that is tracked as follow-up work, not done here.

This is still glue, not a reimplementation: every stage below is a
subprocess call to the exact module `run/run_simfoundry.sh` calls, in the
same order, with the same flags. See `baselines/release_status.yaml`
(status: partial -- no official SimFoundry code exists to audit this
against) and `docs/BASELINE_REPRODUCTION.md`.

Usage:
  .venv/bin/python -m baselines.simfoundry_repro \
    --scene-dir data/recon_scenes/data/behavior_task0011 --out-dir outputs/behavior_task0011_baselines/simfoundry_repro

  # inspect the resolved configuration/manifest without running the GPU pipeline:
  .venv/bin/python -m baselines.simfoundry_repro --scene-dir <dir> --out-dir <dir> --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Same stage sequence as run/run_simfoundry.sh, same order, same per-stage
# interpreter (run/env.sh's run() vs run_sam3() vs run_gs() distinction --
# SAM3 needs its own torch 2.10 env, gsplat render needs the cp310 wheel
# env; every other stage runs in the main .venv). Each entry is
# (stage_label, module, interpreter_env_key, extra_args_fn), where
# extra_args_fn(scene_out_dir) -> list[str] is resolved AFTER the previous
# stage has actually run (mirrors run_simfoundry.sh's `$FRAME=$(...)`
# read-back between s0 and s1 -- s1 needs the exact frame filename s0
# picked, not a guess).
_SAME_PROMPTS = ["bottle", "mug", "computer mouse", "keyboard", "headphones",
                 "telephone", "box"]


def _s1_extra_args(out_dir: Path) -> list[str]:
    rep = json.loads((out_dir / "frame" / "rep_frame.json").read_text())
    frame = rep["frame"]
    return ["--image", str(out_dir / "frame" / frame),
            "--out-dir", str(out_dir / "masks"),
            "--prompts", *_SAME_PROMPTS]


STAGES = [
    ("s0 select representative frame", "agents.discover.s0_select_frame",
     "SIMANY_PY", None),
    ("s1 SAM3 segmentation on the representative frame", "models.s1_segment",
     "SIMANY_SAM3_PY", _s1_extra_args),
    ("s2 DA3 metric depth", "models.s2_depth", "SIMANY_PY", None),
    ("s3 lift objects", "agents.discover.s3_lift", "SIMANY_PY", None),
    ("s4 TRELLIS single-view image-to-3D (no hybrid slot)", "models.s4_trellis",
     "SIMANY_PY", None),
    ("s5 pose alignment + F1 eval", "agents.assets.s5_align", "SIMANY_PY", None),
    ("s6 CoACD + physics annotation + URDF", "agents.assets.s6_physics",
     "SIMANY_PY", None),
    ("s7 PyBullet settle + dynamics", "robo.sim.s7_sim", "SIMANY_PY", None),
]
OPTIONAL_RENDER_STAGE = ("s8 gsplat renders", "agents.render.s8_render",
                         "SIMANY_GSPLAT_PY", None)

# Defaults mirror run/env.sh exactly -- kept here too since this module
# invokes stages directly via subprocess, not through env.sh's run()/
# run_sam3()/run_gs() bash functions.
_INTERPRETER_DEFAULTS = {
    "SIMANY_PY": str(ROOT / ".venv" / "bin" / "python"),
    "SIMANY_SAM3_PY": str(ROOT / ".envs" / "sam3" / "bin" / "python"),
    "SIMANY_GSPLAT_PY": str(ROOT / ".envs" / "mini-viewer" / "bin" / "python"),
}

# Same honesty contract as raw_reconstruction.py: this ablation's config
# (one frame, single-view, no GT, no hybrid) cannot produce a multi-view
# discovery result or a hybrid-generation comparison by construction --
# report N/A with a reason, never a fabricated number.
NA_METRICS = {
    "multi_view_discovery_f1":
        "this configuration segments exactly one representative frame "
        "(s0_select_frame); no multi-view voxel-IoU merging is performed",
    "hybrid_gate_decision":
        "single-view TRELLIS only, no hybrid single/multi-view generator "
        "gate is exercised in this configuration",
    "gt_matched_recall":
        "no ground-truth instance labels are consulted by this ablation "
        "(matches the paper's zero-GT ablation row D, not the GT-driven rows)",
}


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def resolve_scene_env(scene_dir: Path, out_dir: Path) -> dict:
    """Build the env-var overrides run/env.sh needs to point the shared
    pipeline stages at an arbitrary scene dir, honouring both layouts:
      ScanNet++:     <SCANNETPP_ROOT>/data/<scene_id>/...
      recon_scenes:  data/recon_scenes/data/<scene_id>/... (oracle/DROID tracks)
    Never guesses silently: if the scene_dir doesn't look like either known
    layout, still proceeds (env.sh's SCANNETPP_ROOT override covers any
    "<root>/data/<scene_id>" tree) but records which layout was assumed.
    """
    scene_dir = Path(scene_dir).resolve()
    scene_id = scene_dir.name
    data_root = scene_dir.parent.parent  # <root>/data/<scene_id> -> <root>
    layout = ("recon_scenes" if "recon_scenes" in str(data_root)
              else "scannetpp" if str(data_root) == "/data/ScanNetpp"
              else "custom")
    env = dict(os.environ)
    env["SIMANY_SCENE"] = scene_id
    env["SIMANY_SCANNETPP_ROOT"] = str(data_root)
    env["SIMANY_OUT"] = str(Path(out_dir).resolve())
    return {"env": env, "scene_id": scene_id, "layout": layout,
            "data_root": str(data_root)}


def run_pipeline(scene_dir: Path, out_dir: Path, include_render: bool = False,
                  timeout_s: int = 3600) -> dict:
    """Actually invoke the degenerate SimFoundry-repro stage sequence
    against an arbitrary scene dir. Requires a GPU node and the same env
    that run/run_simfoundry.sh needs (run/env.sh sourced per-stage via the
    `run`/`run_sam3`/`run_gs` helpers -- replicated here via subprocess env
    since this module has no bash `run()` wrapper of its own to reuse)."""
    resolved = resolve_scene_env(scene_dir, out_dir)
    env = resolved["env"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_log = []
    stages = STAGES + ([OPTIONAL_RENDER_STAGE] if include_render else [])
    for label, module, interp_key, extra_args_fn in stages:
        interp = env.get(interp_key, _INTERPRETER_DEFAULTS[interp_key])
        try:
            extra = extra_args_fn(out_dir) if extra_args_fn else []
        except Exception as exc:
            stage_log.append({
                "stage": label, "module": module, "seconds": 0.0,
                "returncode": -1,
                "stderr_tail": f"failed to resolve stage args: {type(exc).__name__}: {exc}",
            })
            break
        t0 = time.time()
        proc = subprocess.run([interp, "-m", module] + extra, cwd=ROOT,
                               env=env, capture_output=True, text=True,
                               timeout=timeout_s)
        stage_log.append({
            "stage": label, "module": module, "interpreter": interp,
            "seconds": round(time.time() - t0, 1),
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else None,
        })
        if proc.returncode != 0:
            break  # a failed stage is coverage loss, not a resampled easier trial
    return {"resolved": resolved, "stage_log": stage_log}


def build_manifest(scene_dir, out_dir, stage_log=None, dry_run=False) -> dict:
    resolved = resolve_scene_env(scene_dir, out_dir)
    all_ok = bool(stage_log) and all(s["returncode"] == 0 for s in stage_log)
    manifest = {
        "baseline": "simfoundry_repro",
        "baseline_kind": "in_repo_ablation_configuration",  # not a 3rd-party release
        "config": {
            "frames": 1, "view": "single", "gt_used": False,
            "hybrid_generation": False,
            "matches": "construction paper ablation row D (docs/BASELINES.md)",
        },
        "scene_id": resolved["scene_id"],
        "scene_layout": resolved["layout"],
        "git_commit": _git_commit(),
        "generated_at_utc": None,  # stamped in main() after the pipeline actually runs
        "dry_run": dry_run,
        "build_success": None if dry_run else all_ok,
        "stage_log": stage_log or [],
        "na_metrics": {k: {"value": None, "status": "not_applicable", "reason": v}
                        for k, v in NA_METRICS.items()},
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-render", action="store_true",
                     help="also run the optional s8 gsplat render stage")
    ap.add_argument("--dry-run", action="store_true",
                     help="resolve config/env and print the manifest without "
                          "invoking any GPU stage -- for CI/unit testing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = build_manifest(args.scene_dir, args.out_dir, dry_run=True)
    else:
        result = run_pipeline(Path(args.scene_dir), out_dir, args.include_render)
        manifest = build_manifest(args.scene_dir, args.out_dir,
                                   stage_log=result["stage_log"])
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    report_path = out_dir / "simfoundry_repro_manifest.json"
    report_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if manifest["build_success"] is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
