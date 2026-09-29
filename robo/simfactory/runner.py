"""SimFactory runner: parse a YAML config, resolve one backend per block,
execute the blocks in pipeline order with the SIMANY_* environment the
underlying modules expect.

The runner deliberately does NOT reimplement any pipeline logic: backends
shell out to the same `python -m <module>` invocations the run/*.sh
launchers use (same envs, same knobs), so a SimFactory run and a shell run
are byte-equivalent. What SimFactory adds is selection (per-block baselines),
a single config file per experiment, and a uniform dry-run/plan view.

Interpreters are resolved exactly like run/env.sh does; keep the two in sync.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import yaml

from robo.simfactory.registry import BLOCKS, Context, get

ROOT = Path(__file__).resolve().parents[2]

# interpreter map, mirroring run/env.sh (env overrides win)
PY = {
    "venv": os.environ.get("SIMANY_PY", str(ROOT / ".venv/bin/python")),
    "sam3": os.environ.get("SIMANY_SAM3_PY", str(ROOT / ".envs/sam3/bin/python")),
    "gsplat": os.environ.get("SIMANY_GSPLAT_PY",
                             str(ROOT / ".envs/mini-viewer/bin/python")),
    "sam3d": os.environ.get("SIMANY_SAM3D_PY",
                            str(ROOT / ".envs/sam3d-objects/bin/python")),
    "trellis2": os.environ.get("SIMANY_TRELLIS2_PY",
                               str(ROOT / ".envs/trellis2/bin/python")),
    "h5": os.environ.get("SIMANY_H5_PY", str(ROOT / ".venv/bin/python")),
}


def sh(ctx: Context, module: str, *args, env_kind: str = "venv",
       extra_env: dict | None = None):
    """Run `python -m module args...` from the repo root in the right env."""
    cmd = [PY[env_kind], "-m", module, *map(str, args)]
    env = os.environ.copy()
    env.update({
        "SIMANY_SCENE": ctx.scene,
        "SIMANY_OUT": str(ctx.out),
        "SIMANY_SCANNETPP_ROOT": str(ctx.scannetpp_root),
        "SIMANY_SPLATS_ROOT": str(ctx.splats_root),
        "PYTHONNOUSERSITE": "1",
    })
    env.update(extra_env or {})
    line = " ".join(cmd)
    print(f"[simfactory]$ {line}", flush=True)
    if ctx.dry_run:
        return
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[simfactory] FAILED ({r.returncode}): {line}")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("scene", "blocks"):
        if key not in cfg:
            raise SystemExit(f"[simfactory] config missing top-level '{key}'")
    return cfg


def build_context(cfg: dict, dry_run: bool = False) -> Context:
    sc = cfg["scene"]
    source = sc.get("source", "scannetpp")
    scene = sc["name"]
    if source == "scannetpp":
        scannetpp_root = Path(sc.get("scannetpp_root", "/data/ScanNetpp"))
        splats_root = Path(sc.get("splats_root",
                                  "/data/ScanNetppv2_gsplat/splats"))
    else:
        recon_root = ROOT / "data" / "recon_scenes"
        scannetpp_root = recon_root
        splats_root = recon_root / "splats"
    out = Path(sc.get("out", ROOT / "outputs" /
                      (scene if source == "scannetpp" else f"{source}_{scene}"
                       if not scene.startswith(source) else scene)))
    return Context(root=ROOT, scene=scene, source=source, config=cfg,
                   out=out, scene_dir=scannetpp_root / "data" / scene,
                   splats_root=splats_root, scannetpp_root=scannetpp_root,
                   options=cfg.get("options", {}) or {}, dry_run=dry_run)


def run_config(path: Path, dry_run: bool = False,
               only: list[str] | None = None):
    from robo.simfactory import blocks  # noqa: F401

    cfg = load_config(path)
    ctx = build_context(cfg, dry_run)
    ctx.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for block in BLOCKS:
        if only and block not in only:
            continue
        sel = cfg["blocks"].get(block)
        if not sel or sel in ("none", "skip"):
            continue
        # generation may carry a dict {method, candidates}; others are names
        name = sel["method"] if isinstance(sel, dict) else \
            sel[0] if isinstance(sel, list) else sel
        entries = sel if isinstance(sel, list) else [sel]
        for entry in entries:
            name = entry["method"] if isinstance(entry, dict) else entry
            be = get(block, name)
            if be.available:
                reason = be.available(ctx)
                if reason:
                    raise SystemExit(f"[simfactory] {block}/{name} "
                                     f"unavailable: {reason}")
            print(f"\n=== [{time.strftime('%H:%M:%S')}] {block} :: {name} "
                  f"===", flush=True)
            be.run(ctx, entry if isinstance(entry, dict) else {})
    print(f"\n[simfactory] done in {(time.time() - t0) / 60:.1f} min "
          f"-> {ctx.out}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="simfactory", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="execute a config")
    runp.add_argument("config", type=Path)
    runp.add_argument("--dry-run", action="store_true",
                      help="print the exact commands without executing")
    runp.add_argument("--only", nargs="*", help="run only these blocks")
    sub.add_parser("list", help="list every block and registered baseline")

    args = ap.parse_args()
    if args.cmd == "list":
        from robo.simfactory import blocks  # noqa: F401
        from robo.simfactory.registry import catalog
        for block, entries in catalog().items():
            print(f"{block}:")
            for name, be in sorted(entries.items()):
                print(f"  {name:22s} {be.help}")
        return
    run_config(args.config, dry_run=args.dry_run, only=args.only)


if __name__ == "__main__":
    main()
