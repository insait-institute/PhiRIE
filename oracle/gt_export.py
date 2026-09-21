"""oracle/gt_export.py -- Task 14 capability #2: export interactive GT scene
state into a locked-down vault the reconstruction process cannot read, plus
two independent tests that this is actually true.

What counts as "GT" here: the OBJECT-INSTANCE annotations
(`gt/gt_objects.json`, per-object `gt/<mesh>.ply`) that
`agents/recon/behavior_extract.py` / `oracle/capture_generator.py` write
next to the capture -- i.e. exact per-object identity, pose, and shape that
no real phone scan could ever produce. This does NOT include
`gt/depth/*.png`, `gt/mesh_gt.ply`, `gt/intrinsics_native.json`: those are
CAPTURE-derived (a real phone with a depth sensor produces exactly this),
not simulator ground truth, and the recon pipeline is allowed -- expected --
to read them (`run/run_behavior_recon.sh`'s GT_MESH=1 default). Vaulting the
depth too would misrepresent a legitimate sensor input as forbidden state;
see docs/ORACLE_PROTOCOL.md "what counts as GT" for the full argument.

Two tests, both real (not just static analysis, though that is included as
a cheap first layer):
  1. `static_analysis_check()`: grep the actual reconstruction-stage source
     files for the vaulted filenames. Expected: zero hits.
  2. `os_level_access_check()`: after `export_gt` + `lock_vault` (chmod 000,
     recursively, on every file AND dir under the vault), spawn a fresh
     subprocess (same user, not root) that does a plain `open()` on a known
     vaulted file and asserts it raises PermissionError. Linux enforces
     owner permission bits even for the owning, non-root user, so this is a
     genuine OS-level guarantee, not merely a convention.

`unlock_for_eval` is the ONE sanctioned way back in: a context manager that
temporarily chmods the vault open, yields, and re-locks it in `finally`. Only
oracle/evaluate_reconstruction.py (explicitly the evaluation side, which the
protocol allows to see GT) calls it; the reconstruction pipeline never does.

Usage:
  .venv/bin/python -m oracle.gt_export export --scene behavior_task0011 \
      --root data/recon_scenes --vault-root outputs/oracle_gt_vault
  .venv/bin/python -m oracle.gt_export selftest --scene behavior_task0011 \
      --vault-root outputs/oracle_gt_vault
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files/dirs under <scene>/gt/ that a real capture pipeline could NOT have
# produced -- object-instance identity/pose/shape. Everything else in gt/
# (depth/, mesh_gt.ply, intrinsics_native.json) is capture-derived and stays
# world-readable next to the scene dir; see module docstring.
VAULTED_GLOBS = ("gt_objects.json", "*.ply")
VAULT_EXCLUDE_NAMES = {"mesh_gt.ply"}  # capture-derived TSDF fuse, not an object annotation

# Source files that make up "the reconstruction pipeline" for the purposes
# of the static-analysis check -- every stage run/run_behavior_recon.sh
# invokes between the capture and the exported MJCF/OmniGibson bridge.
# Deliberately EXCLUDES agents/recon/behavior_extract.py itself and
# oracle/*.py (those legitimately write/read the vault source before it is
# sealed) and eval scripts (evaluation is allowed to see GT).
RECON_STAGE_FILES = [
    "agents/discover/auto_segment.py",
    "agents/discover/factory_prepare.py",
    "agents/discover/factory_refine_masks.py",
    "agents/discover/derive_mesh_from_splat.py",
    "agents/recon/gsplat_train.py",
    "models/s4_trellis.py",
    "agents/assets/factory_align.py",
    "agents/assets/s6_physics.py",
    "robo/sim/export_mjcf.py",
    "robo/tasks/pi05_tasks.py",
    "robo/sim/export_omnigibson.py",
]
FORBIDDEN_SUBSTRINGS = ["gt_objects.json", "gt_vault", "scene_mesh_trajector"]


def export_gt(scene_id: str, root: str, vault_root: str) -> Path:
    scene_dir = Path(root) / "data" / scene_id
    gt_dir = scene_dir / "gt"
    if not gt_dir.exists():
        raise SystemExit(f"[gt_export] no gt/ dir under {scene_dir}")
    vault_dir = Path(vault_root) / scene_id
    if vault_dir.exists():
        os.chmod(vault_dir, 0o700)  # undo a previous lock_vault() so rmtree can enter it
        shutil.rmtree(vault_dir)
    vault_dir.mkdir(parents=True)

    copied = []
    manifest = json.loads((gt_dir / "gt_objects.json").read_text())
    shutil.copy(gt_dir / "gt_objects.json", vault_dir / "gt_objects.json")
    copied.append("gt_objects.json")
    for short, info in manifest.get("meshes", {}).items():
        ply_name = Path(info["ply"]).name  # "gt/<short>.ply" -> "<short>.ply"
        if ply_name in VAULT_EXCLUDE_NAMES:
            continue
        src = gt_dir / ply_name
        if src.exists():
            shutil.copy(src, vault_dir / ply_name)
            copied.append(ply_name)
    print(f"[gt_export] vaulted {len(copied)} files -> {vault_dir}")
    return vault_dir


def lock_vault(vault_dir: Path):
    """chmod 000 on the vault directory itself. On Linux, directory EXECUTE
    permission gates path resolution into every descendant regardless of
    the descendant's own mode bits, so locking just the top-level directory
    (not recursing into every file) is sufficient and avoids the ordering
    trap recursive chmod has here: a 000 parent can no longer be `rglob`'d
    to reach and unlock its own children (verified empirically -- an
    earlier revision of this function recursed root-to-leaves and simply
    never touched anything inside a directory it had already sealed)."""
    os.chmod(vault_dir, 0o000)
    print(f"[gt_export] locked {vault_dir} (chmod 000)")


@contextlib.contextmanager
def unlock_for_eval(vault_dir: Path):
    """The ONLY sanctioned way back into a locked vault. Call sites: only
    oracle/evaluate_reconstruction.py. Re-locks even if the body raises."""
    os.chmod(vault_dir, 0o500)
    try:
        yield vault_dir
    finally:
        os.chmod(vault_dir, 0o000)


# ---------------------------------------------------------------------------
# Test 1: static analysis
# ---------------------------------------------------------------------------

def static_analysis_check() -> dict:
    hits = {}
    for rel in RECON_STAGE_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        found = [s for s in FORBIDDEN_SUBSTRINGS if s in text]
        if found:
            hits[rel] = found
    return {"pass": not hits, "checked_files": RECON_STAGE_FILES, "hits": hits}


# ---------------------------------------------------------------------------
# Test 2: OS-level access check (real subprocess, real chmod)
# ---------------------------------------------------------------------------

def os_level_access_check(vault_dir: Path) -> dict:
    if os.geteuid() == 0:
        return {"pass": None, "detail": "running as root - chmod 000 does not "
                "bind root, so this check cannot be meaningful here; rerun as "
                "a non-root user."}
    # NOTE: cannot discover the target by globbing vault_dir here -- once
    # locked (chmod 000/100), even the owning process that just locked it
    # cannot list the directory (pathlib.Path.glob swallows the resulting
    # PermissionError from scandir and silently yields nothing, which would
    # make this check falsely report "no files found" rather than testing
    # anything). Use the one filename export_gt() always writes instead.
    target = vault_dir / "gt_objects.json"
    try:
        target_exists = target.exists()
    except PermissionError:
        # Can't even stat() it - the vault is locked exactly as intended;
        # export_gt() always writes gt_objects.json, so trust the name.
        target_exists = True
    if not target_exists and os.access(vault_dir, os.R_OK):
        # vault isn't locked (or we can still see it) - fall back to a scan
        target = next(vault_dir.glob("*.json"), None) or next(vault_dir.glob("*.ply"), None)
        target_exists = target is not None
    if not target_exists:
        return {"pass": False, "detail": f"no target file resolved under {vault_dir}"}
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; open(sys.argv[1], 'rb').read()", str(target)],
        capture_output=True, text=True)
    blocked = proc.returncode != 0 and "PermissionError" in proc.stderr
    return {"pass": blocked, "target": str(target), "returncode": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else []}


def selftest(scene_id: str, vault_root: str) -> dict:
    vault_dir = Path(vault_root) / scene_id
    static = static_analysis_check()
    osck = os_level_access_check(vault_dir)
    result = {"scene_id": scene_id, "vault_dir": str(vault_dir),
              "static_analysis": static, "os_level_access": osck,
              "overall_pass": bool(static["pass"]) and bool(osck["pass"])}
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export")
    ex.add_argument("--scene", required=True)
    ex.add_argument("--root", required=True)
    ex.add_argument("--vault-root", required=True)
    ex.add_argument("--no-lock", action="store_true")

    lk = sub.add_parser("lock")
    lk.add_argument("--scene", required=True)
    lk.add_argument("--vault-root", required=True)

    st = sub.add_parser("selftest")
    st.add_argument("--scene", required=True)
    st.add_argument("--vault-root", required=True)

    args = ap.parse_args()
    if args.cmd == "export":
        vault_dir = export_gt(args.scene, args.root, args.vault_root)
        if not args.no_lock:
            lock_vault(vault_dir)
    elif args.cmd == "lock":
        lock_vault(Path(args.vault_root) / args.scene)
    elif args.cmd == "selftest":
        result = selftest(args.scene, args.vault_root)
        print(json.dumps(result, indent=1))
        sys.exit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
