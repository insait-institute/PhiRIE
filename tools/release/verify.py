#!/usr/bin/env python3
"""Verify module entry points, versions, and bundled or submodule PhiView source."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

from phiroom import __version__
from phiroom.core.execution import plan
from phiroom.core.registry import catalog


def verify_bundled_source(directory):
    """Check every upstream file against the recorded immutable source snapshot."""
    manifest = json.loads((directory / "SOURCE_PROVENANCE.json").read_text())
    files = manifest["files"]
    if not files:
        raise ValueError("bundled source manifest is empty")
    for name, expected in files.items():
        file = directory / name
        if not file.resolve().is_relative_to(directory.resolve()) or file.is_symlink():
            raise ValueError(f"invalid bundled source path: {name}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"bundled source checksum mismatch: {name}")
    return {"status": "passed", "commit": manifest["commit"], "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-submodule",
        action="store_true",
        help="parent-only CI; explicitly leaves submodule validation NOT_RUN",
    )
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if project["version"] != __version__:
        raise SystemExit("package and CLI versions differ")
    modules = catalog()
    bundled = (ROOT / "integrations/phiview/SOURCE_PROVENANCE.json").is_file()
    count = 0
    for name, block in modules.items():
        if not (ROOT / "pipelines" / name / "README.md").is_file():
            raise SystemExit(f"module guide missing: {name}")
        for action, spec in block["actions"].items():
            if args.skip_submodule and not bundled and spec.get("kind") == "phiview":
                continue
            plan(name, action, [], ROOT, {})
            count += 1
    result = {
        "version": __version__,
        "modules": len(modules),
        "actions_checked": count,
        "submodule": "NOT_RUN",
        "gpu_execution": "NOT_RUN",
    }
    if bundled:
        result["bundled_phiview"] = verify_bundled_source(ROOT / "integrations/phiview")
        result["submodule"] = "NOT_APPLICABLE: PhiView source is bundled"
    elif not args.skip_submodule:

        def git(*argv, cwd=ROOT):
            return subprocess.check_output(["git", *argv], cwd=cwd, text=True).strip()

        entry = git("ls-tree", "HEAD", "integrations/phiview").split()
        actual = git("rev-parse", "HEAD", cwd=ROOT / "integrations/phiview")
        if entry[:2] != ["160000", "commit"] or entry[2] != actual:
            raise SystemExit("PhiView is not checked out at the committed gitlink")
        if git("status", "--porcelain", cwd=ROOT / "integrations/phiview"):
            raise SystemExit("PhiView checkout is dirty")
        result["submodule"] = {"status": "passed", "commit": actual}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
