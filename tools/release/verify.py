#!/usr/bin/env python3
"""Verify module entry points, version agreement and the bundled PhiView source."""

import argparse
import hashlib
import json
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
    parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if project["version"] != __version__:
        raise SystemExit("package and CLI versions differ")
    modules = catalog()
    count = 0
    for name, block in modules.items():
        if not (ROOT / "phiroom" / "pipelines" / name / "README.md").is_file():
            raise SystemExit(f"module guide missing: {name}")
        for action in block["actions"]:
            plan(name, action, [], ROOT, {})
            count += 1
    result = {
        "version": __version__,
        "modules": len(modules),
        "actions_checked": count,
        "gpu_execution": "NOT_RUN",
        "bundled_phiview": verify_bundled_source(ROOT / "tools/phiview"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
