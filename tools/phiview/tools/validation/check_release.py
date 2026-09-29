"""Check portable release resources and built distributions; no GPU/model claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile


def check(repo: Path, dist: Path):
    project = tomllib.loads((repo / "pyproject.toml").read_text())
    version = project["project"]["version"]
    resources = repo / "physicalview/resources"
    assert (repo / "configs/default.yaml").read_bytes() == (
        resources / "default.yaml"
    ).read_bytes()
    manifests = sorted((resources / "pipelines").glob("*.json"))
    assert len(manifests) == 10, "Expected ten documented pipeline blocks"
    for path in manifests:
        block = json.loads(path.read_text())
        assert block["id"] == path.stem and block["branch"] == f"block/{path.stem}"
        assert (repo / "pipelines" / path.stem / "README.md").is_file()
        assert (
            block["commands"]
            and block["tools"]
            and block["inputs"]
            and block["outputs"]
        )
        assert all((repo / p).exists() for p in block["implementation"])
    for project_dir in [
        repo,
        *(repo / "envs" / p for p in ("studio", "inference", "generation")),
    ]:
        assert (project_dir / "uv.lock").is_file()
        assert "pypi.hub." not in (project_dir / "uv.lock").read_text()
    wheel = dist / f"physicalview-{version}-py3-none-any.whl"
    source = dist / f"physicalview-{version}.tar.gz"
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        required = [
            "physicalview/cli.py",
            "physicalview/paper_download_pack.py",
            "physicalview/resources/default.yaml",
            "physicalview/web/phiview.html",
        ]
        required += [f"physicalview/resources/pipelines/{p.name}" for p in manifests]
        assert all(n in names for n in required), sorted(set(required) - names)
        assert not any(
            set(Path(n).parts) & {"outputs", ".venv", "weights", "backends"}
            for n in names
        )
    with tarfile.open(source) as t:
        names = {
            str(Path(n).relative_to(f"physicalview-{version}"))
            for n in t.getnames()
            if n != f"physicalview-{version}"
        }
        required = [
            "uv.lock",
            "tools/backends/sources.json",
            "tools/env/sync.sh",
            "docs/RELEASE.md",
            "envs/studio/uv.lock",
            "envs/inference/uv.lock",
            "envs/generation/uv.lock",
        ]
        assert all(n in names for n in required), sorted(set(required) - names)
        assert not any(
            set(Path(n).parts) & {"outputs", ".venv", "weights"} for n in names
        )
    return {
        "version": version,
        "blocks": len(manifests),
        "lockfiles": 4,
        "wheel_bytes": wheel.stat().st_size,
        "sdist_bytes": source.stat().st_size,
        "package_resources": "passed",
        "boundary": "Packaging only; no model or paper-quality validation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.dist), indent=2))
