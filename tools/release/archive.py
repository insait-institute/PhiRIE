#!/usr/bin/env python3
"""Build an exact Git source ZIP including bundled or submodule PhiView files."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]


def git(*argv, cwd=ROOT):
    return subprocess.check_output(["git", *argv], cwd=cwd, text=True).strip()


def build(ref, out, prefix):
    sha = git("rev-parse", "--verify", ref + "^{commit}")
    entry = git("ls-tree", sha, "integrations/phiview").split()
    is_submodule = entry[:2] == ["160000", "commit"]
    if is_submodule:
        child = entry[2]
    elif entry[:2] == ["040000", "tree"]:
        manifest = json.loads(
            git("show", sha + ":integrations/phiview/SOURCE_PROVENANCE.json")
        )
        child = manifest["commit"]
    else:
        raise ValueError("reference has no bundled or submodule PhiView source")
    if not prefix or "/" in prefix or "\\" in prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be one directory name")
    out = Path(out).absolute()
    if out.exists():
        raise ValueError(f"archive already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=out.parent) as tmp:
        parent_zip, child_zip = Path(tmp) / "parent.zip", Path(tmp) / "child.zip"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=zip",
                f"--prefix={prefix}/",
                "-o",
                str(parent_zip),
                sha,
            ],
            cwd=ROOT,
            check=True,
        )
        if is_submodule:
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=zip",
                    f"--prefix={prefix}/integrations/phiview/",
                    "-o",
                    str(child_zip),
                    child,
                ],
                cwd=ROOT / "integrations/phiview",
                check=True,
            )
            with (
                zipfile.ZipFile(parent_zip, "a") as archive,
                zipfile.ZipFile(child_zip) as viewer,
            ):
                names = set(archive.namelist())
                for info in viewer.infolist():
                    if info.filename in names:
                        if info.is_dir():
                            continue
                        raise ValueError(f"duplicate file: {info.filename}")
                    archive.writestr(info, viewer.read(info))
                    names.add(info.filename)
        with zipfile.ZipFile(parent_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            for path in (
                "pyproject.toml",
                "envs/control/uv.lock",
                "phiroom/cli.py",
                "integrations/phiview/physicalview/web/phiview.html",
                "integrations/phiview/uv.lock",
            ):
                archive.getinfo(prefix + "/" + path)
            files = sum(not i.is_dir() for i in archive.infolist())
        parent_zip.replace(out)
    with out.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "parent_commit": sha,
        "phiview_commit": child,
        "file": out.name,
        "files": files,
        "bytes": out.stat().st_size,
        "sha256": digest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prefix", default="PhiRIE-v2.0.1")
    args = parser.parse_args()
    print(json.dumps(build(args.ref, args.out, args.prefix), indent=2))
