"""Plan or create a draft GitHub release from a clean, tagged main checkout.

Build and validate dist/ first. No credentials are read until --create is supplied.
Existing published releases or differing assets are preserved and cause an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import urllib.parse
import urllib.request


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", default="RunyiYang/PhysicalView")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if args.tag != "v" + version:
        raise SystemExit("Tag must match pyproject.toml version")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Preserve your edits; release from a clean checkout")
    head = git("rev-parse", "HEAD")
    if git("rev-parse", args.tag + "^{commit}") != head:
        raise SystemExit("HEAD must be the tagged commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "origin/main"], check=True
    )
    assets = [
        args.dist / f"physicalview-{version}-py3-none-any.whl",
        args.dist / f"physicalview-{version}.tar.gz",
    ]
    sums = args.dist / "SHA256SUMS"
    sums.write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in assets
        )
    )
    assets.append(sums)
    plan = {
        "repository": args.repo,
        "tag": args.tag,
        "commit": head,
        "draft": True,
        "assets": [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in assets
        ],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.create:
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            text=True,
            capture_output=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        fields = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        token = fields.get("password") if result.returncode == 0 else None
    if not token:
        raise SystemExit("Configure GH_TOKEN or a GitHub HTTPS credential helper")

    def request(url, method="GET", payload=None, content_type="application/json"):
        data = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode()
            if payload is not None
            else None
        )
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "Content-Type": content_type,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    base = "https://api.github.com/repos/" + args.repo
    # Paginate so a retry never accidentally duplicates an older draft.
    release = None
    page = 1
    while True:
        items = request(base + f"/releases?per_page=100&page={page}")
        release = next((r for r in items if r["tag_name"] == args.tag), None)
        if release or len(items) < 100:
            break
        page += 1
    if release and not release["draft"]:
        raise SystemExit("Existing published release preserved")
    if not release:
        release = request(
            base + "/releases",
            "POST",
            {
                "tag_name": args.tag,
                "target_commitish": head,
                "name": "PhysicalView " + args.tag,
                "body": Path("CHANGELOG.md").read_text(),
                "draft": True,
                "prerelease": False,
            },
        )
    existing = {
        a["name"]: a
        for a in request(base + f"/releases/{release['id']}/assets?per_page=100")
    }
    for path, receipt in zip(assets, plan["assets"]):
        if path.name in existing:
            if existing[path.name].get("digest") != "sha256:" + receipt["sha256"]:
                raise SystemExit(
                    f"Existing asset differs or has no verifiable digest: {path.name}; preserved"
                )
            continue
        upload = (
            release["upload_url"].split("{")[0]
            + "?name="
            + urllib.parse.quote(path.name)
        )
        request(upload, "POST", path.read_bytes(), "application/octet-stream")
    print(
        json.dumps(
            {
                "draft_release_url": release["html_url"],
                "id": release["id"],
                "draft": True,
            }
        )
    )


if __name__ == "__main__":
    main()
