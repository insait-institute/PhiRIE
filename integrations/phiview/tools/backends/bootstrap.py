"""Fetch pinned source repositories without overwriting an existing checkout."""

import argparse, json, subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sources", nargs="+", choices=["simany", "trellis", "menagerie", "openpi"]
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for name in args.sources:
        spec = sources[name]
        dest = repo / spec["destination"]
        if dest.exists():
            head = subprocess.check_output(
                ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(dest),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            )
            if head != spec["revision"] or dirty:
                raise SystemExit(
                    f"{dest}: existing checkout differs or has edits; preserved. Use a separate checkout or configure SIMANY_ROOT."
                )
            print(f"{name}: existing pinned checkout verified")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["url"], str(dest)], check=True
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "--detach", spec["revision"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
        print(f"{name}: {spec['revision']}")


if __name__ == "__main__":
    main()
