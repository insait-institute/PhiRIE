"""Read/write/diff helpers for robo.manifest schemas.

Fast-validation entrypoint:
    python -m robo.manifest.io validate configs/evaluation/example_manifest.yaml

Frozen-field diff CLI (highlights only the declared frozen fields that
differ between two manifests):
    python -m robo.manifest.io diff <manifest_a> <manifest_b> [--all-fields]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import yaml

from robo.manifest.hash import (
    canonical_hash,
    find_semantic_issues,
    manifest_content_hash,
)
from robo.manifest.schema import (
    FROZEN_ROLLOUT_FIELDS,
    MANIFEST_KIND_TO_CLASS,
    RolloutManifest,
    SceneBuildManifest,
)

ManifestT = Union[SceneBuildManifest, RolloutManifest]

DEFAULT_MANIFEST_FILENAME = "manifest.json"


class ManifestConflictError(RuntimeError):
    """Raised by write_manifest when an existing manifest at the target path
    would be silently clobbered by different content."""


class ManifestKindError(ValueError):
    """Raised when a dict has a missing/unknown `manifest_kind`."""


# ---------------------------------------------------------------- parsing --

def manifest_from_dict(data: dict) -> ManifestT:
    kind = data.get("manifest_kind")
    cls = MANIFEST_KIND_TO_CLASS.get(kind)
    if cls is None:
        raise ManifestKindError(
            f"unknown or missing manifest_kind {kind!r}; expected one of "
            f"{sorted(MANIFEST_KIND_TO_CLASS)}")
    return cls.model_validate(data)


def to_dict(manifest: ManifestT) -> dict:
    return manifest.model_dump(mode="json")


# ------------------------------------------------------------- dump / load --

def dump_json(manifest: ManifestT, path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(manifest), indent=2, sort_keys=True) + "\n")


def dump_yaml(manifest: ManifestT, path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(manifest), sort_keys=True))


def load_json(path: "str | Path") -> ManifestT:
    return manifest_from_dict(json.loads(Path(path).read_text()))


def load_yaml(path: "str | Path") -> ManifestT:
    return manifest_from_dict(yaml.safe_load(Path(path).read_text()))


def load_any(path: "str | Path") -> ManifestT:
    """Dispatch on file suffix; used by the CLI and by write_manifest's
    conflict check so callers never have to know a target's format."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return load_yaml(path)
    return load_json(path)


# ---------------------------------------------------------- write + guard --

def write_manifest(
    dir_path: "str | Path",
    manifest: ManifestT,
    *,
    filename: str = DEFAULT_MANIFEST_FILENAME,
    allow_overwrite_if_same: bool = True,
) -> Path:
    """Write `manifest` to `dir_path/filename`, refusing to clobber a
    DIFFERENT manifest already there (never overwrite a completed output
    directory).

    Semantics of `allow_overwrite_if_same`:
      - An existing manifest with a DIFFERENT content_hash always raises
        ManifestConflictError -- this is not gated by the flag; there is no
        way to force-clobber through this function on purpose.
      - An existing manifest with the SAME content_hash (e.g. a RESUME=1
        rerun that reproduces byte-identical config) is a no-op rewrite
        when `allow_overwrite_if_same=True` (the default). Pass
        `allow_overwrite_if_same=False` for a stricter "first write wins,
        full stop" mode that also rejects an identical rewrite.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / filename
    new_hash = manifest_content_hash(manifest)

    if target.exists():
        existing = load_any(target)
        existing_hash = manifest_content_hash(existing)
        if existing_hash != new_hash:
            diff = diff_manifests(existing, manifest, frozen_only=False)
            raise ManifestConflictError(
                f"{target} already exists with a DIFFERENT manifest "
                f"(existing content_hash={existing_hash[:12]}, "
                f"new={new_hash[:12]}); refusing to overwrite a completed "
                f"output directory. Differing fields: {sorted(diff)}"
            )
        if not allow_overwrite_if_same:
            raise ManifestConflictError(
                f"{target} already exists (content-identical) and "
                "allow_overwrite_if_same=False refuses any pre-existing "
                "manifest at this path"
            )
        # identical content: fall through and rewrite (idempotent no-op).

    if target.suffix in (".yaml", ".yml"):
        dump_yaml(manifest, target)
    else:
        dump_json(manifest, target)
    return target


# -------------------------------------------------------------------- diff --

def diff_dicts(a: dict, b: dict, fields: "list[str] | None" = None) -> dict:
    """Field-name -> (value_in_a, value_in_b) for every field in `fields`
    (or the union of both dicts' keys if None) whose canonical hash differs.
    A missing key is compared against the sentinel "<missing>" so an added/
    removed field shows up as a difference rather than being skipped.
    """
    keys = fields if fields is not None else sorted(set(a) | set(b))
    out = {}
    for k in keys:
        va = a.get(k, "<missing>")
        vb = b.get(k, "<missing>")
        if canonical_hash(va) != canonical_hash(vb):
            out[k] = (va, vb)
    return out


def diff_manifests(a: ManifestT, b: ManifestT, *, frozen_only: bool = True) -> dict:
    """Highlight which fields differ between two manifests.

    frozen_only=True (default) restricts the comparison to
    schema.FROZEN_ROLLOUT_FIELDS when both `a` and `b` are RolloutManifests
    -- this is implementation step 6's "manifest diff CLI highlighting only
    declared independent variables between methods": two rollouts being
    compared as reconstructed-vs-reference (mujoco_paired) should differ in
    outcome fields (success, staged_progress) but must NOT differ in any
    declared frozen field, and this is the function that check catches.
    frozen_only=False (or either manifest not a RolloutManifest) diffs every
    field.
    """
    da, db = to_dict(a), to_dict(b)
    fields = None
    if frozen_only and isinstance(a, RolloutManifest) and isinstance(b, RolloutManifest):
        fields = FROZEN_ROLLOUT_FIELDS
    return diff_dicts(da, db, fields=fields)


# --------------------------------------------------------------------- CLI --

def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        manifest = load_any(args.path)
    except Exception as e:  # noqa: BLE001 -- CLI boundary, report and exit
        print(f"FAIL: could not load/validate {args.path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"OK: {args.path} is a valid {manifest.manifest_kind!r} manifest "
          f"(schema_version={manifest.schema_version})")
    print(f"content_hash={manifest_content_hash(manifest)}")

    issues = find_semantic_issues(to_dict(manifest))
    if issues:
        print(f"{len(issues)} semantic warning(s) (non-fatal):")
        for i in issues:
            print(f"  - {i}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    a = load_any(args.manifest_a)
    b = load_any(args.manifest_b)
    d = diff_manifests(a, b, frozen_only=not args.all_fields)
    scope = "all fields" if args.all_fields else "declared frozen fields only"
    if not d:
        print(f"no differences ({scope})")
        return 0
    print(f"{len(d)} differing field(s) ({scope}):")
    for k in sorted(d):
        va, vb = d[k]
        print(f"  {k}: {va!r} != {vb!r}")
    return 1 if (not args.all_fields) else 0  # frozen-field drift is a contract violation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robo.manifest.io")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="load + schema-validate a manifest file")
    p_validate.add_argument("path")

    p_diff = sub.add_parser("diff", help="diff two manifest files")
    p_diff.add_argument("manifest_a")
    p_diff.add_argument("manifest_b")
    p_diff.add_argument("--all-fields", action="store_true",
                         help="diff every field instead of just the "
                              "declared frozen-field set")

    args = ap.parse_args(argv)
    if args.cmd == "validate":
        return _cmd_validate(args)
    return _cmd_diff(args)


if __name__ == "__main__":
    raise SystemExit(main())
