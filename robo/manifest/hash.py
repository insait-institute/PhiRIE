"""Canonicalization, hashing, and provenance-snapshot helpers for robo.manifest.

Hashing contract: every hash in this module is sha256 over a canonical JSON
encoding (sorted keys, compact separators, NaN/Infinity rejected) so the same
semantic content hashes identically regardless of key order or float-repr
noise picked up from a YAML<->JSON round trip. Kept as a reusable module
because the manifests need several hash "flavors"
(whole-manifest content hash, config-subsection hash, checkpoint fingerprint).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"object of type {type(o)!r} is not JSON-canonicalizable: {o!r}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Sorted-key, compact JSON encoding.

    `allow_nan=False` makes a NaN/Infinity anywhere in `obj` raise
    ValueError at encode time ("reject NaN") instead of
    silently emitting the non-standard `NaN`/`Infinity` JSON tokens that
    `json.dumps` produces by default.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_hash(obj: Any) -> str:
    """sha256 hex digest of `obj`'s canonical JSON encoding.

    Two objects that are equal as Python data (dict key order and float
    formatting aside) always hash identically; this is the single hashing
    primitive every other helper in this module and in schema.py builds on.
    """
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def manifest_content_hash(manifest, *, exclude: set[str] | None = None) -> str:
    """Content-address a manifest model.

    canonical_hash() of the manifest's dict form with volatile bookkeeping
    fields excluded (by default just `created_utc`), so two manifests
    describing the exact same build/rollout hash identically regardless of
    the wall-clock instant they were written -- this is what `write_manifest`
    compares to decide whether a rerun is an idempotent no-op or a real
    conflict, and what an "example_manifest.yaml" hash pins for the frozen
    fields it declares.
    """
    exclude = {"created_utc"} if exclude is None else exclude
    data = manifest.model_dump(mode="json", exclude=exclude)
    return canonical_hash(data)


def git_snapshot(root: "str | os.PathLike | None" = None) -> dict:
    """{'commit': full sha (or 'nogit'), 'dirty': bool, 'branch': str|None}.

    Never raises: any git failure (not a repo, git binary missing, detached
    weirdness) degrades to the 'nogit' sentinel used elsewhere in this repo
    (`agents/eval/validate_contract.py:_git_sha`) so manifest construction
    never crashes an eval run just because git introspection failed.
    """
    def _run(args):
        return subprocess.check_output(
            ["git"] + (["-C", str(root)] if root is not None else []) + args,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = _run(["rev-parse", "HEAD"])
    except Exception:
        return {"commit": "nogit", "dirty": False, "branch": None}
    try:
        dirty = bool(_run(["status", "--porcelain"]).strip())
    except Exception:
        dirty = False
    try:
        branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        branch = None
    return {"commit": commit, "dirty": dirty, "branch": branch}


def hash_checkpoint_path(path: "str | os.PathLike") -> str:
    """Fingerprint a policy checkpoint file or directory WITHOUT reading its
    bytes.

    LIMITATION (deliberate: hashing multi-GB checkpoint bytes directly is
    too slow): this hashes the sorted list of `(relative_path, size_bytes,
    mtime_ns)` tuples for every file under `path` (or just the one file),
    NOT the file contents. Consequences:
      - a byte-identical rewrite that happens to preserve both size AND
        mtime is invisible to this hash -- in practice checkpoints are
        written once by a training/export job and never edited in place,
        so this is a reliable "did the checkpoint on disk change" signal
        for this pipeline, but it is not a cryptographic content hash and
        must not be described as one.
      - touching/replacing a checkpoint file (even with identical bytes)
        changes mtime and therefore the hash, which is the intended
        conservative direction (false positives on "changed" are safe;
        false negatives would silently invalidate provenance).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {p}")
    entries = []
    if p.is_file():
        st = p.stat()
        entries.append((p.name, st.st_size, st.st_mtime_ns))
    else:
        for f in sorted(p.rglob("*")):
            if f.is_file():
                st = f.stat()
                entries.append(
                    (str(f.relative_to(p).as_posix()), st.st_size, st.st_mtime_ns)
                )
        if not entries:
            raise FileNotFoundError(f"checkpoint directory has no files: {p}")
    return canonical_hash(entries)


def hash_config_section(section: Any) -> str:
    """Semantic alias of canonical_hash for a config sub-tree (e.g. the
    'control' or 'cameras' block of configs/policies/frozen_fields.yaml).

    Kept as a separate name (rather than call sites doing
    `canonical_hash(frozen["control"])` directly) so `controller_config_hash`
    / `camera_config_hash` construction reads as intent at the call site.
    """
    return canonical_hash(section)


# Prefixes that make a path meaningless off the machine it was written on.
# Not exhaustive, but covers every path convention actually used in this
# repo (see agents/core/common.py, run/env.sh).
_MACHINE_PATH_PREFIXES = ("/group/", "/data/", "/home/", "/mnt/", "/tmp/")


def find_semantic_issues(obj: Any, _path: str = "$") -> list[str]:
    """Recursively walk `obj` (typically a manifest's `model_dump()` dict)
    and collect human-readable warnings for the non-hash-breaking hazards
    worth flagging:

    - non-finite floats: canonical_hash() already hard-rejects these
      (raises ValueError from `json.dumps(allow_nan=False)`); this walk runs
      first so `validate` can report exactly WHERE before that happens.
    - absolute machine-specific paths: legal (checkpoint_path in
      frozen_fields.yaml is necessarily absolute) but
      flagged as a warning, since a manifest containing them cannot be
      replayed as-is on a different mount layout.
    """
    issues: list[str] = []
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            issues.append(f"{_path}: non-finite float {obj!r}")
    elif isinstance(obj, str):
        if obj.startswith(_MACHINE_PATH_PREFIXES):
            issues.append(f"{_path}: absolute machine-specific path {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            issues.extend(find_semantic_issues(v, f"{_path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            issues.extend(find_semantic_issues(v, f"{_path}[{i}]"))
    return issues
