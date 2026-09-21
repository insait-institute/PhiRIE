"""Create an immutable, self-contained experiment-contract snapshot.

The freeze command is intentionally CPU-only.  It resolves and copies the
seven ICRA configuration files, records source/resource fingerprints and the
runtime environment, then publishes the completed directory with one rename.
No partially written contract is visible at the requested output path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from robo.manifest.hash import canonical_hash, git_snapshot, hash_checkpoint_path


CONFIG_FIELDS = (
    "construction_config",
    "fidelity_manifest",
    "harness_config",
    "audit_config",
    "harmony_manifest",
    "real_world_config",
)
REQUIRED_FIELDS = (
    "freeze_id",
    "code_commit",
    "git_dirty",
    "created_utc",
    "paper_repository",
    "paper_commit_before_update",
    "input_roots",
    "checkpoint_roots",
    "hardware",
    *CONFIG_FIELDS,
)


class FreezeError(RuntimeError):
    """The requested contract cannot be frozen without weakening provenance."""


CANONICAL_FREEZE_ID = re.compile(r"[0-9]{8}-[0-9a-f]{7}-v[1-9][0-9]*")


def reserve_freeze_id(repo_root: Path, output_root: Path, *,
                      now_utc: str | None = None, dry_run: bool = False) -> str:
    """Allocate a canonical ID; durable reservations survive failed runs.

    The main SHA names the project baseline; the manifest separately records
    the actual source SHA. Exclusive mkdir serializes simultaneous allocators.
    Existing experiment directories and consumed IDs are never reused.
    """
    output_root = output_root.resolve()
    _assert_not_tmp(output_root, "freeze output root")
    try:
        main_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "refs/heads/main"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise FreezeError("automatic freeze ID requires a local main branch") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", main_sha):
        raise FreezeError("main branch has no valid 40-character commit SHA")
    when = datetime.fromisoformat((now_utc or _now_utc()).replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise FreezeError("automatic freeze ID timestamp must have a timezone")
    prefix = f"{when.astimezone(timezone.utc):%Y%m%d}-{main_sha[:7]}-v"
    reservations = output_root / ".freeze_ids"
    if not dry_run:
        reservations.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        freeze_id = f"{prefix}{version}"
        reservation = reservations / freeze_id
        if (output_root / freeze_id).exists() or reservation.exists():
            version += 1
            continue
        if dry_run:
            return freeze_id
        try:
            reservation.mkdir()
        except FileExistsError:
            version += 1
            continue
        # A concurrent legacy writer may have used the ID after the first
        # existence check. Keep this tombstone and allocate another ID.
        if (output_root / freeze_id).exists():
            version += 1
            continue
        return freeze_id


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_structured(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FreezeError(f"cannot read config {path}: {exc}") from exc
    try:
        return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise FreezeError(f"invalid structured config {path}: {exc}") from exc


def _resolve_path(value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _under_tmp(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == Path("/tmp") or Path("/tmp") in resolved.parents


def _assert_not_tmp(path: Path, label: str) -> None:
    if _under_tmp(path):
        raise FreezeError(f"{label} resolves under forbidden /tmp: {path}")


def _normalize_resource_entries(value: Any, *, category: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        normalized = []
        for resource_id, resource_value in value.items():
            if isinstance(resource_value, str):
                normalized.append({"id": str(resource_id), "path": resource_value})
            elif isinstance(resource_value, dict):
                normalized.append({"id": str(resource_id), **resource_value})
            else:
                raise FreezeError(
                    f"{category}.{resource_id} must be a path or mapping")
        value = normalized
    if not isinstance(value, list):
        raise FreezeError(f"{category} must be a list or mapping")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            item = {"id": f"{category}-{index:02d}", "path": item}
        if not isinstance(item, dict) or not item.get("path"):
            raise FreezeError(f"{category}[{index}] must contain a non-empty path")
        entry = dict(item)
        entry.setdefault("id", f"{category}-{index:02d}")
        entry.setdefault("kind", "checkpoint" if category == "checkpoint_roots" else "metadata")
        entry.setdefault("required", True)
        if not isinstance(entry["required"], bool):
            raise FreezeError(f"{category}[{index}].required must be boolean")
        entries.append(entry)
    ids = [str(entry["id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise FreezeError(f"{category} resource ids must be unique")
    return entries


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _directory_size(path: Path) -> int:
    """Return total regular-file bytes without following directory symlinks."""
    total = 0
    for child in path.rglob("*"):
        resolved = child.resolve(strict=False)
        _assert_not_tmp(resolved, f"resource child {child}")
        if child.is_file():
            total += child.stat().st_size
    return total


def _inventory_resource(entry: dict[str, Any], *, category: str,
                        repo_root: Path) -> dict[str, Any]:
    declared = str(entry["path"])
    path = _resolve_path(declared, repo_root)
    _assert_not_tmp(path, f"resource {entry['id']!r}")
    result: dict[str, Any] = {
        "category": category,
        "id": str(entry["id"]),
        "declared_path": declared,
        "resolved_path": str(path),
        "kind": str(entry.get("kind", "metadata")),
        "required": bool(entry.get("required", True)),
        "exists": path.exists(),
        "type": "missing",
        "size_bytes": None,
        "sha256": None,
        "hash_method": None,
    }
    if not path.exists():
        if result["required"]:
            raise FreezeError(f"required resource does not exist: {path}")
        return result
    if path.is_file():
        result["type"] = "file"
        result["size_bytes"] = _file_size(path)
        if result["kind"] in {"checkpoint", "large_directory"}:
            result["sha256"] = hash_checkpoint_path(path)
            result["hash_method"] = "path_size_mtime_sha256"
        else:
            result["sha256"] = _sha256_bytes(path.read_bytes())
            result["hash_method"] = "content_sha256"
    elif path.is_dir():
        result["type"] = "directory"
        # The existing checkpoint helper deliberately hashes metadata rather
        # than reading potentially multi-GB tensors.
        result["size_bytes"] = _directory_size(path)
        result["sha256"] = hash_checkpoint_path(path)
        result["hash_method"] = "tree_path_size_mtime_sha256"
    else:
        raise FreezeError(f"resource is neither a file nor directory: {path}")
    return result


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "installed-version-unknown"


def _driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
    return ",".join(versions) or None


def _environment_snapshot() -> dict[str, Any]:
    torch_version = _module_version("torch")
    cuda_version = None
    if torch_version is not None:
        try:
            import torch

            cuda_version = torch.version.cuda
        except Exception:
            cuda_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch_version,
        "cuda": cuda_version,
        "nvidia_driver": _driver_version(),
        "mujoco": _module_version("mujoco"),
    }


def _config_record(field: str, path: Path, destination_name: str) -> dict[str, Any]:
    value = _load_structured(path)
    if value is None:
        raise FreezeError(f"config is empty: {path}")
    raw = path.read_bytes()
    return {
        "field": field,
        "declared_path": None,
        "resolved_path": str(path),
        "frozen_copy": f"resolved_configs/{destination_name}",
        "size_bytes": len(raw),
        "sha256": canonical_hash(value),
        "hash_method": "canonical_structured_sha256",
        "source_content_sha256": _sha256_bytes(raw),
        "_source": path,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_hashes_csv(path: Path, inventory: Iterable[dict[str, Any]]) -> None:
    columns = (
        "category", "id", "declared_path", "resolved_path", "type",
        "size_bytes", "sha256", "hash_method",
    )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in inventory:
            writer.writerow({key: record.get(key) for key in columns})
        handle.flush()
        os.fsync(handle.fileno())


def _report(manifest: dict[str, Any]) -> str:
    missing = sum(not item["exists"] for item in manifest["resource_inventory"])
    return "\n".join((
        f"freeze_id: {manifest['freeze_id']}",
        f"created_utc: {manifest['created_utc']}",
        f"code: {manifest['code']['commit']} ({manifest['code']['branch']})",
        f"dirty_worktree: {str(manifest['code']['dirty']).lower()}",
        f"paper_commit_before_update: {manifest['paper']['commit_before_update']}",
        f"configs: {len(manifest['configs'])}",
        f"resources: {len(manifest['resource_inventory'])} ({missing} missing optional)",
        f"contract_sha256: {manifest['contract_sha256']}",
        "status: PASS",
        "",
    ))


def _validate_root_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise FreezeError("freeze config must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise FreezeError(f"freeze config is missing required fields: {missing}")
    if config["git_dirty"] is not False:
        raise FreezeError("freeze config must declare git_dirty: false")
    if not isinstance(config["hardware"], dict):
        raise FreezeError("hardware must be a mapping")
    if str(config.get("mode", "paper")) not in {"smoke", "paper"}:
        raise FreezeError("mode must be 'smoke' or 'paper'")
    return dict(config)


def create_freeze(
    config_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    allow_dirty_for_smoke: bool = False,
    dry_run: bool = False,
    preflight_log: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
    now_utc: str | None = None,
    _inject_failure_after: str | None = None,
    auto_freeze_id: bool = False,
) -> dict[str, Any]:
    """Resolve, validate, and optionally publish one freeze directory."""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    config_source = _resolve_path(config_path, root)
    output = _resolve_path(out_dir, root)
    _assert_not_tmp(config_source, "freeze config")
    _assert_not_tmp(output, "freeze output")
    if output.exists() and not auto_freeze_id:
        raise FreezeError(f"refusing to overwrite existing freeze directory: {output}")

    config = _validate_root_config(_load_structured(config_source))
    mode = str(config.get("mode", "paper"))
    code = git_snapshot(root)
    if code["commit"] == "nogit":
        raise FreezeError(f"repository has no readable git commit: {root}")
    if code["dirty"] and not (allow_dirty_for_smoke and mode == "smoke"):
        raise FreezeError(
            "working tree is dirty; commit/stash changes or use "
            "--allow-dirty-for-smoke with a smoke-mode config"
        )
    if auto_freeze_id:
        config["freeze_id"] = reserve_freeze_id(
            root, output, now_utc=now_utc, dry_run=dry_run)
        output = output / config["freeze_id"] / "contract"
    if mode == "paper" and not CANONICAL_FREEZE_ID.fullmatch(str(config["freeze_id"])):
        raise FreezeError("paper freeze_id must be YYYYMMDD-<main-short-sha>-v<integer>")
    declared_commit = str(config["code_commit"])
    if declared_commit != "auto" and not (
        declared_commit.startswith(str(code["commit"]))
        or str(code["commit"]).startswith(declared_commit)
    ):
        raise FreezeError(
            f"code_commit mismatch: config={declared_commit!r}, checkout={code['commit']!r}"
        )

    paper_path = _resolve_path(config["paper_repository"], root)
    _assert_not_tmp(paper_path, "paper_repository")
    paper = git_snapshot(paper_path)
    declared_paper_commit = str(config["paper_commit_before_update"])
    if declared_paper_commit != "auto" and not (
        declared_paper_commit.startswith(str(paper["commit"]))
        or str(paper["commit"]).startswith(declared_paper_commit)
    ):
        raise FreezeError(
            "paper_commit_before_update mismatch: "
            f"config={declared_paper_commit!r}, checkout={paper['commit']!r}"
        )
    if mode == "paper" and paper["commit"] == "nogit":
        raise FreezeError(f"paper_repository has no readable git commit: {paper_path}")

    config_records: list[dict[str, Any]] = []
    root_record = _config_record("freeze_config", config_source, "freeze.yaml")
    root_record["declared_path"] = str(config_path)
    config_records.append(root_record)
    destination_names: set[str] = {"freeze.yaml"}
    for field in CONFIG_FIELDS:
        source = _resolve_path(config[field], root)
        _assert_not_tmp(source, field)
        if not source.is_file():
            raise FreezeError(f"referenced config is not a file: {field}={source}")
        destination_name = source.name
        if destination_name in destination_names:
            destination_name = f"{field}-{destination_name}"
        destination_names.add(destination_name)
        record = _config_record(field, source, destination_name)
        record["declared_path"] = str(config[field])
        config_records.append(record)

    harness_record = next(record for record in config_records
                          if record["field"] == "harness_config")
    harness_value = _load_structured(harness_record["_source"])
    if mode == "paper" or (
        isinstance(harness_value, dict)
        and ("contract" in harness_value or "treatments" in harness_value)
    ):
        if mode == "paper" and not bool(harness_value.get("paper_mode", False)):
            raise FreezeError("paper freeze requires harness_config paper_mode: true")
        try:
            from robo.eval.harness_spec import load_harness_spec

            load_harness_spec(harness_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise FreezeError(f"invalid harness_config: {exc}") from exc

    input_entries = _normalize_resource_entries(config["input_roots"], category="input_roots")
    checkpoint_entries = _normalize_resource_entries(
        config["checkpoint_roots"], category="checkpoint_roots"
    )
    if mode == "paper" and not checkpoint_entries:
        raise FreezeError("paper-mode freeze requires at least one checkpoint_roots entry")
    inventory = [
        _inventory_resource(entry, category=category, repo_root=root)
        for category, entries in (
            ("input_roots", input_entries),
            ("checkpoint_roots", checkpoint_entries),
        )
        for entry in entries
    ]
    # Referenced configs are resources too; include their semantic hashes in
    # resource_inventory.json without re-reading or duplicating traversal.
    inventory.extend({
        "category": "config",
        "id": record["field"],
        "declared_path": record["declared_path"],
        "resolved_path": record["resolved_path"],
        "kind": "structured_config",
        "required": True,
        "exists": True,
        "type": "file",
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "hash_method": record["hash_method"],
    } for record in config_records)

    created = now_utc or (_now_utc() if config["created_utc"] == "auto" else str(config["created_utc"]))
    clean_configs = [{k: v for k, v in record.items() if k != "_source"}
                     for record in config_records]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "freeze_id": str(config["freeze_id"]),
        "mode": mode,
        "paper_ready": mode == "paper",
        "created_utc": created,
        "code": {
            "repository": str(root),
            "commit": code["commit"],
            "branch": code["branch"],
            "dirty": code["dirty"],
            "dirty_override_for_smoke": bool(code["dirty"] and allow_dirty_for_smoke),
        },
        "paper": {
            "repository": str(paper_path),
            "commit_before_update": paper["commit"],
            "branch": paper["branch"],
            "dirty": paper["dirty"],
        },
        "hardware_request": config["hardware"],
        "environment": _environment_snapshot(),
        "configs": clean_configs,
        "resource_inventory": inventory,
    }
    manifest["contract_sha256"] = canonical_hash({
        key: value for key, value in manifest.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    })

    if dry_run:
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    if staging.exists():  # practically impossible, but never reuse stale state
        raise FreezeError(f"staging directory unexpectedly exists: {staging}")
    try:
        (staging / "resolved_configs").mkdir(parents=True)
        for record in config_records:
            shutil.copyfile(record["_source"], staging / record["frozen_copy"])
        if _inject_failure_after == "resolved_configs":
            raise RuntimeError("injected freeze failure after resolved_configs")
        _atomic_json(staging / "resource_inventory.json", {
            "schema_version": 1,
            "freeze_id": manifest["freeze_id"],
            "resources": inventory,
        })
        _write_hashes_csv(staging / "hashes.csv", inventory)
        (staging / "freeze_report.txt").write_text(_report(manifest), encoding="utf-8")
        if preflight_log is None:
            log_text = "freeze invoked without an external preflight log\n"
        else:
            source_log = _resolve_path(preflight_log, root)
            _assert_not_tmp(source_log, "preflight_log")
            if not source_log.is_file():
                raise FreezeError(f"preflight log does not exist: {source_log}")
            log_text = source_log.read_text(encoding="utf-8")
        (staging / "preflight.log").write_text(log_text, encoding="utf-8")
        if _inject_failure_after == "inventory":
            raise RuntimeError("injected freeze failure after inventory")
        _atomic_json(staging / "freeze_manifest.json", manifest)
        if _inject_failure_after == "manifest":
            raise RuntimeError("injected freeze failure after manifest")
        # output was checked before staging and rename refuses a non-empty
        # destination on POSIX; callers must always use a new freeze path.
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="root freeze YAML")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--out", help="new contract output directory")
    output_group.add_argument("--out-root", help="experiment root for automatic ID allocation")
    parser.add_argument("--auto-freeze-id", action="store_true",
                        help="allocate UTC-date/main-short/vN; requires --out-root")
    parser.add_argument(
        "--allow-dirty-for-smoke",
        action="store_true",
        help="permit a dirty tree only when the config explicitly says mode: smoke",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print without writing")
    parser.add_argument("--preflight-log", help="repository-local log to copy into the contract")
    parser.add_argument("--repo-root", help="override repository root (primarily for tests)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.auto_freeze_id) != bool(args.out_root):
        print("--auto-freeze-id and --out-root must be supplied together", file=sys.stderr)
        return 2
    try:
        manifest = create_freeze(
            args.config,
            args.out_root if args.auto_freeze_id else args.out,
            allow_dirty_for_smoke=args.allow_dirty_for_smoke,
            dry_run=args.dry_run,
            preflight_log=args.preflight_log,
            repo_root=args.repo_root,
            auto_freeze_id=args.auto_freeze_id,
        )
    except (FreezeError, OSError, RuntimeError, ValueError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "freeze_id": manifest["freeze_id"],
            "contract_sha256": manifest["contract_sha256"],
            "configs": len(manifest["configs"]),
            "resources": len(manifest["resource_inventory"]),
        }, indent=2, sort_keys=True))
    else:
        output = _resolve_path(args.out_root if args.auto_freeze_id else args.out,
                               Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2])
        if args.auto_freeze_id:
            output = output / manifest["freeze_id"] / "contract"
        print(f"freeze created: {output}")
        print(f"contract_sha256: {manifest['contract_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
