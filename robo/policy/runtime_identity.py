"""Fail-closed runtime identity for an OpenPI policy server.

The checkpoint registry proves which bytes *should* be served.  This module
connects that declaration to the process which actually answers inference
requests: the server publishes one self-authenticating identity document in
both ``/healthz`` and the OpenPI websocket handshake, and the client verifies
it before the first observation can be sent.

Checkpoint hashes use :func:`robo.manifest.hash.hash_checkpoint_path`; they
are conservative path/size/mtime fingerprints, not content hashes.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from robo.manifest.hash import canonical_hash, hash_checkpoint_path

IDENTITY_SCHEMA_VERSION = 1
IDENTITY_KIND = "simany_bound_openpi_server"
IDENTITY_METADATA_KEY = "simany_policy_runtime_identity"
CHECKPOINT_FINGERPRINT_KIND = "tree_path_size_mtime_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")


class PolicyRuntimeIdentityError(RuntimeError):
    """A declared, local, or server-reported policy identity disagrees."""


def _sha256(value: Any, owner: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise PolicyRuntimeIdentityError(
            f"{owner} must be a lowercase 64-character sha256 value")
    return normalized


def _run_git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        detail = str(output).strip() or str(exc)
        raise PolicyRuntimeIdentityError(
            f"cannot inspect git checkout {root}: {detail}") from exc


def verify_clean_git_checkout(
    root: str | Path, expected_commit: str, *, owner: str,
) -> dict[str, Any]:
    """Return a normalized identity for one exact, clean Git worktree."""
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise PolicyRuntimeIdentityError(f"{owner}.root must be absolute")
    path = path.resolve(strict=False)
    if not path.is_dir():
        raise PolicyRuntimeIdentityError(f"{owner}.root is not a directory: {path}")
    expected = str(expected_commit).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(expected):
        raise PolicyRuntimeIdentityError(
            f"{owner}.commit must be a full 40- or 64-character Git object id")
    top = Path(_run_git(path, "rev-parse", "--show-toplevel")).resolve()
    if top != path:
        raise PolicyRuntimeIdentityError(
            f"{owner}.root is not the Git worktree root: declared={path}, git={top}")
    actual = _run_git(path, "rev-parse", "HEAD").lower()
    if actual != expected:
        raise PolicyRuntimeIdentityError(
            f"{owner}.commit differs: declared={expected}, current={actual}")
    status = _run_git(path, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise PolicyRuntimeIdentityError(
            f"{owner} worktree is dirty: {status.splitlines()[:10]}")
    return {"root": str(path), "commit": actual, "dirty": False}


def verify_checkpoint(
    checkpoint_path: str | Path, declared_fingerprint: str, *, owner: str = "policy",
) -> dict[str, Any]:
    """Resolve and compare the checkpoint currently visible on this host."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        raise PolicyRuntimeIdentityError(f"{owner}.checkpoint_path must be absolute")
    path = path.resolve(strict=False)
    if not path.exists():
        raise PolicyRuntimeIdentityError(f"{owner} checkpoint does not exist: {path}")
    declared = _sha256(declared_fingerprint, f"{owner}.checkpoint_hash")
    try:
        actual = hash_checkpoint_path(path)
    except (OSError, ValueError) as exc:
        raise PolicyRuntimeIdentityError(
            f"cannot fingerprint {owner} checkpoint {path}: {exc}") from exc
    if actual != declared:
        raise PolicyRuntimeIdentityError(
            f"declared {owner} checkpoint hash differs from the checkpoint "
            f"visible now: declared={declared}, current={actual}, path={path}")
    return {
        "path": str(path),
        "fingerprint": actual,
        "fingerprint_kind": CHECKPOINT_FINGERPRINT_KIND,
    }


def runtime_checkpoint_path(
    cache_root: str | Path, policy_id: str, checkpoint_fingerprint: str,
) -> Path:
    """Return the deterministic cache path for one sealed checkpoint."""
    root = Path(cache_root).expanduser()
    if not root.is_absolute():
        raise PolicyRuntimeIdentityError("checkpoint_cache_root must be absolute")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", policy_id):
        raise PolicyRuntimeIdentityError("policy id is unsafe for a cache path")
    fingerprint = _sha256(checkpoint_fingerprint, "policy.checkpoint_hash")
    return root.resolve(strict=False) / policy_id / fingerprint / "checkpoint"


def stage_checkpoint(
    *, source_path: str | Path, checkpoint_fingerprint: str,
    cache_root: str | Path, policy_id: str,
) -> Path:
    """Copy a checkpoint into the cache root with locking and full verification.

    A receipt or sentinel is never trusted.  Every reuse fingerprints the
    complete destination.  A corrupt prior destination is moved aside for
    inspection; staging uses a unique sibling followed by an atomic rename.
    """
    source = Path(source_path).expanduser().resolve(strict=False)
    verify_checkpoint(source, checkpoint_fingerprint, owner="declared source")
    destination = runtime_checkpoint_path(
        cache_root, policy_id, checkpoint_fingerprint)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / ".stage.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            try:
                verify_checkpoint(
                    destination, checkpoint_fingerprint, owner="runtime cache")
            except PolicyRuntimeIdentityError:
                quarantine = parent / (
                    f"checkpoint.invalid.{time.time_ns()}.{os.getpid()}")
                destination.rename(quarantine)
            else:
                return destination
        temporary = Path(tempfile.mkdtemp(
            prefix="checkpoint.partial.", dir=parent))
        try:
            try:
                subprocess.run(
                    ["rsync", "-a", f"{source}/", f"{temporary}/"],
                    check=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise PolicyRuntimeIdentityError(
                    f"checkpoint rsync to {temporary} failed: {exc}") from exc
            verify_checkpoint(
                temporary, checkpoint_fingerprint, owner="staged runtime")
            temporary.rename(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        verify_checkpoint(
            destination, checkpoint_fingerprint, owner="runtime cache")
        return destination


def build_server_identity(
    *,
    policy_id: str,
    checkpoint_path: str | Path,
    checkpoint_fingerprint: str,
    training_config: str,
    openpi_root: str | Path,
    openpi_commit: str,
    port: int,
    runtime_checkpoint_path_value: str | Path | None = None,
    verify_runtime_checkpoint_now: bool = True,
    sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact identity a bound OpenPI server must publish."""
    if not isinstance(policy_id, str) or not policy_id:
        raise PolicyRuntimeIdentityError("policy.id must be a non-empty string")
    if not isinstance(training_config, str) or not training_config:
        raise PolicyRuntimeIdentityError(
            "policy.training_config must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise PolicyRuntimeIdentityError("server port must be in [1, 65535]")
    source = verify_checkpoint(
        checkpoint_path, checkpoint_fingerprint, owner="declared source")
    runtime_path_value = Path(
        runtime_checkpoint_path_value or checkpoint_path).expanduser()
    if not runtime_path_value.is_absolute():
        raise PolicyRuntimeIdentityError("runtime.checkpoint_path must be absolute")
    runtime_path_value = runtime_path_value.resolve(strict=False)
    if verify_runtime_checkpoint_now:
        runtime = verify_checkpoint(
            runtime_path_value, checkpoint_fingerprint, owner="runtime")
    else:
        runtime = {
            "path": str(runtime_path_value),
            "fingerprint": source["fingerprint"],
        }
    checkpoint = {
        "declared_source_path": source["path"],
        "runtime_path": runtime["path"],
        "fingerprint": source["fingerprint"],
        "fingerprint_kind": CHECKPOINT_FINGERPRINT_KIND,
    }
    openpi = verify_clean_git_checkout(
        openpi_root, openpi_commit, owner="runtime_dependencies.openpi")
    payload: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "kind": IDENTITY_KIND,
        "policy": {
            "id": policy_id,
            "checkpoint": checkpoint,
            "training_config": training_config,
        },
        "openpi": openpi,
        "server": {"protocol": "openpi_websocket", "port": port},
    }
    if sampling is not None:
        from robo.policy.sampling_contract import validate_contract
        payload["policy"]["sampling"] = validate_contract(sampling, training_config)
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def expected_server_identity_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the expected server identity from a serving contract config."""
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise PolicyRuntimeIdentityError("harness contract is missing")
    policy = contract.get("policy")
    dependencies = contract.get("runtime_dependencies")
    if not isinstance(policy, Mapping):
        raise PolicyRuntimeIdentityError("contract.policy is missing")
    if not isinstance(dependencies, Mapping):
        raise PolicyRuntimeIdentityError(
            "contract.runtime_dependencies is missing")
    openpi = dependencies.get("openpi")
    if not isinstance(openpi, Mapping):
        raise PolicyRuntimeIdentityError(
            "contract.runtime_dependencies.openpi is missing")
    cache_root = dependencies.get("checkpoint_cache_root")
    runtime_path = None
    if cache_root:
        runtime_path = runtime_checkpoint_path(
            str(cache_root), str(policy.get("id", "")),
            str(policy.get("checkpoint_hash", "")))
    return build_server_identity(
        policy_id=str(policy.get("id", "")),
        checkpoint_path=str(policy.get("checkpoint_path", "")),
        checkpoint_fingerprint=str(policy.get("checkpoint_hash", "")),
        training_config=str(policy.get("training_config", "")),
        openpi_root=str(openpi.get("root", "")),
        openpi_commit=str(openpi.get("commit", "")),
        port=int(config.get("port", 8000)),
        runtime_checkpoint_path_value=runtime_path,
        verify_runtime_checkpoint_now=False,
        sampling=policy.get("sampling"),
    )


def validate_server_identity(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
    expected_policy: Mapping[str, Any] | None = None,
    verify_runtime_checkpoint_now: bool = True,
) -> dict[str, Any]:
    """Validate a websocket/health identity and return an isolated copy.

    ``expected`` is an exact full identity generated from the serving contract.
    ``expected_policy`` is the registry-level subset used by legacy callers;
    it still prevents a server for another checkpoint or training config from
    being accepted.
    """
    if not isinstance(value, Mapping):
        raise PolicyRuntimeIdentityError("server runtime identity is missing")
    identity = copy.deepcopy(dict(value))
    required = {
        "schema_version", "kind", "policy", "openpi", "server",
        "identity_sha256",
    }
    if set(identity) != required:
        raise PolicyRuntimeIdentityError(
            "server runtime identity schema differs: "
            f"expected={sorted(required)}, current={sorted(identity)}")
    if identity["schema_version"] != IDENTITY_SCHEMA_VERSION:
        raise PolicyRuntimeIdentityError("server runtime identity version differs")
    if identity["kind"] != IDENTITY_KIND:
        raise PolicyRuntimeIdentityError("server runtime identity kind differs")
    recorded_hash = _sha256(
        identity.pop("identity_sha256"), "server identity_sha256")
    actual_hash = canonical_hash(identity)
    identity["identity_sha256"] = recorded_hash
    if recorded_hash != actual_hash:
        raise PolicyRuntimeIdentityError(
            "server runtime identity_sha256 does not authenticate its payload")

    policy = identity.get("policy")
    required_policy = {"id", "checkpoint", "training_config"}
    if isinstance(policy, Mapping) and "sampling" in policy:
        from robo.policy.sampling_contract import validate_contract
        try:
            validate_contract(policy["sampling"], policy.get("training_config", ""))
        except (TypeError, ValueError) as exc:
            raise PolicyRuntimeIdentityError(str(exc)) from exc
        required_policy.add("sampling")
    if not isinstance(policy, Mapping) or set(policy) != required_policy:
        raise PolicyRuntimeIdentityError("server policy identity schema differs")
    checkpoint = policy.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "declared_source_path", "runtime_path", "fingerprint",
        "fingerprint_kind",
    }:
        raise PolicyRuntimeIdentityError("server checkpoint identity schema differs")
    if checkpoint.get("fingerprint_kind") != CHECKPOINT_FINGERPRINT_KIND:
        raise PolicyRuntimeIdentityError("server checkpoint fingerprint kind differs")
    verify_checkpoint(
        str(checkpoint.get("declared_source_path", "")),
        str(checkpoint.get("fingerprint", "")), owner="server declared source")
    if verify_runtime_checkpoint_now:
        verify_checkpoint(
            str(checkpoint.get("runtime_path", "")),
            str(checkpoint.get("fingerprint", "")), owner="server runtime")

    openpi = identity.get("openpi")
    if not isinstance(openpi, Mapping) or set(openpi) != {
        "root", "commit", "dirty",
    }:
        raise PolicyRuntimeIdentityError("server OpenPI identity schema differs")
    if openpi.get("dirty") is not False:
        raise PolicyRuntimeIdentityError("server reports a dirty OpenPI checkout")
    verify_clean_git_checkout(
        str(openpi.get("root", "")), str(openpi.get("commit", "")),
        owner="server.openpi",
    )
    server = identity.get("server")
    if not isinstance(server, Mapping) or set(server) != {"protocol", "port"}:
        raise PolicyRuntimeIdentityError("server transport identity schema differs")
    if server.get("protocol") != "openpi_websocket":
        raise PolicyRuntimeIdentityError("server protocol differs")

    if expected is not None and identity != dict(expected):
        raise PolicyRuntimeIdentityError(
            "server runtime identity differs from the exact harness declaration")
    if expected_policy is not None:
        comparisons = {
            "id": policy.get("id"),
            "checkpoint_path": checkpoint.get("declared_source_path"),
            "checkpoint_fingerprint": checkpoint.get("fingerprint"),
            "training_config": policy.get("training_config"),
        }
        for field, actual in comparisons.items():
            declared = expected_policy.get(field)
            if declared != actual:
                raise PolicyRuntimeIdentityError(
                    f"server policy {field} differs: declared={declared!r}, "
                    f"server={actual!r}")
    return identity


def identity_from_server_metadata(
    metadata: Any,
    *,
    expected: Mapping[str, Any] | None = None,
    expected_policy: Mapping[str, Any] | None = None,
    verify_runtime_checkpoint_now: bool = True,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise PolicyRuntimeIdentityError("OpenPI websocket metadata is missing")
    if IDENTITY_METADATA_KEY not in metadata:
        raise PolicyRuntimeIdentityError(
            f"OpenPI websocket metadata lacks {IDENTITY_METADATA_KEY!r}")
    return validate_server_identity(
        metadata[IDENTITY_METADATA_KEY], expected=expected,
        expected_policy=expected_policy,
        verify_runtime_checkpoint_now=verify_runtime_checkpoint_now)


def health_document(identity: Mapping[str, Any]) -> str:
    """Canonical JSON body returned by the bound server's ``/healthz``."""
    validated = validate_server_identity(identity, expected=identity)
    return json.dumps(
        {"status": "ok", "identity": validated},
        sort_keys=True, separators=(",", ":"),
    ) + "\n"


__all__ = [
    "CHECKPOINT_FINGERPRINT_KIND",
    "IDENTITY_KIND",
    "IDENTITY_METADATA_KEY",
    "IDENTITY_SCHEMA_VERSION",
    "PolicyRuntimeIdentityError",
    "build_server_identity",
    "expected_server_identity_from_config",
    "health_document",
    "identity_from_server_metadata",
    "runtime_checkpoint_path",
    "stage_checkpoint",
    "validate_server_identity",
    "verify_checkpoint",
    "verify_clean_git_checkout",
]
